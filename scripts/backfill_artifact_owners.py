"""
Attribute CLI-launched artifacts to an existing webapp user.

Walks ``store_root`` for every artifact kind that the read endpoints expose
(runs, comparisons, holdout evaluations, studies, top-level HPO studies) and
inserts a synthetic ``jobs`` row pointing at the named user for any artifact
that has no matching row yet. The actual artifact contents under the store
are never touched — this is a metadata-only fixup.

Idempotent. Re-running over the same tree skips artifacts that already have
an owner. ``--dry-run`` prints the planned inserts without writing anything.

Usage::

    python -m scripts.backfill_artifact_owners --user alex
    python -m scripts.backfill_artifact_owners --user alex --dry-run
    python -m scripts.backfill_artifact_owners --user alex --store-root experiment_results
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import click

from scripts._attribution import insert_synthetic_job, resolve_user_id
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.infrastructure.db import bootstrap_schema, open_db
from webapp.backend.app.infrastructure.store import (
    iter_comparison_dirs,
    iter_holdout_eval_dirs,
    iter_run_dirs,
    iter_study_dirs,
)
from webapp.backend.app.schemas.jobs import JobKind

Walker = Callable[[Path], Iterator[Path]]


@dataclass(frozen=True)
class _ArtifactKind:
    """
    One backfill target: a directory walker + the JobKind to stamp on the row.
    """

    label: str
    job_kind: JobKind
    walker: Walker


def _hpo_top_level_dirs(root: Path) -> Iterator[Path]:
    """
    Yield only the top-level HPO studies under ``<root>/hpo/<name>``.

    Nested HPO studies (under ``studies/<x>/hpo/<name>``) inherit ownership
    from the enclosing study's row, so the backfill leaves them alone.
    """

    hpo_root = root / "hpo"
    if not hpo_root.is_dir():
        return
    for child in hpo_root.iterdir():
        if child.is_dir():
            yield child


_ARTIFACT_KINDS: tuple[_ArtifactKind, ...] = (
    _ArtifactKind("run", JobKind.RUN, iter_run_dirs),
    _ArtifactKind("comparison", JobKind.COMPARE, iter_comparison_dirs),
    _ArtifactKind("holdout", JobKind.HOLDOUT, iter_holdout_eval_dirs),
    _ArtifactKind("study", JobKind.STUDY, iter_study_dirs),
    _ArtifactKind("hpo", JobKind.TUNE, _hpo_top_level_dirs),
)


def _require_user_id(conn: sqlite3.Connection, username: str) -> int:
    """
    Return the active user_id for ``username`` or raise ``click.ClickException``.
    """

    user_id = resolve_user_id(conn, username)
    if user_id is None:
        raise click.ClickException(
            f"webapp user '{username}' not found — create it first via "
            f"`python -m scripts.create_user {username}`"
        )
    return user_id


def _existing_experiment_ids(conn: sqlite3.Connection) -> set[str]:
    """
    One pass over jobs.experiment_id to skip already-owned artifacts.
    """

    rows = conn.execute("SELECT experiment_id FROM jobs WHERE experiment_id IS NOT NULL").fetchall()
    return {str(r["experiment_id"]) for r in rows}


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


@dataclass(frozen=True)
class BackfillPlan:
    """
    Planned synthetic row — emitted in dry-run, written otherwise.
    """

    kind_label: str
    job_kind: JobKind
    experiment_id: str
    timestamp_iso: str


def _walk_artifacts(
    root: Path, kinds: Iterable[_ArtifactKind]
) -> Iterator[tuple[_ArtifactKind, Path]]:
    for kind in kinds:
        for artifact_dir in kind.walker(root):
            yield kind, artifact_dir


def _plan(
    root: Path,
    *,
    existing: set[str],
    kinds: Iterable[_ArtifactKind] = _ARTIFACT_KINDS,
) -> list[BackfillPlan]:
    plans: list[BackfillPlan] = []
    seen: set[str] = set()
    for kind, artifact_dir in _walk_artifacts(root, kinds):
        experiment_id = artifact_dir.name
        # An artifact can be reached by two different walkers in pathological
        # store layouts; dedupe here so the same row isn't queued twice.
        if experiment_id in existing or experiment_id in seen:
            continue
        seen.add(experiment_id)
        plans.append(
            BackfillPlan(
                kind_label=kind.label,
                job_kind=kind.job_kind,
                experiment_id=experiment_id,
                timestamp_iso=_mtime_iso(artifact_dir),
            )
        )
    return plans


def backfill(
    conn: sqlite3.Connection,
    *,
    username: str,
    store_root: Path,
    dry_run: bool,
) -> list[BackfillPlan]:
    """
    Public entrypoint — resolves the user, builds the plan, optionally commits.

    Returns the plan regardless of ``dry_run`` so callers (and tests) can
    inspect what would change. A return list of length 0 means every
    artifact already has an owner.
    """

    user_id = _require_user_id(conn, username)
    existing = _existing_experiment_ids(conn)
    plans = _plan(store_root, existing=existing)
    if dry_run:
        return plans
    for plan in plans:
        insert_synthetic_job(
            conn,
            user_id=user_id,
            kind=plan.job_kind,
            experiment_id=plan.experiment_id,
            command="backfill",
            timestamp_iso=plan.timestamp_iso,
        )
    conn.commit()
    return plans


@click.command()
@click.option("--user", "username", required=True, help="Existing webapp username to attribute to.")
@click.option(
    "--store-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Artifact tree root (defaults to WebappSettings.store_root).",
)
@click.option("--dry-run", is_flag=True, help="Print the planned inserts without writing.")
def main(username: str, store_root: Path | None, dry_run: bool) -> None:
    settings = get_settings()
    root = store_root or settings.store_root
    if not root.is_dir():
        click.echo(f"store-root does not exist: {root}", err=True)
        sys.exit(1)
    with open_db() as conn:
        bootstrap_schema(conn)
        plans = backfill(conn, username=username, store_root=root, dry_run=dry_run)
    if not plans:
        click.echo("nothing to backfill — every artifact already has an owner")
        return
    verb = "would attribute" if dry_run else "attributed"
    click.echo(f"{verb} {len(plans)} artifact(s) to '{username}':")
    for plan in plans:
        click.echo(f"  [{plan.kind_label:11s}] {plan.experiment_id}")


if __name__ == "__main__":
    main()
