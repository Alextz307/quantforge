"""
Backfill ``.save_complete`` markers for models saved before the marker existed.

Every model/strategy ``load()`` calls ``assert_save_complete`` and refuses a
save directory that lacks the ``.save_complete`` marker - the invariant that
the producing ``save()`` ran to completion. Models persisted before the marker
was introduced have none, so they fail to load even though their directories
are intact. This walks ``store_root`` for save directories (any directory that
holds a ``metadata.json``) and, for each run still missing markers, certifies
the save by *actually loading the owning strategy* with the markers
provisionally written. A run that loads keeps its markers; a run that fails to
load has the provisional markers removed and is reported, so a genuinely
half-written save is never silently certified.

Only marker-less directories are touched, so an already-marked store is walked
cheaply (no load). The model data under the store is never modified - this
writes only the empty marker file. Idempotent.

Usage::

    python -m scripts.backfill_save_markers
    python -m scripts.backfill_save_markers --dry-run
    python -m scripts.backfill_save_markers --store-root experiment_results/studies/main
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import click

from src.core.persistence import (
    EXPERIMENT_STRATEGY_SUBDIR,
    METADATA_JSON,
    SAVE_COMPLETE_MARKER,
    mark_save_complete,
)
from src.orchestration.run_loader import load_strategy_from_run_dir

DEFAULT_STORE_ROOT = Path("experiment_results")


@dataclass(frozen=True)
class MarkerResult:
    """
    Outcome for one run that needed marker work.

    ``status`` is ``"planned"`` (dry-run), ``"backfilled"`` (markers written
    and the strategy loaded), or ``"failed"`` (load raised; markers reverted).
    """

    run_dir: Path
    status: str
    marked: int
    error: str | None = None


def _unmarked_save_dirs(state_dir: Path) -> list[Path]:
    """
    Every save directory under (and including) ``state_dir`` lacking a marker.

    A save directory is one that holds a ``metadata.json`` - the strategy's
    own ``strategy_state/`` plus each nested leaf model dir (e.g. ``garch/``,
    ``hybrid_vol/lstm/``). Leaf ``load()`` calls assert their own marker, so
    all levels must be certified together.
    """

    pending: list[Path] = []
    for candidate in (state_dir, *(d for d in state_dir.rglob("*") if d.is_dir())):
        if (candidate / METADATA_JSON).is_file() and not (
            candidate / SAVE_COMPLETE_MARKER
        ).is_file():
            pending.append(candidate)
    return pending


def backfill_run(run_dir: Path, *, dry_run: bool) -> MarkerResult | None:
    """
    Certify and mark one run's save tree; ``None`` when nothing needs doing.

    Writes provisional markers to every unmarked save dir, then loads the
    strategy as the completeness oracle. On any load failure the provisional
    markers are removed so the run is left exactly as found.
    """

    state_dir = run_dir / EXPERIMENT_STRATEGY_SUBDIR
    if not state_dir.is_dir():
        return None
    pending = _unmarked_save_dirs(state_dir)
    if not pending:
        return None
    if dry_run:
        return MarkerResult(run_dir, "planned", len(pending))
    for save_dir in pending:
        mark_save_complete(save_dir)
    try:
        load_strategy_from_run_dir(run_dir)
    except Exception as exc:  # noqa: BLE001 - any load failure means "not certified"
        for save_dir in pending:
            (save_dir / SAVE_COMPLETE_MARKER).unlink(missing_ok=True)
        return MarkerResult(run_dir, "failed", 0, error=str(exc))
    return MarkerResult(run_dir, "backfilled", len(pending))


def _iter_run_dirs(store_root: Path) -> Iterator[Path]:
    """
    Yield each run dir (the parent of a ``strategy_state/``) under the store.

    Matches both the flat ``runs/<id>/strategy_state`` and study-nested
    ``studies/<x>/runs/<id>/strategy_state`` layouts.
    """

    for state_dir in sorted(store_root.glob(f"**/{EXPERIMENT_STRATEGY_SUBDIR}")):
        if state_dir.is_dir():
            yield state_dir.parent


def backfill_store(store_root: Path, *, dry_run: bool) -> list[MarkerResult]:
    """
    Walk the store and certify every run with missing markers.

    Returns one :class:`MarkerResult` per run that needed work; runs whose
    saves are already marked are skipped silently.
    """

    results: list[MarkerResult] = []
    for run_dir in _iter_run_dirs(store_root):
        result = backfill_run(run_dir, dry_run=dry_run)
        if result is not None:
            results.append(result)
    return results


@click.command()
@click.option(
    "--store-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_STORE_ROOT,
    help="Artifact tree root to scan.",
)
@click.option("--dry-run", is_flag=True, help="List runs that would be certified, without writing.")
def main(store_root: Path, dry_run: bool) -> None:
    if not store_root.is_dir():
        click.echo(f"store-root does not exist: {store_root}", err=True)
        sys.exit(1)
    results = backfill_store(store_root, dry_run=dry_run)
    if not results:
        click.echo("nothing to backfill - every save already carries its marker")
        return
    failed = [r for r in results if r.status == "failed"]
    certified = [r for r in results if r.status != "failed"]
    verb = "would certify" if dry_run else "certified"
    click.echo(f"{verb} {len(certified)} run(s):")
    for result in certified:
        click.echo(f"  [{result.marked:2d} marker(s)] {result.run_dir}")
    if failed:
        click.echo(f"{len(failed)} run(s) failed to load and were left untouched:", err=True)
        for result in failed:
            click.echo(f"  {result.run_dir}: {result.error}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
