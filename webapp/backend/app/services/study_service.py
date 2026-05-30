"""
Read-only services for the persisted studies tree.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from src.core import json_io
from src.orchestration.study import STUDY_STATE_FILENAME
from src.orchestration.study_report import consolidate_study
from src.orchestration.study_state import StudyState, read_study_state
from src.visualization.plots import MANIFEST_FILENAME
from src.visualization.study_report_reporter import StudyReportReporter
from webapp.backend.app.infrastructure.store import (
    StudyNotFoundError,
    find_study_dir,
    iter_study_dirs,
)
from webapp.backend.app.schemas.jobs import TERMINAL_STATUSES, JobKind
from webapp.backend.app.schemas.studies import (
    LegStateRow,
    StudyConsolidatedDTO,
    StudyDetail,
    StudySummary,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services._dir_cache import cached_artifact_dirs
from webapp.backend.app.services.ownership import (
    ArtifactAccessDeniedError,
    check_artifact_access,
    resolve_owner_usernames,
    scope_and_stamp_summaries,
)
from webapp.backend.app.services.plots import (
    PLOTS_DIRNAME,
    TABLES_DIRNAME,
    PlotNotFoundError,
    list_files_under,
    resolve_file_under,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ArtifactAccessDeniedError",
    "ConsolidatedReportNotFoundError",
    "PlotNotFoundError",
    "StudyConsolidationError",
    "StudyNotFoundError",
    "build_study_detail",
    "find_live_study_job_for",
    "generate_consolidated",
    "get_consolidated",
    "get_study",
    "list_studies",
    "resolve_consolidated_plot",
    "resolve_consolidated_table",
]


class ConsolidatedReportNotFoundError(LookupError):
    """
    Raised when a study has no consolidated report (``manifest.json`` absent).
    """


class StudyConsolidationError(ValueError):
    """
    Raised when consolidation cannot complete (e.g. missing per-leg artifacts).
    """


def list_studies(
    root: Path,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
    all_users: bool,
) -> list[StudySummary]:
    """
    List every study under ``root`` visible to ``user``, newest first.
    """

    summaries: list[StudySummary] = []
    for study_dir in cached_artifact_dirs(root, "study", iter_study_dirs):
        try:
            state = read_study_state(study_dir / STUDY_STATE_FILENAME)
        except Exception as exc:  # noqa: BLE001 - one bad study must not 500 the whole listing
            logger.warning("skipping unreadable study at %s: %s", study_dir, exc)
            continue
        summaries.append(_summary_from_state(study_dir.name, state))
    scoped = scope_and_stamp_summaries(
        summaries, key_fn=lambda s: s.name, conn=conn, user=user, all_users=all_users
    )
    scoped.sort(key=lambda s: s.started_at, reverse=True)
    return scoped


def get_study(
    root: Path,
    name: str,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
) -> StudyDetail:
    """
    Read the full detail payload for one study.

    Raises :class:`ArtifactAccessDeniedError` when ``user`` is neither owner
    nor admin; the router maps that to 404.
    """

    check_artifact_access(conn, experiment_id=name, user=user)
    detail = build_study_detail(find_study_dir(root, name))
    usernames = resolve_owner_usernames(conn, experiment_ids=[name])
    return detail.model_copy(update={"launched_by_username": usernames.get(name)})


def build_study_detail(study_dir: Path) -> StudyDetail:
    """
    Read the full detail payload from an already-resolved study directory.

    Skips the recursive glob inside :func:`find_study_dir` - callers that
    already hold the resolved path (e.g. the WS streamer) reuse it across
    mtime ticks without re-globbing the store on every frame.
    """

    state = read_study_state(study_dir / STUDY_STATE_FILENAME)
    completed, total = _completion_counts(state)
    return StudyDetail(
        name=study_dir.name,
        spec_name=state.spec_name,
        spec_hash=state.spec_hash,
        started_at=state.started_at,
        total_legs=total,
        completed_legs=completed,
        completion_pct=_pct(completed, total),
        cross_strategy_compares_done=list(state.cross_strategy_compares_done),
        legs=[
            LegStateRow(
                leg_id=leg.leg_id,
                strategy=leg.strategy,
                universe=leg.universe,
                is_complete=leg.is_complete,
                error=leg.error,
                run_experiment_id=leg.run_experiment_id,
                started_at=leg.started_at,
                completed_at=leg.completed_at,
                steps_completed=list(leg.steps_completed),
            )
            for leg in state.legs
        ],
        has_consolidated_report=(study_dir / MANIFEST_FILENAME).is_file(),
    )


def get_consolidated(
    root: Path,
    name: str,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
) -> StudyConsolidatedDTO:
    """
    Read the consolidated-report manifest + tables/plots index for one study.
    """

    check_artifact_access(conn, experiment_id=name, user=user)
    study_dir = find_study_dir(root, name)
    manifest_path = study_dir / MANIFEST_FILENAME

    try:
        raw = json_io.read_dict(manifest_path)
    except FileNotFoundError as exc:
        raise ConsolidatedReportNotFoundError(name) from exc

    return StudyConsolidatedDTO(
        study_name=json_io.get_str(raw, "study_name"),
        publish_label=json_io.get_str(raw, "publish_label"),
        created_at=json_io.get_timestamp(raw, "created_at"),
        git_sha=json_io.get_str(raw, "git_sha"),
        strategies=list(json_io.get_str_list(raw, "strategies")),
        universes=list(json_io.get_str_list(raw, "universes")),
        incomplete_leg_ids=list(json_io.get_str_list(raw, "incomplete_leg_ids")),
        n_legs_with_holdout=json_io.get_int(raw, "n_legs_with_holdout"),
        n_universes_with_pairwise=json_io.get_int(raw, "n_universes_with_pairwise"),
        tables=list_files_under(study_dir, TABLES_DIRNAME),
        plots=list_files_under(study_dir, PLOTS_DIRNAME),
    )


def generate_consolidated(
    root: Path,
    name: str,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
) -> StudyConsolidatedDTO:
    """
    Build (or rebuild) the consolidated report for a study and return its DTO.

    Idempotent: safe to call against a study that already has a consolidated
    report - old tables/plots are overwritten. Raises :class:`StudyNotFoundError`
    if the study doesn't exist, :class:`StudyConsolidationError` if the per-leg
    artifacts are missing or malformed (e.g. running against a study that
    hasn't completed any legs yet).
    """

    check_artifact_access(conn, experiment_id=name, user=user)
    study_dir = find_study_dir(root, name)
    try:
        report = consolidate_study(study_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise StudyConsolidationError(str(exc)) from exc
    StudyReportReporter().generate_full_report(report, study_dir)
    return get_consolidated(root, name, conn=conn, user=user)


def resolve_consolidated_plot(
    root: Path,
    name: str,
    plot_name: str,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
) -> Path:
    """
    Resolve a consolidated plot filename to an absolute path, blocking traversal.
    """

    check_artifact_access(conn, experiment_id=name, user=user)
    return resolve_file_under(find_study_dir(root, name), PLOTS_DIRNAME, plot_name)


def resolve_consolidated_table(
    root: Path,
    name: str,
    table_name: str,
    *,
    conn: sqlite3.Connection,
    user: UserPublic,
) -> Path:
    """
    Resolve a consolidated table filename to an absolute path, blocking traversal.
    """

    check_artifact_access(conn, experiment_id=name, user=user)
    return resolve_file_under(find_study_dir(root, name), TABLES_DIRNAME, table_name)


def _summary_from_state(name: str, state: StudyState) -> StudySummary:
    completed, total = _completion_counts(state)
    return StudySummary(
        name=name,
        spec_name=state.spec_name,
        started_at=state.started_at,
        total_legs=total,
        completed_legs=completed,
        completion_pct=_pct(completed, total),
    )


def _completion_counts(state: StudyState) -> tuple[int, int]:
    completed = sum(1 for leg in state.legs if leg.is_complete)
    return completed, len(state.legs)


def _pct(completed: int, total: int) -> float:
    return (completed / total * 100.0) if total > 0 else 0.0


def find_live_study_job_for(conn: sqlite3.Connection, output_dir_name: str) -> str | None:
    """
    Return the id of a non-terminal STUDY job populating ``studies/<name>``.

    Study jobs persist ``experiment_id = spec.output_dir.name`` at
    submission time. At most one non-terminal STUDY job per output dir
    is expected - submit_job rejects collisions; this helper powers the
    rejection check.
    """

    terminal = tuple(s.value for s in TERMINAL_STATUSES)
    placeholders = ",".join("?" * len(terminal))
    row = conn.execute(
        f"SELECT id FROM jobs "
        f"WHERE kind = ? AND experiment_id = ? AND status NOT IN ({placeholders}) "
        f"ORDER BY id DESC LIMIT 1",
        (JobKind.STUDY.value, output_dir_name, *terminal),
    ).fetchone()
    if row is None:
        return None
    return str(row["id"])
