"""Read-only services for the persisted studies tree."""

from __future__ import annotations

from pathlib import Path

from src.core import json_io
from src.orchestration.study import STUDY_STATE_FILENAME
from src.orchestration.study_state import StudyState, read_study_state
from src.visualization.plots import MANIFEST_FILENAME
from webapp.backend.app.infrastructure.store import (
    StudyNotFoundError,
    find_study_dir,
    iter_study_dirs,
)
from webapp.backend.app.schemas.studies import (
    LegStateRow,
    StudyConsolidatedDTO,
    StudyDetail,
    StudySummary,
)
from webapp.backend.app.services.plots import (
    PLOTS_DIRNAME,
    TABLES_DIRNAME,
    PlotNotFoundError,
    list_files_under,
    resolve_file_under,
)

__all__ = [
    "ConsolidatedReportNotFoundError",
    "PlotNotFoundError",
    "StudyNotFoundError",
    "get_consolidated",
    "get_study",
    "list_studies",
    "resolve_consolidated_plot",
    "resolve_consolidated_table",
]


class ConsolidatedReportNotFoundError(LookupError):
    """Raised when a study has no consolidated report (``manifest.json`` absent)."""


def list_studies(root: Path) -> list[StudySummary]:
    """List every study under ``root``, newest first."""
    summaries: list[StudySummary] = []
    for study_dir in iter_study_dirs(root):
        state = read_study_state(study_dir / STUDY_STATE_FILENAME)
        summaries.append(_summary_from_state(study_dir.name, state))
    summaries.sort(key=lambda s: s.started_at, reverse=True)
    return summaries


def get_study(root: Path, name: str) -> StudyDetail:
    """Read the full detail payload for one study."""
    study_dir = find_study_dir(root, name)
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
    )


def get_consolidated(root: Path, name: str) -> StudyConsolidatedDTO:
    """Read the consolidated-report manifest + tables/plots index for one study."""
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
        n_legs_with_regime=json_io.get_int(raw, "n_legs_with_regime"),
        n_legs_with_holdout=json_io.get_int(raw, "n_legs_with_holdout"),
        n_universes_with_pairwise=json_io.get_int(raw, "n_universes_with_pairwise"),
        tables=list_files_under(study_dir, TABLES_DIRNAME),
        plots=list_files_under(study_dir, PLOTS_DIRNAME),
    )


def resolve_consolidated_plot(root: Path, name: str, plot_name: str) -> Path:
    """Resolve a consolidated plot filename to an absolute path, blocking traversal."""
    return resolve_file_under(find_study_dir(root, name), PLOTS_DIRNAME, plot_name)


def resolve_consolidated_table(root: Path, name: str, table_name: str) -> Path:
    """Resolve a consolidated table filename to an absolute path, blocking traversal."""
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
