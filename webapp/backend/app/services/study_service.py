"""Read-only services for the persisted studies tree."""

from __future__ import annotations

from pathlib import Path

from src.orchestration.study import STUDY_STATE_FILENAME
from src.orchestration.study_state import StudyState, read_study_state
from webapp.backend.app.infrastructure.store import (
    StudyNotFoundError,
    find_study_dir,
    iter_study_dirs,
)
from webapp.backend.app.schemas.studies import (
    LegStateRow,
    StudyDetail,
    StudySummary,
)

__all__ = [
    "StudyNotFoundError",
    "get_study",
    "list_studies",
]


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
