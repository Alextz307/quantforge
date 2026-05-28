"""
Wire DTOs for the studies read API.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from src.orchestration.study_state import LegStep


class LegStateRow(BaseModel):
    leg_id: str
    strategy: str
    universe: str
    is_complete: bool
    error: str | None
    run_experiment_id: str | None
    started_at: datetime | None
    completed_at: datetime | None
    steps_completed: list[LegStep]


class StudySummary(BaseModel):
    name: str
    spec_name: str
    started_at: datetime
    total_legs: int
    completed_legs: int
    completion_pct: float
    launched_by_username: str | None = None


class StudyDetail(BaseModel):
    name: str
    spec_name: str
    spec_hash: str
    started_at: datetime
    total_legs: int
    completed_legs: int
    completion_pct: float
    cross_strategy_compares_done: list[str]
    legs: list[LegStateRow]
    # Pre-flight signal so the frontend skips the consolidated-report fetch
    # (a 404 round-trip) when no report has been generated yet.
    has_consolidated_report: bool
    launched_by_username: str | None = None


class StudyConsolidatedDTO(BaseModel):
    """
    Read view of a study's consolidated report (tables + plots tree).
    """

    study_name: str
    publish_label: str
    created_at: datetime
    git_sha: str
    strategies: list[str]
    universes: list[str]
    incomplete_leg_ids: list[str]
    n_legs_with_holdout: int
    n_universes_with_pairwise: int
    tables: list[str]
    plots: list[str]
