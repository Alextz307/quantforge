"""Wire DTOs for the studies read API."""

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
