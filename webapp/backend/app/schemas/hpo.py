"""Wire DTOs for the HPO studies read API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class TrialRow(BaseModel):
    number: int
    state: str
    value: float | None
    params: dict[str, object]
    datetime_start: datetime | None
    datetime_complete: datetime | None
    experiment_id: str | None


class HpoSummary(BaseModel):
    name: str
    store: str
    created_at: datetime
    n_trials: int
    n_complete: int
    best_value: float | None
    best_trial_number: int | None


class HpoDetail(BaseModel):
    name: str
    store: str
    created_at: datetime
    n_trials: int
    n_complete: int
    best_value: float | None
    best_trial_number: int | None
    best_config: dict[str, object]
