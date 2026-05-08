"""Wire DTOs for the HPO studies read API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class StudyDirection(StrEnum):
    """Optimization direction surfaced from the Optuna study.

    The framework hardcodes ``maximize`` (loss-style metrics negate at
    the objective layer); this enum exists so the chart layer can pick
    cummax vs cummin without re-reading the optuna sqlite per request.
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


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
    direction: StudyDirection


class HpoDetail(BaseModel):
    name: str
    store: str
    created_at: datetime
    n_trials: int
    n_complete: int
    best_value: float | None
    best_trial_number: int | None
    direction: StudyDirection
    best_config: dict[str, object]
    live_job_id: str | None


class TrialFrame(BaseModel):
    """WebSocket frame published whenever a new trial lands in ``trials.jsonl``."""

    type: Literal["trial"] = "trial"
    trial: TrialRow
