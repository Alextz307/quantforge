"""
Wire DTOs for the HPO studies read API.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class StudyDirection(StrEnum):
    """
    Optimization direction surfaced from the Optuna study.

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
    # Store-relative POSIX path (e.g. ``"hpo/X"`` or ``"studies/main/hpo/X"``).
    # Disambiguates nested studies that share a basename so the listing key,
    # the detail URL, and the holdout flow can route deterministically.
    wire_id: str
    # Display-only basename. Two rows can share ``name`` but never ``wire_id``.
    name: str
    store: str
    created_at: datetime
    n_trials: int
    n_complete: int
    best_value: float | None
    best_trial_number: int | None
    direction: StudyDirection
    has_best_config: bool
    # True iff best_config.yaml exists AND its validation block reserves a
    # holdout region (``holdout_pct > 0`` or ``holdout_start`` pinned). The
    # holdout-launcher picker filters to eligible studies on this flag so the
    # user can't pick a source the CLI would reject for "no holdout
    # reservation" after the subprocess spawns.
    best_config_reserves_holdout: bool
    launched_by_username: str | None = None


class HpoDetail(BaseModel):
    wire_id: str
    name: str
    store: str
    created_at: datetime
    n_trials: int
    n_complete: int
    best_value: float | None
    best_trial_number: int | None
    direction: StudyDirection
    best_config: dict[str, object]
    best_config_reserves_holdout: bool
    live_job_id: str | None
    launched_by_username: str | None = None


class TrialFrame(BaseModel):
    """
    WebSocket frame published whenever a new trial lands in ``trials.jsonl``.
    """

    type: Literal["trial"] = "trial"
    trial: TrialRow


class ParamImportanceResponse(BaseModel):
    """
    Per-hyperparameter relative importance for the live HPO monitor.

    ``importance`` is empty and ``message`` is set when the study has too few
    completed trials, the optuna DB is missing, or the importance evaluator
    raises on a degenerate search space - the endpoint stays 200 so the
    frontend renders an empty-state card instead of erroring.
    """

    importance: dict[str, float]
    message: str | None = None
