"""
Wire DTOs for the deployments API.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.core.types import Interval
from src.orchestration.holdout_eval import SourceKind


class DeploymentCreate(BaseModel):
    """
    POST /deployments body.

    ``name`` defaults to ``"<ticker>-<strategy>-<train_end>"`` and
    ``warmup_bars`` defaults to the strategy's
    ``required_warmup_bars + convergence_margin_bars`` (subject to an
    absolute floor) when omitted.
    """

    source_kind: SourceKind
    source_id: str
    name: str | None = None
    warmup_bars: int | None = Field(default=None, ge=1)


class DeploymentRename(BaseModel):
    """PATCH /deployments/{id} body."""

    name: str = Field(min_length=1, max_length=200)


class DeploymentSummary(BaseModel):
    """Row in the deployments list view."""

    id: str
    name: str
    source_kind: SourceKind
    source_id: str
    ticker: str
    strategy_name: str
    interval: Interval
    train_end: datetime
    warmup_bars: int
    created_at: datetime
    owner_username: str


class SignalRowOut(BaseModel):
    """
    One row from the deployment's signal log.

    ``bar_ts`` is the last completed bar the signal was computed from;
    ``signal_date`` is the trading day the signal is *for* (the next
    session after ``bar_ts``).
    """

    submitted_at: datetime
    bar_ts: datetime
    signal_date: datetime
    signal: float
    warmup_fingerprint: str
    source_run_id: str
    warmup_bars_used: int


class DeploymentDetail(DeploymentSummary):
    """Summary + the tail of the signal log."""

    latest_signal: SignalRowOut | None = None


class PredictIfStaleResponse(BaseModel):
    """
    Response for POST /deployments/{id}/predict-if-stale.

    ``stale=False`` means the cached signal already covered the latest
    available bar; ``stale=True`` means a fresh predict ran.
    """

    stale: bool
    signal: SignalRowOut
