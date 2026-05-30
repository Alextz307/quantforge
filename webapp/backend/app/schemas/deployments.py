"""
Wire DTOs for the deployments API.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.core.types import Interval
from src.engine.scenarios import SlippageScenario
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


class ScoredSignalOut(BaseModel):
    """
    One emitted signal scored open->open against realised session opens.

    Entered at ``entry_date``'s open (the first session after ``bar_ts``),
    exited at ``exit_date``'s open (the next session). ``listened_return``
    is the signed, leverage-scaled realised return. The realised fields are
    populated together iff ``scored`` is true - a signal stays unscored
    until its exit session has opened. ``hit`` is null for a FLAT signal.
    """

    bar_ts: datetime
    signal: float
    entry_date: datetime | None
    entry_open: float | None
    exit_date: datetime | None
    exit_open: float | None
    asset_return: float | None
    listened_return: float | None
    hit: bool | None
    cumulative_return: float | None
    cost: float | None
    net_listened_return: float | None
    net_cumulative_return: float | None
    scored: bool


class SignalEvaluationOut(BaseModel):
    """
    Response for GET /deployments/{id}/signal-evaluation.

    Per-signal scores plus headline stats over the scored subset.
    ``hit_rate`` covers directional (non-FLAT) scored signals only;
    ``cumulative_return`` compounds every scored ``listened_return``
    (gross), and ``net_cumulative_return`` does so after costs. All summary
    stats are null when nothing has been scored yet. ``cost_scenario`` is
    the tier whose slippage + commission produced the net figures.
    """

    rows: list[ScoredSignalOut]
    n_signals: int
    n_scored: int
    n_hits: int
    hit_rate: float | None
    cumulative_return: float | None
    mean_return: float | None
    net_cumulative_return: float | None
    net_mean_return: float | None
    cost_scenario: SlippageScenario
