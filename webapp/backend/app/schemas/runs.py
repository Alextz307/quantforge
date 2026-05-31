"""
Wire DTOs for the runs read API.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from src.analysis.feature_importance import ImportanceMethod
from src.engine.scenarios import SlippageScenario


class RunSortBy(StrEnum):
    CREATED_AT = "created_at"
    SHARPE_MEAN = "sharpe_mean"
    CALMAR_MEAN = "calmar_mean"


class SortOrder(StrEnum):
    ASC = "asc"
    DESC = "desc"


class RunSummary(BaseModel):
    """
    List-view row for `/api/runs`.

    Strategy/tickers/interval are sourced from the run's frozen
    `config.yaml` so any future strategy works without changing this code.
    ``has_holdout`` mirrors ``manifest.holdout_start is not None`` so the
    holdout-launcher form can filter eligible runs without a per-row
    detail fetch. ``data_hash`` is the run's bar-series fingerprint so the
    compare picker can lock subsequent selections to the first run's
    universe (the paired-bootstrap test rejects mixed bar indices).
    """

    experiment_id: str
    name: str
    strategy: str
    tickers: list[str]
    interval: str
    store: str
    created_at: datetime
    sharpe_mean: float | None
    calmar_mean: float | None
    has_holdout: bool
    data_hash: str
    launched_by_username: str | None = None


class RunDetail(BaseModel):
    """
    Detail-view payload for `/api/runs/{experiment_id}`.
    """

    experiment_id: str
    name: str
    strategy: str
    tickers: list[str]
    interval: str
    store: str
    created_at: datetime
    git_sha: str
    seed: int
    data_hash: str
    slippage_scenario: SlippageScenario
    holdout_start: datetime | None
    metrics: dict[str, float]
    plots: list[str]
    launched_by_username: str | None = None


class FoldRow(BaseModel):
    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    equity_curve: list[float]


class RunsPage(BaseModel):
    """
    Paginated envelope for `/api/runs`.
    """

    items: list[RunSummary]
    total: int
    limit: int
    offset: int


class FeatureImportanceEntry(BaseModel):
    """
    One feature's cross-fold aggregated importance under one method.

    ``importance`` and ``std`` are ``None`` when the aggregate is non-finite
    (a degenerate fold the driver could not score).
    """

    feature: str
    importance: float | None
    std: float | None
    n_folds: int
    method: ImportanceMethod


class FeatureImportanceResponse(BaseModel):
    """
    Cross-fold feature importance for `/api/runs/{experiment_id}/feature-importance`.

    ``entries`` is empty and ``message`` is set for the common "no artifact"
    cases (importance not requested for the run, a rule-based strategy that
    emits none, or a pre-importance run) so the endpoint stays 200.

    ``computable`` is whether this run's strategy can produce importance at all
    (feature-consuming strategies can; rule-based ones cannot). The frontend
    offers an on-demand "compute importance" action only when ``computable`` is
    true and ``entries`` is empty.

    ``diverged_run_id`` is set when a prior recompute diverged (the re-fit
    didn't reproduce this run's metrics) and saved importance as a separate
    run; the detail page links to it persistently across reloads.
    """

    entries: list[FeatureImportanceEntry]
    message: str | None = None
    computable: bool = False
    diverged_run_id: str | None = None
