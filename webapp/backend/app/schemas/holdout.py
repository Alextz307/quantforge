"""
Wire DTOs for the holdout-evaluations read API.

Holdout artifacts are produced by ``experiment holdout-eval``. The on-disk
``holdout_eval.json`` carries identity, the source bundle reference
(``source_kind`` / ``source_id`` / ``source_path``), the holdout boundary,
single-pass metrics, and the holdout equity curve.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from src.engine.scenarios import SlippageScenario
from src.orchestration.holdout_eval import SourceKind


class HoldoutSortBy(StrEnum):
    CREATED_AT = "created_at"
    HOLDOUT_START = "holdout_start"
    SHARPE_RATIO = "sharpe_ratio"


class HoldoutEvalSummary(BaseModel):
    name: str
    store: str
    created_at: datetime
    source_kind: SourceKind
    source_id: str
    holdout_start: datetime
    # ``None`` for in-flight evals whose ``metrics`` block hasn't been written
    # yet. Surfacing it on the listing lets the table render + sort on Sharpe
    # without forcing a per-row detail fetch.
    sharpe_ratio: float | None
    launched_by_username: str | None = None


class HoldoutEvalDetail(BaseModel):
    name: str
    store: str
    created_at: datetime
    source_kind: SourceKind
    source_id: str
    source_path: str
    holdout_start: datetime
    n_dev_bars: int
    n_holdout_bars: int
    slippage_scenario: SlippageScenario
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    equity_curve: list[float]
    plots: list[str]
    launched_by_username: str | None = None


class HoldoutEvalsPage(BaseModel):
    """
    Paginated envelope for `/api/holdout-evals`.

    ``source_kinds`` enumerates the distinct source kinds across the full
    visible set (independent of the current page or filters) so the filter
    dropdown stays fully populated under server-side pagination.
    """

    items: list[HoldoutEvalSummary]
    total: int
    limit: int
    offset: int
    source_kinds: list[SourceKind]
