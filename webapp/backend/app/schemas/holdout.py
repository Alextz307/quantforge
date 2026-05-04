"""Wire DTOs for the holdout-evaluations read API.

Holdout artifacts are produced by ``experiment holdout-eval``. The on-disk
``holdout_eval.json`` carries identity, the source bundle reference
(``source_kind`` / ``source_id`` / ``source_path``), the holdout boundary,
single-pass metrics, and the holdout equity curve.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from src.engine.scenarios import SlippageScenario
from src.orchestration.holdout_eval import SourceKind


class HoldoutEvalSummary(BaseModel):
    name: str
    store: str
    created_at: datetime
    source_kind: SourceKind
    source_id: str
    holdout_start: datetime


class HoldoutEvalDetail(BaseModel):
    name: str
    store: str
    created_at: datetime
    git_sha: str
    source_kind: SourceKind
    source_id: str
    source_path: str
    holdout_start: datetime
    data_hash: str
    n_dev_bars: int
    n_holdout_bars: int
    slippage_scenario: SlippageScenario
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
    plots: list[str]
