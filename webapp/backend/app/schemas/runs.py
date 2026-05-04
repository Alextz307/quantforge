"""Wire DTOs for the runs read API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from src.engine.scenarios import SlippageScenario


class RunSummary(BaseModel):
    """List-view row for `/api/runs`.

    Strategy/tickers/interval are sourced from the run's frozen
    `config.yaml` so any future strategy works without changing this code.
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


class PretrainedLeafDTO(BaseModel):
    key: str
    path: str
    data_hash: str
    train_start: datetime
    train_end: datetime


class RunDetail(BaseModel):
    """Detail-view payload for `/api/runs/{experiment_id}`."""

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
    pretrained_leaves: list[PretrainedLeafDTO]
    metrics: dict[str, float]
    plots: list[str]


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
