"""Wire DTOs for the regime-reports read API.

Regime artifacts are produced by ``experiment regime``. The on-disk
``manifest.json`` carries identity, the underlying run's ``experiment_id``,
detector metadata (``kind`` / ``detector_name``), per-regime aggregate
stats, the per-regime fold indices, the mixed-fold indices, and the
list of detector slices.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from src.orchestration.types import RegimeKind


class RegimeSliceDTO(BaseModel):
    label: str
    start: datetime
    end: datetime


class PerRegimeStatsRow(BaseModel):
    regime_label: str
    n_folds: int
    sharpe_mean: float
    sharpe_std: float
    sharpe_ci95_low: float
    sharpe_ci95_high: float
    sortino_mean: float
    sortino_std: float
    sortino_ci95_low: float
    sortino_ci95_high: float
    calmar_mean: float
    calmar_std: float
    calmar_ci95_low: float
    calmar_ci95_high: float
    total_return_mean: float
    total_return_std: float
    max_drawdown_mean: float
    max_drawdown_worst: float
    win_rate_mean: float
    trade_count_total: int


class RegimeReportSummary(BaseModel):
    name: str
    store: str
    created_at: datetime
    experiment_id: str
    kind: RegimeKind
    detector_name: str
    regime_labels: list[str]


class RegimeReportDetail(BaseModel):
    name: str
    store: str
    created_at: datetime
    git_sha: str
    experiment_id: str
    kind: RegimeKind
    detector_name: str
    per_regime_stats: list[PerRegimeStatsRow]
    per_regime_fold_indices: dict[str, list[int]]
    mixed_fold_indices: list[int]
    slices: list[RegimeSliceDTO]
    plots: list[str]
