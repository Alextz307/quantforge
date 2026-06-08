"""
Wire DTOs for the comparisons read API.

Comparison artifacts are produced by ``experiment compare``. The on-disk
``manifest.json`` carries identity (``out_name`` / ``created_at`` /
``git_sha``), the per-strategy run linkage (``per_strategy_experiment_id``),
and ``per_strategy_stats`` (one :class:`AggregateStats` per strategy).

Pairwise significance is computed in memory but **not** persisted; surfacing
it would require a comparison-reporter change. ``ComparisonDetail`` surfaces
the user-facing subset of the manifest -- identity (minus the integrity-only
``git_sha``) plus ``per_strategy_stats`` -- not the full on-disk record.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class PerStrategyStatsRow(BaseModel):
    strategy: str
    experiment_id: str
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


class ComparisonSummary(BaseModel):
    name: str
    store: str
    created_at: datetime
    strategies: list[str]
    launched_by_username: str | None = None


class ComparisonDetail(BaseModel):
    name: str
    store: str
    created_at: datetime
    per_strategy_stats: list[PerStrategyStatsRow]
    plots: list[str]
    launched_by_username: str | None = None
