"""Walk-forward orchestrator: strategy → engine → metrics, per fold.

The runtime leakage tripwire lives here, not inside the engine. Each
fold:

1. ``strategy.train(fold.train)``
2. ``strategy.training_metadata.validate_no_overlap(fold.test)`` —
   raises ``LeakageError`` if eval data overlaps the training period.
3. ``signals = strategy.generate_signals(fold.test)``
4. ``raw = engine.run(fold.test, signals, slippage)``
5. ``metrics = MetricsCalculator.compute(raw.equity_curve, ...)``

Returned ``FoldResult`` bundles fold metadata + raw engine output +
computed performance metrics so callers don't need to recompute.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_engine import (
    BacktestResult,
    MetricsCalculator,
    PerformanceMetrics,
    SlippageConfig,
)
from src.core.temporal import WalkForwardValidator
from src.core.types import Interval
from src.engine.interface import IBacktestEngine
from src.strategies.interface import IStrategy


@dataclass(frozen=True)
class FoldResult:
    """Per-fold orchestration output: metadata + raw + metrics."""

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    backtest: BacktestResult
    metrics: PerformanceMetrics


def evaluate_walk_forward(
    strategy: IStrategy,
    bars: pd.DataFrame,
    validator: WalkForwardValidator,
    engine: IBacktestEngine,
    slippage: SlippageConfig,
    interval: Interval,
    risk_free_rate: float = 0.0,
) -> list[FoldResult]:
    """Run train → leakage check → signals → engine → metrics, per fold.

    Args:
        strategy: A trained-from-scratch ``IStrategy`` instance. The
            orchestrator calls ``train()`` for each fold, so the same
            strategy instance is reused across folds (fresh fit each time).
        bars: Full OHLCV DataFrame to be split by ``validator``.
        validator: Walk-forward splitter producing ``TemporalSplit``s.
        engine: Backtest engine adapter (typically ``CppBacktestEngine``).
        slippage: Slippage scenario applied uniformly across folds.
        interval: Bar interval, used to pick the annualization factor
            for ``MetricsCalculator``.
        risk_free_rate: Per-period risk-free rate for Sharpe / Sortino
            (default 0.0).

    Returns:
        One ``FoldResult`` per validator fold, in fold order.

    Raises:
        LeakageError: From ``training_metadata.validate_no_overlap`` if a
            fold's test data overlaps its training period (should be
            impossible with a well-formed validator — this is a tripwire,
            not a routine check).
        RuntimeError: If a strategy fails to populate
            ``training_metadata`` after ``train()`` (contract violation).
    """
    annualization = interval.annualization_factor()
    results: list[FoldResult] = []
    for fold in validator.split(bars):
        strategy.train(fold.train)
        if strategy.training_metadata is None:
            raise RuntimeError(
                f"{type(strategy).__name__}.train() did not populate "
                "training_metadata — every IStrategy must set it."
            )
        strategy.training_metadata.validate_no_overlap(fold.test)

        signals = strategy.generate_signals(fold.test)
        raw = engine.run(fold.test, signals, slippage)
        metrics = MetricsCalculator.compute(
            raw.equity_curve,
            annualization,
            risk_free_rate,
        )
        results.append(
            FoldResult(
                fold_index=fold.fold_index,
                train_start=fold.train.index[0],
                train_end=fold.train.index[-1],
                test_start=fold.test.index[0],
                test_end=fold.test.index[-1],
                backtest=raw,
                metrics=metrics,
            )
        )
    return results
