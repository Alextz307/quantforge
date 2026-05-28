"""
Backtest engine abstract interface.

Returns the raw C++ ``BacktestResult`` (``equity_curve`` +
``total_return`` + ``trade_count`` populated; statistical metric fields
default-zero). Callers compute the statistical metrics via
``quant_engine.MetricsCalculator``; the ``walk_forward`` orchestrator
bundles raw + metrics into ``FoldResult`` per fold.

Direct callers of ``run()`` are responsible for their own anti-leakage
hygiene — the engine is a pure number cruncher and does not inspect
``strategy.training_metadata``. Use ``evaluate_walk_forward`` to get the
``validate_no_overlap`` tripwire wired in for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd

from quant_engine import BacktestResult, SlippageConfig


class IBacktestEngine(ABC):
    """
    Backtest engine interface — implemented by ``CppBacktestEngine``.
    """

    @abstractmethod
    def run(
        self,
        bars: pd.DataFrame,
        signals: pd.Series,
        slippage: SlippageConfig,
    ) -> BacktestResult:
        """
        Run a single-scenario backtest.

        Args:
            bars: DataFrame with DatetimeIndex and columns
                {open, high, low, close, volume}, sorted chronologically.
            signals: Series aligned with ``bars.index`` carrying target
                position values. NaN entries map to position = 0 (flat).
                The engine reads ``signals[i]`` and fills at
                ``bars[i+1].open``.
            slippage: Slippage model + parameters applied per fill.

        Returns:
            ``BacktestResult`` with ``equity_curve``, ``total_return``,
            and ``trade_count`` populated. Statistical metrics fields
            (``sharpe_ratio``, ``sortino_ratio``, ...) default-zero —
            compute via ``MetricsCalculator``.
        """

    @abstractmethod
    def run_scenarios(
        self,
        bars: pd.DataFrame,
        signals: pd.Series,
        scenarios: Sequence[SlippageConfig],
    ) -> list[BacktestResult]:
        """
        Run the same bars + signals across multiple slippage scenarios.

        The bars vector is constructed once and reused across scenarios
        on the C++ side; this is the recommended API for slippage sweeps.
        """

    @abstractmethod
    def run_pairs(
        self,
        bars_a: pd.DataFrame,
        bars_b: pd.DataFrame,
        signals: pd.Series,
        hedge_ratio: float,
        slippage: SlippageConfig,
    ) -> BacktestResult:
        """
        Run a two-leg (pairs / cointegration) backtest.

        Args:
            bars_a: OHLCV DataFrame for leg A, DatetimeIndex aligned with
                ``bars_b`` and ``signals``.
            bars_b: OHLCV DataFrame for leg B, same shape as ``bars_a``.
            signals: Series aligned with both bar frames carrying leg A's
                target leverage. NaN → 0 (flat). Leg B's target is
                ``-hedge_ratio * signals[t]`` (the cointegration short).
            hedge_ratio: Cointegration hedge ratio from
                ``PairsTradingStrategy.hedge_ratio`` (slope of OLS
                regression of A on B). Sized at the leg-B notional level —
                leg-A always gets a unit-leverage exposure.
            slippage: Slippage model applied per fill on both legs.

        Returns:
            ``BacktestResult`` with combined ``equity_curve``,
            ``total_return``, and ``trade_count`` (one per bar in which
            either leg traded). Statistical metrics default-zero — compute
            via ``MetricsCalculator``.
        """
