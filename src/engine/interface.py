"""Backtest engine abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.core.temporal import WalkForwardValidator
from src.core.types import BacktestResult, WalkForwardResult
from src.strategies.interface import IStrategy


class IBacktestEngine(ABC):
    """Backtest engine interface — implemented by the C++ engine."""

    @abstractmethod
    def run_backtest(
        self,
        prices: np.ndarray,  # type: ignore[type-arg]
        signals: np.ndarray,  # type: ignore[type-arg]
        transaction_fee: float,
        initial_capital: float,
    ) -> BacktestResult:
        """Run a single backtest on price and signal arrays.

        Args:
            prices: 1-D array of close prices, shape (n_bars,), dtype float64,
                sorted chronologically.
            signals: 1-D array of position signals, shape (n_bars,), dtype float64,
                range [MIN_POSITION, MAX_POSITION]. Must already be shifted
                (signal at index i is applied to price change from i to i+1).
            transaction_fee: Fraction of trade value charged per transaction
                (e.g., 0.001 = 10 bps = 0.1%).
            initial_capital: Starting equity in currency units.

        Returns:
            BacktestResult with performance metrics and equity curve.
        """

    @abstractmethod
    def run_walk_forward(
        self,
        bar_data: pd.DataFrame,
        strategies: dict[str, IStrategy],
        validator: WalkForwardValidator,
        transaction_fee: float,
    ) -> list[WalkForwardResult]:
        """Run full walk-forward evaluation across all folds and strategies.

        Args:
            bar_data: DataFrame with DatetimeIndex and OHLCV columns.
            strategies: Map of strategy name to IStrategy instance.
            validator: Walk-forward splitter for temporal train/test splits.
            transaction_fee: Fraction of trade value charged per transaction.

        Returns:
            One WalkForwardResult per strategy, aggregating across folds.
        """
