"""Pairs trading strategy using Engle-Granger cointegration and z-score mean reversion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

import quant_engine
from src.core.registry import strategy_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.models.cointegration import CointegrationTester
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


@strategy_registry.register("PairsTrading")
class PairsTradingStrategy(IStrategy):
    """Pairs trading on a cointegrated spread via rolling z-score.

    Expects ``train_data`` / ``data`` with ``close_a`` and ``close_b`` columns.
    ``generate_signals()`` returns leg_a position in ``{-1, 0, +1}``; the
    backtest engine can derive leg_b position as
    ``-hedge_ratio * leg_a_position`` via the ``hedge_ratio`` property.
    """

    def __init__(
        self,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_loss_zscore: float = 4.0,
        zscore_lookback: int = 60,
        p_value_threshold: float = 0.05,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if entry_zscore <= 0 or exit_zscore < 0 or stop_loss_zscore <= 0:
            raise ValueError("z-score thresholds must be positive")
        if exit_zscore >= entry_zscore:
            raise ValueError(f"exit_zscore ({exit_zscore}) must be < entry_zscore ({entry_zscore})")
        if stop_loss_zscore <= entry_zscore:
            raise ValueError(
                f"stop_loss_zscore ({stop_loss_zscore}) must be > entry_zscore ({entry_zscore})"
            )
        if zscore_lookback < 2:
            raise ValueError(f"zscore_lookback must be >= 2, got {zscore_lookback}")

        self._entry_zscore = entry_zscore
        self._exit_zscore = exit_zscore
        self._stop_loss_zscore = stop_loss_zscore
        self._zscore_lookback = zscore_lookback
        self._p_value_threshold = p_value_threshold
        self._interval = interval

        self._hedge_ratio = 0.0
        self._spread_mean = 0.0
        self._spread_std = 0.0
        self._is_cointegrated = False
        self._fitted = False
        self._training_metadata: TrainingMetadata | None = None

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        """Run Engle-Granger cointegration and cache hedge ratio / spread stats."""
        if "close_a" not in train_data.columns or "close_b" not in train_data.columns:
            raise ValueError(
                "PairsTradingStrategy.train() requires 'close_a' and 'close_b' columns"
            )

        result = CointegrationTester.engle_granger(
            train_data["close_a"],
            train_data["close_b"],
            self._p_value_threshold,
        )
        if not result.is_cointegrated:
            raise ValueError(
                f"Pair not cointegrated (p-value {result.p_value:.4f} "
                f">= {self._p_value_threshold:.4f})"
            )

        self._hedge_ratio = result.hedge_ratio
        self._spread_mean = result.spread_mean
        self._spread_std = result.spread_std
        self._is_cointegrated = True

        self._training_metadata = TrainingMetadata.from_fit(
            train_data, self._interval, ("close_a", "close_b")
        )
        self._fitted = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Produce {-1, 0, +1} leg_a position. Leading lookback bars are NaN."""
        if not self._fitted:
            raise RuntimeError("PairsTradingStrategy.generate_signals() called before train()")
        if "close_a" not in data.columns or "close_b" not in data.columns:
            raise ValueError(
                "PairsTradingStrategy.generate_signals() requires 'close_a' and 'close_b' columns"
            )

        spread = data["close_a"] - self._hedge_ratio * data["close_b"]
        rolling_mean = spread.rolling(self._zscore_lookback).mean()
        rolling_std = spread.rolling(self._zscore_lookback).std()
        zscore = (spread - rolling_mean) / rolling_std

        signal = quant_engine.run_pairs_state_machine(
            zscore=zscore.to_numpy(),
            entry_zscore=self._entry_zscore,
            exit_zscore=self._exit_zscore,
            stop_loss_zscore=self._stop_loss_zscore,
        )
        return pd.Series(signal, index=data.index, name="pairs_signal")

    @property
    def hedge_ratio(self) -> float:
        """Cointegration hedge ratio (slope of OLS regression of a on b)."""
        if not self._fitted:
            raise RuntimeError("PairsTradingStrategy.hedge_ratio accessed before train()")
        return self._hedge_ratio

    @property
    def name(self) -> str:
        return "PairsTrading"

    @property
    def required_warmup_bars(self) -> int:
        return self._zscore_lookback

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for PairsTrading hyperparameters."""
        return {
            "entry_zscore": trial.suggest_float("pairs_entry_z", 1.5, 3.0),
            "exit_zscore": trial.suggest_float("pairs_exit_z", 0.0, 1.0),
            "stop_loss_zscore": trial.suggest_float("pairs_stop_z", 3.5, 5.0),
            "zscore_lookback": trial.suggest_int("pairs_lookback", 30, 120),
        }
