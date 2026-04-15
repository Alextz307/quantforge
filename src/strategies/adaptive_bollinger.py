"""Adaptive Bollinger Bands strategy with GARCH-scaled band widths."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.core.registry import strategy_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.core.utils import compute_log_returns
from src.models.garch import GARCHPredictor
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


@strategy_registry.register("AdaptiveBollinger")
class AdaptiveBollingerStrategy(IStrategy):
    """Mean-reversion Bollinger-band strategy with GARCH-adaptive band widths.

    Bands are computed as ``mid ± k * daily_price_sigma`` where
    ``daily_price_sigma = (garch_vol_annual / sqrt(ann_factor)) * close``.
    A longer-window SMA filters trend direction: longs entered only in
    bullish regimes, shorts only in bearish regimes.
    """

    def __init__(
        self,
        window: int = 20,
        k: float = 2.0,
        trend_window: int = 100,
        garch_p_max: int = 5,
        garch_q_max: int = 5,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if trend_window < 2:
            raise ValueError(f"trend_window must be >= 2, got {trend_window}")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        self._window = window
        self._k = k
        self._trend_window = trend_window
        self._interval = interval

        self._garch = GARCHPredictor(p_max=garch_p_max, q_max=garch_q_max, interval=interval)
        self._fitted = False
        self._training_metadata: TrainingMetadata | None = None

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        """Fit the GARCH volatility model on training log returns."""
        log_returns = compute_log_returns(train_data["close"]).dropna()
        aligned = train_data.loc[log_returns.index]
        self._garch.fit(aligned, log_returns)

        self._training_metadata = TrainingMetadata.from_fit(train_data, self._interval, ("close",))
        self._fitted = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Produce {-1, 0, +1} position signals. Leading warmup bars are NaN."""
        if not self._fitted:
            raise RuntimeError("AdaptiveBollingerStrategy.generate_signals() called before train()")

        close = data["close"]
        mid = close.rolling(self._window).mean()
        trend_ma = close.rolling(self._trend_window).mean()

        garch_vol_annual = self._garch.predict(data)
        ann_factor_sqrt = math.sqrt(self._interval.annualization_factor())
        daily_price_sigma = (garch_vol_annual / ann_factor_sqrt) * close

        upper = mid + self._k * daily_price_sigma
        lower = mid - self._k * daily_price_sigma

        signal = self._run_state_machine(
            close=close.to_numpy(),
            mid=mid.to_numpy(),
            upper=upper.to_numpy(),
            lower=lower.to_numpy(),
            trend_ma=trend_ma.to_numpy(),
        )
        return pd.Series(signal, index=data.index, name="adaptive_bollinger_signal")

    @staticmethod
    def _run_state_machine(
        close: np.ndarray[tuple[int], np.dtype[np.float64]],
        mid: np.ndarray[tuple[int], np.dtype[np.float64]],
        upper: np.ndarray[tuple[int], np.dtype[np.float64]],
        lower: np.ndarray[tuple[int], np.dtype[np.float64]],
        trend_ma: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        n = len(close)
        out = np.full(n, np.nan, dtype=np.float64)
        position = 0.0
        for t in range(n):
            if (
                np.isnan(mid[t])
                or np.isnan(upper[t])
                or np.isnan(lower[t])
                or np.isnan(trend_ma[t])
            ):
                continue
            is_bull = close[t] > trend_ma[t]
            if position == 0.0:
                if is_bull and close[t] < lower[t]:
                    position = 1.0
                elif (not is_bull) and close[t] > upper[t]:
                    position = -1.0
            elif position == 1.0:
                if close[t] >= mid[t]:
                    position = 0.0
            elif position == -1.0:
                if close[t] <= mid[t]:
                    position = 0.0
            out[t] = position
        return out

    @property
    def name(self) -> str:
        return "AdaptiveBollinger"

    @property
    def required_warmup_bars(self) -> int:
        return max(self._window, self._trend_window)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for AdaptiveBollinger hyperparameters."""
        return {
            "window": trial.suggest_int("bollinger_window", 10, 50),
            "k": trial.suggest_float("bollinger_k", 1.0, 3.0),
            "trend_window": trial.suggest_int("bollinger_trend_window", 50, 200),
            "garch_p_max": trial.suggest_int("bollinger_garch_p_max", 1, 5),
            "garch_q_max": trial.suggest_int("bollinger_garch_q_max", 1, 5),
        }
