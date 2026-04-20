"""Adaptive Bollinger Bands strategy with GARCH-scaled band widths."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd

import quant_engine
from src.core import json_io
from src.core.persistence import (
    CONFIG_JSON,
    GARCH_SUBDIR,
    METADATA_JSON,
    save_model_skeleton,
)
from src.core.registry import strategy_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.core.utils import compute_log_returns
from src.models.garch import GARCHPredictor
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

_STRATEGY_NAME = "AdaptiveBollinger"


@strategy_registry.register(_STRATEGY_NAME)
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
        # Retained so ``save()`` can snapshot the ctor kwargs without
        # reaching into the leaf GARCH's private state.
        self._garch_p_max = garch_p_max
        self._garch_q_max = garch_q_max

        self._garch = GARCHPredictor(p_max=garch_p_max, q_max=garch_q_max, interval=interval)
        self._fitted = False
        self._training_metadata: TrainingMetadata | None = None
        self._cpp_strategy = quant_engine.AdaptiveBollingerStrategy(
            quant_engine.AdaptiveBollingerStrategy.Config(
                band_window=self._window,
                k=self._k,
                trend_window=self._trend_window,
            )
        )

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
        garch_vol_annual = self._garch.predict(data)
        ann_factor_sqrt = math.sqrt(self._interval.annualization_factor())
        daily_price_sigma = (garch_vol_annual / ann_factor_sqrt) * close

        signal = self._cpp_strategy.generate_signals(
            close=np.asarray(close, dtype=np.float64),
            cond_vol=np.asarray(daily_price_sigma, dtype=np.float64),
        )
        return pd.Series(signal, index=data.index, name="adaptive_bollinger_signal")

    def update(self, new_data: pd.DataFrame, **kwargs: object) -> None:
        """Delegate to GARCH's warm-start refit on the extended return series.

        See :meth:`IStrategy.update` for the shared contract.
        """
        if not self._fitted or self._training_metadata is None:
            raise RuntimeError("AdaptiveBollingerStrategy.update() called before train()")

        new_metadata = self._training_metadata.extend_from(new_data)

        new_returns = compute_log_returns(new_data["close"]).dropna()
        self._garch.update(new_data.loc[new_returns.index], new_returns)
        self._training_metadata = new_metadata

    def save(self, path: str | Path) -> None:
        """Persist AdaptiveBollinger config + nested GARCH to ``path``."""
        if not self._fitted:
            raise RuntimeError("AdaptiveBollingerStrategy.save() called before train()")
        if self._training_metadata is None:
            raise RuntimeError("AdaptiveBollingerStrategy.save() missing training metadata")

        def write_weights(root: Path) -> None:
            self._garch.save(root / GARCH_SUBDIR)

        save_model_skeleton(
            path,
            config=self._ctor_kwargs_as_json(),
            training_metadata=self._training_metadata,
            write_weights=write_weights,
        )

    def _ctor_kwargs_as_json(self) -> dict[str, object]:
        """Snapshot of this strategy's constructor kwargs as JSON-ready values."""
        return {
            "window": self._window,
            "k": self._k,
            "trend_window": self._trend_window,
            "garch_p_max": self._garch_p_max,
            "garch_q_max": self._garch_q_max,
            "interval": self._interval.value,
        }

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a trained AdaptiveBollingerStrategy from ``path``.

        Narrow the strategy's ``config.json`` into ctor kwargs BEFORE loading
        the GARCH subdir — a corrupt composite config fast-fails with a
        named-field error rather than crashing deep inside ``GARCH.load()``.
        """
        root = Path(path)
        config = json_io.read_dict(root / CONFIG_JSON)
        metadata = json_io.read_dict(root / METADATA_JSON)

        instance = cls(
            window=json_io.get_int(config, "window"),
            k=json_io.get_float(config, "k"),
            trend_window=json_io.get_int(config, "trend_window"),
            garch_p_max=json_io.get_int(config, "garch_p_max"),
            garch_q_max=json_io.get_int(config, "garch_q_max"),
            interval=Interval(json_io.get_str(config, "interval")),
        )
        instance._garch = GARCHPredictor.load(root / GARCH_SUBDIR)
        instance._training_metadata = TrainingMetadata.from_dict(metadata)
        instance._fitted = True
        return instance

    @property
    def name(self) -> str:
        return _STRATEGY_NAME

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
