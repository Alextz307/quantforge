"""Adaptive Bollinger Bands strategy with GARCH-scaled band widths."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
import pandas as pd

import quant_engine
from src.core import json_io
from src.core.logging import get_logger
from src.core.persistence import (
    CONFIG_JSON,
    GARCH_SUBDIR,
    METADATA_JSON,
    save_model_skeleton,
)
from src.core.registry import strategy_registry
from src.core.temporal import TrackedMetadata, TrainingMetadata, collect_metadata
from src.core.types import Interval
from src.core.utils import compute_log_returns
from src.models.garch import GARCHPredictor
from src.orchestration.pretrained_leaves import normalize_pretrained_leaves
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = get_logger(__name__)

_STRATEGY_NAME = "AdaptiveBollinger"


@strategy_registry.register(_STRATEGY_NAME)
class AdaptiveBollingerStrategy(IStrategy):
    """Mean-reversion Bollinger-band strategy with GARCH-adaptive band widths.

    Bands are computed as ``mid ± k * daily_price_sigma`` where
    ``daily_price_sigma = (garch_vol_annual / sqrt(ann_factor)) * close``.
    A longer-window SMA filters trend direction: longs entered only in
    bullish regimes, shorts only in bearish regimes.
    """

    # GARCH refits cheaply per fold with no scaler/feature contract worth
    # pinning — pretrained-leaf injection unused. ``normalize_pretrained_leaves``
    # raises on any non-empty map.
    _leaf_keys: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        window: int = 20,
        k: float = 2.0,
        trend_window: int = 100,
        garch_p_max: int = 5,
        garch_q_max: int = 5,
        interval: Interval = Interval.DAILY,
        *,
        pretrained_leaves: Mapping[str, object] | None = None,
    ) -> None:
        if window < 2:
            raise ValueError(
                f"window must be >= 2, got {window}; fix by passing a band "
                f"window of at least 2 bars (typical: 20)."
            )
        if trend_window < 2:
            raise ValueError(
                f"trend_window must be >= 2, got {trend_window}; fix by passing "
                f"a long-term MA window of at least 2 bars (typical: 100-200)."
            )
        if k <= 0:
            raise ValueError(
                f"k must be > 0, got {k}; fix by passing a strictly positive "
                f"band-width multiplier (typical: 2.0 for ~95% confidence)."
            )

        self._pretrained_leaves = normalize_pretrained_leaves(
            pretrained_leaves, self._leaf_keys, type(self).__name__
        )

        self._window = window
        self._k = k
        self._trend_window = trend_window
        self._interval = interval
        # Retained so ``save()`` can snapshot the ctor kwargs without
        # reaching into the leaf GARCH's private state.
        self._garch_p_max = garch_p_max
        self._garch_q_max = garch_q_max

        self._garch = GARCHPredictor(p_max=garch_p_max, q_max=garch_q_max, interval=interval)
        self._cpp_strategy = quant_engine.AdaptiveBollingerStrategy(
            quant_engine.AdaptiveBollingerStrategy.Config(
                band_window=self._window,
                k=self._k,
                trend_window=self._trend_window,
            )
        )

    def train(
        self,
        train_data: pd.DataFrame,
        *,
        checkpoint_path: Path | None = None,  # noqa: ARG002
        **kwargs: object,
    ) -> None:
        """Fit the GARCH volatility model on training log returns."""
        log_returns = compute_log_returns(train_data["close"]).dropna()
        aligned = train_data.loc[log_returns.index]
        self._garch.fit(aligned, log_returns)

        self._set_fitted_with_metadata(
            TrainingMetadata.from_fit(train_data, self._interval, ("close",))
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Produce {-1, 0, +1} position signals. Leading warmup bars are NaN."""
        self._assert_fitted_with_metadata()

        close = data["close"]
        garch_vol_annual = self._garch.predict(data)
        ann_factor_sqrt = math.sqrt(self._interval.annualization_factor())
        daily_price_sigma = (garch_vol_annual / ann_factor_sqrt) * close

        signal = self._cpp_strategy.generate_signals(
            close=np.asarray(close, dtype=np.float64),
            cond_vol=np.asarray(daily_price_sigma, dtype=np.float64),
        )
        return pd.Series(signal, index=data.index, name="adaptive_bollinger_signal")

    def save(self, path: str | Path) -> None:
        """Persist AdaptiveBollinger config + nested GARCH to ``path``."""
        metadata = self._assert_fitted_with_metadata()

        def write_weights(root: Path) -> None:
            self._garch.save(root / GARCH_SUBDIR)

        save_model_skeleton(
            path,
            config=self._ctor_kwargs_as_json(),
            training_metadata=metadata,
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
        instance._set_fitted_with_metadata(TrainingMetadata.from_dict(metadata))
        return instance

    @property
    def name(self) -> str:
        return _STRATEGY_NAME

    @property
    def required_warmup_bars(self) -> int:
        return max(self._window, self._trend_window)

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        """Expose strategy + owned GARCH metadata for the deep leakage check."""
        return collect_metadata(
            ("strategy", self._training_metadata),
            ("garch", self._garch.training_metadata),
        )

    @staticmethod
    def suggest_params(trial: optuna.trial.BaseTrial) -> dict[str, object]:
        """Optuna search space for AdaptiveBollinger hyperparameters."""
        return {
            "window": trial.suggest_int("bollinger_window", 10, 50),
            "k": trial.suggest_float("bollinger_k", 1.0, 3.0),
            "trend_window": trial.suggest_int("bollinger_trend_window", 50, 200),
            "garch_p_max": trial.suggest_int("bollinger_garch_p_max", 1, 5),
            "garch_q_max": trial.suggest_int("bollinger_garch_q_max", 1, 5),
        }
