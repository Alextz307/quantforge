"""Momentum strategy gated by a long-horizon trend filter and a classifier."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.core.registry import strategy_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Device, Interval
from src.features.pipeline import FeatureEngineeringPipeline
from src.models.xgboost_classifier import DirectionalClassifier
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


@strategy_registry.register("MomentumGatekeeper")
class MomentumGatekeeperStrategy(IStrategy):
    """Long-only momentum strategy gated by a trend MA and a directional classifier.

    Pipeline:
      1. ``FeatureEngineeringPipeline`` produces standard features (returns,
         rolling vol, MA ratio, RSI, MACD triplet).
      2. ``DirectionalClassifier`` (XGBoost) predicts P(next close > this close).
      3. Signal = 1 iff ``close > SMA(close, ma_window)`` AND
         ``P(up) > prob_threshold``; else 0.
    """

    def __init__(
        self,
        ma_window: int = 50,
        prob_threshold: float = 0.55,
        feature_columns: list[str] | None = None,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        vol_window: int = 20,
        ma_ratio_window: int = 20,
        short_return_period: int = 5,
        long_return_period: int = 21,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        val_split_ratio: float = 0.2,
        device: Device | None = None,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if ma_window < 2:
            raise ValueError(f"ma_window must be >= 2, got {ma_window}")
        if not (0.0 < prob_threshold < 1.0):
            raise ValueError(f"prob_threshold must be in (0, 1), got {prob_threshold}")
        if macd_fast >= macd_slow:
            raise ValueError(
                f"macd_fast must be < macd_slow, got fast={macd_fast}, slow={macd_slow}"
            )

        self._ma_window = ma_window
        self._prob_threshold = prob_threshold
        self._configured_feature_columns = (
            list(feature_columns) if feature_columns is not None else None
        )
        self._rsi_period = rsi_period
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._vol_window = vol_window
        self._ma_ratio_window = ma_ratio_window
        self._short_return_period = short_return_period
        self._long_return_period = long_return_period
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._val_split_ratio = val_split_ratio
        # Unresolved preference — DirectionalClassifier calls select_xgboost_device(...)
        # when `train()` instantiates it, so resolution/validation is deferred to that point.
        self._device_preference = device
        self._interval = interval

        self._pipeline = self._build_pipeline()
        self._classifier: DirectionalClassifier | None = None
        self._resolved_feature_columns: list[str] = []
        self._fitted = False
        self._training_metadata: TrainingMetadata | None = None

    def _build_pipeline(self) -> FeatureEngineeringPipeline:
        return FeatureEngineeringPipeline(
            rsi_period=self._rsi_period,
            macd_fast=self._macd_fast,
            macd_slow=self._macd_slow,
            macd_signal=self._macd_signal,
            vol_window=self._vol_window,
            ma_ratio_window=self._ma_ratio_window,
            short_return_period=self._short_return_period,
            long_return_period=self._long_return_period,
        )

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        """Fit feature pipeline + directional classifier on training data."""
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(train_data)
        features = self._pipeline.transform(train_data)

        resolved = self._configured_feature_columns
        if resolved is None:
            resolved = list(features.columns)
        else:
            missing = set(resolved) - set(features.columns)
            if missing:
                raise ValueError(
                    f"feature_columns {sorted(missing)} not produced by pipeline "
                    f"(available: {list(features.columns)})"
                )
        self._resolved_feature_columns = resolved

        close = train_data["close"]
        direction = (close.shift(-1) > close).astype(int).iloc[:-1]
        features_aligned = features.iloc[:-1]

        valid_mask = features_aligned[resolved].notna().all(axis=1)
        features_ready = features_aligned.loc[valid_mask]
        target_ready = direction.loc[valid_mask]

        self._classifier = DirectionalClassifier(
            feature_columns=resolved,
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            max_depth=self._max_depth,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            val_split_ratio=self._val_split_ratio,
            device=self._device_preference,
            interval=self._interval,
        )
        self._classifier.fit(features_ready, target_ready)

        self._training_metadata = TrainingMetadata.from_fit(
            train_data, self._interval, tuple(resolved)
        )
        self._fitted = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Produce {0, 1} long-only signals. Bars with NaN features stay NaN."""
        if not self._fitted or self._classifier is None:
            raise RuntimeError(
                "MomentumGatekeeperStrategy.generate_signals() called before train()"
            )

        features = self._pipeline.transform(data)[self._resolved_feature_columns]
        valid_mask = features.notna().all(axis=1)

        prob_up = pd.Series(np.nan, index=data.index, name="up_prob")
        if valid_mask.any():
            prob_valid = self._classifier.predict_proba(features.loc[valid_mask])
            prob_up.loc[prob_valid.index] = prob_valid

        trend_ma = data["close"].rolling(self._ma_window).mean()
        is_bull = data["close"] > trend_ma

        raw_signal = (is_bull & (prob_up > self._prob_threshold)).astype(float)
        signal = raw_signal.where(trend_ma.notna() & prob_up.notna(), np.nan)
        signal.name = "momentum_gatekeeper_signal"
        return signal

    @property
    def name(self) -> str:
        return "MomentumGatekeeper"

    @property
    def required_warmup_bars(self) -> int:
        return max(self._ma_window, self._pipeline.hard_nan_warmup_bars)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for MomentumGatekeeper hyperparameters."""
        macd_fast = trial.suggest_int("momentum_macd_fast", 8, 16)
        macd_slow = trial.suggest_int("momentum_macd_slow", macd_fast + 4, 40)
        short_return_period = trial.suggest_int("momentum_short_return_period", 3, 10)
        long_return_period = trial.suggest_int(
            "momentum_long_return_period", short_return_period + 5, 40
        )
        return {
            "ma_window": trial.suggest_int("momentum_ma_window", 20, 100),
            "prob_threshold": trial.suggest_float("momentum_prob_threshold", 0.5, 0.7),
            "rsi_period": trial.suggest_int("momentum_rsi_period", 7, 28),
            "macd_fast": macd_fast,
            "macd_slow": macd_slow,
            "macd_signal": trial.suggest_int("momentum_macd_signal", 5, 12),
            "vol_window": trial.suggest_int("momentum_vol_window", 10, 40),
            "ma_ratio_window": trial.suggest_int("momentum_ma_ratio_window", 10, 50),
            "short_return_period": short_return_period,
            "long_return_period": long_return_period,
            "n_estimators": trial.suggest_int("momentum_n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("momentum_lr", 1e-3, 3e-1, log=True),
            "max_depth": trial.suggest_int("momentum_max_depth", 3, 8),
        }
