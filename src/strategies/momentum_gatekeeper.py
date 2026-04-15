"""Momentum strategy gated by a long-horizon trend filter and a classifier."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.core.registry import strategy_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.features.pipeline import FeatureEngineeringPipeline
from src.models.xgboost_classifier import DirectionalClassifier
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

# return_21d (pct_change window=21) is the longest hard-NaN warmup in FeaturePipeline
_FEATURE_PIPELINE_WARMUP = 21


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
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if ma_window < 2:
            raise ValueError(f"ma_window must be >= 2, got {ma_window}")
        if not (0.0 < prob_threshold < 1.0):
            raise ValueError(f"prob_threshold must be in (0, 1), got {prob_threshold}")

        self._ma_window = ma_window
        self._prob_threshold = prob_threshold
        self._configured_feature_columns = (
            list(feature_columns) if feature_columns is not None else None
        )
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._interval = interval

        self._pipeline = FeatureEngineeringPipeline()
        self._classifier: DirectionalClassifier | None = None
        self._resolved_feature_columns: list[str] = []
        self._fitted = False
        self._training_metadata: TrainingMetadata | None = None

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        """Fit feature pipeline + directional classifier on training data."""
        self._pipeline = FeatureEngineeringPipeline()
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
        return max(self._ma_window, _FEATURE_PIPELINE_WARMUP)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for MomentumGatekeeper hyperparameters."""
        return {
            "ma_window": trial.suggest_int("momentum_ma_window", 20, 100),
            "prob_threshold": trial.suggest_float("momentum_prob_threshold", 0.5, 0.7),
            "n_estimators": trial.suggest_int("momentum_n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("momentum_lr", 1e-3, 3e-1, log=True),
            "max_depth": trial.suggest_int("momentum_max_depth", 3, 8),
        }
