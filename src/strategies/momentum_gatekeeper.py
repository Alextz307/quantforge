"""Momentum strategy gated by a long-horizon trend filter and a classifier."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self, cast

import numpy as np
import pandas as pd

from src.core import json_io
from src.core.leaf_keys import LEAF_KEY_DIRECTIONAL_CLASSIFIER
from src.core.logging import get_logger
from src.core.persistence import (
    CLASSIFIER_SUBDIR,
    CONFIG_JSON,
    METADATA_JSON,
    PIPELINE_SCALER_JSON,
    frozen_params_to_json,
    load_standard_scaler,
    save_model_skeleton,
    save_standard_scaler,
)
from src.core.registry import strategy_registry
from src.core.temporal import (
    TrackedMetadata,
    TrainingMetadata,
    collect_metadata,
)
from src.core.types import Device, Interval
from src.core.utils import align_features_for_directional_target, validate_open_unit_interval
from src.features.pipeline import FeatureEngineeringPipeline
from src.models.xgboost_classifier import DirectionalClassifier
from src.orchestration.pretrained_leaves import (
    normalize_pretrained_leaves,
    validate_pretrained_leaf,
)
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = get_logger(__name__)


@dataclass(frozen=True)
class _MomentumConfig:
    """Frozen snapshot of every ``MomentumGatekeeperStrategy.__init__`` kwarg.

    One source of truth for save/load + drift-guard tests. ``feature_columns``
    is ``tuple[str, ...] | None`` so ``frozen=True`` actually guarantees
    immutability (a list field would still be mutable). Field names MUST
    mirror the ctor param names.
    """

    ma_window: int
    prob_threshold: float
    feature_columns: tuple[str, ...] | None
    rsi_period: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    vol_window: int
    ma_ratio_window: int
    short_return_period: int
    long_return_period: int
    n_estimators: int
    learning_rate: float
    max_depth: int
    subsample: float
    colsample_bytree: float
    val_split_ratio: float
    device: Device | None
    interval: Interval


@strategy_registry.register("MomentumGatekeeper")
class MomentumGatekeeperStrategy(IStrategy):
    """Long-only momentum strategy gated by a trend MA and a directional classifier.

    Pipeline:
      1. ``FeatureEngineeringPipeline`` produces standard features (returns,
         rolling vol, MA ratio, RSI, MACD triplet).
      2. ``DirectionalClassifier`` (XGBoost) predicts P(next close > this close).
      3. Signal = 1 iff ``close > SMA(close, ma_window)`` AND
         ``P(up) > prob_threshold``; else 0.

    Supports pretrained-leaf injection: passing
    ``pretrained_leaves={"directional_classifier": loaded_classifier}``
    freezes the classifier across every ``train()`` call. The feature
    pipeline's ``StandardScaler`` re-fits per fold (the pipeline is not
    part of the leaf artifact) — the classifier's decision boundaries
    are pinned, but the numerical scaling of inputs shifts by the small
    amount the fold's training distribution differs from the leaf's.
    For the thesis-scale folds this drift is negligible relative to the
    gain of a stable leaf comparison; strategies that need a fully-frozen
    feature distribution should use the hybrid composites where the
    scaler is internal to the leaf.
    """

    _leaf_keys: ClassVar[frozenset[str]] = frozenset({LEAF_KEY_DIRECTIONAL_CLASSIFIER})

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
        pretrained_leaves: Mapping[str, object] | None = None,
    ) -> None:
        if ma_window < 2:
            raise ValueError(
                f"ma_window must be >= 2, got {ma_window}; fix by passing a "
                f"long-term MA window of at least 2 bars (typical: 200)."
            )
        validate_open_unit_interval(prob_threshold, "prob_threshold")
        if macd_fast >= macd_slow:
            raise ValueError(
                f"macd_fast must be < macd_slow, got fast={macd_fast}, "
                f"slow={macd_slow}; fix by lowering macd_fast (typical: 12) "
                f"or raising macd_slow (typical: 26)."
            )

        self._pretrained_leaves = normalize_pretrained_leaves(
            pretrained_leaves, self._leaf_keys, type(self).__name__
        )

        resolved_feature_columns = feature_columns
        self._classifier: DirectionalClassifier | None = None
        if LEAF_KEY_DIRECTIONAL_CLASSIFIER in self._pretrained_leaves:
            injected = self._pretrained_leaves[LEAF_KEY_DIRECTIONAL_CLASSIFIER]
            # Auto-adopt feature_columns from the leaf's training_metadata
            # when the user left the strategy default ``None`` — spares
            # them from spelling the same list in both the standalone-model
            # YAML and the strategy YAML. ``validate_pretrained_leaf`` below
            # raises on missing metadata or column mismatch.
            if resolved_feature_columns is None:
                meta = getattr(injected, "training_metadata", None)
                resolved_feature_columns = list(meta.feature_columns) if meta is not None else []
            validate_pretrained_leaf(
                injected,
                interval=interval,
                feature_columns=resolved_feature_columns,
            )
            self._classifier = cast(DirectionalClassifier, injected)

        self._params = _MomentumConfig(
            ma_window=ma_window,
            prob_threshold=prob_threshold,
            feature_columns=(
                tuple(resolved_feature_columns) if resolved_feature_columns is not None else None
            ),
            rsi_period=rsi_period,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            vol_window=vol_window,
            ma_ratio_window=ma_ratio_window,
            short_return_period=short_return_period,
            long_return_period=long_return_period,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            val_split_ratio=val_split_ratio,
            # Unresolved preference — DirectionalClassifier calls select_xgboost_device(...)
            # when train() instantiates it, so resolution/validation is deferred.
            device=device,
            interval=interval,
        )

        self._pipeline = self._build_pipeline()
        self._resolved_feature_columns: list[str] = (
            list(resolved_feature_columns) if resolved_feature_columns is not None else []
        )

    def _build_pipeline(self) -> FeatureEngineeringPipeline:
        return FeatureEngineeringPipeline(
            rsi_period=self._params.rsi_period,
            macd_fast=self._params.macd_fast,
            macd_slow=self._params.macd_slow,
            macd_signal=self._params.macd_signal,
            vol_window=self._params.vol_window,
            ma_ratio_window=self._params.ma_ratio_window,
            short_return_period=self._params.short_return_period,
            long_return_period=self._params.long_return_period,
        )

    def _build_classifier_batch(
        self, data: pd.DataFrame, resolved: list[str]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Compute valid-row features + next-bar direction target from ``data``."""
        features = self._pipeline.transform(data)[resolved]
        return align_features_for_directional_target(features, data["close"])

    def train(
        self,
        train_data: pd.DataFrame,
        *,
        checkpoint_path: Path | None = None,
        **kwargs: object,
    ) -> None:
        """Fit feature pipeline + directional classifier on training data.

        When ``pretrained_leaves["directional_classifier"]`` was injected, the
        classifier stays frozen (no rebuild, no ``fit()``); only the pipeline
        scaler and ``_training_metadata`` advance.
        """
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(train_data)
        features = self._pipeline.transform(train_data)

        if self._params.feature_columns is None:
            resolved = list(features.columns)
        else:
            resolved = list(self._params.feature_columns)
            missing = set(resolved) - set(features.columns)
            if missing:
                raise ValueError(
                    f"feature_columns {sorted(missing)} not produced by pipeline "
                    f"(available: {list(features.columns)}); fix by removing the "
                    f"unknown names from feature_columns or by extending the "
                    f"pipeline to emit them."
                )
        self._resolved_feature_columns = resolved

        if LEAF_KEY_DIRECTIONAL_CLASSIFIER not in self._pretrained_leaves:
            features_ready, target_ready = self._build_classifier_batch(train_data, resolved)

            self._classifier = DirectionalClassifier(
                feature_columns=resolved,
                n_estimators=self._params.n_estimators,
                learning_rate=self._params.learning_rate,
                max_depth=self._params.max_depth,
                subsample=self._params.subsample,
                colsample_bytree=self._params.colsample_bytree,
                val_split_ratio=self._params.val_split_ratio,
                device=self._params.device,
                interval=self._params.interval,
            )
            self._classifier.fit(features_ready, target_ready, checkpoint_path=checkpoint_path)

        self._set_fitted_with_metadata(
            TrainingMetadata.from_fit(train_data, self._params.interval, tuple(resolved))
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Produce {0, 1} long-only signals. Bars with NaN features stay NaN."""
        self._assert_fitted_with_metadata()
        if self._classifier is None:
            raise RuntimeError(
                "MomentumGatekeeperStrategy.generate_signals() invoked with no "
                "classifier wired; fix by checking the pretrained-leaf injection "
                "or by re-running train(train_data)."
            )

        features = self._pipeline.transform(data)[self._resolved_feature_columns]
        valid_mask = features.notna().all(axis=1)

        prob_up = pd.Series(np.nan, index=data.index, name="up_prob")
        if valid_mask.any():
            prob_valid = self._classifier.predict_proba(features.loc[valid_mask])
            prob_up.loc[prob_valid.index] = prob_valid

        trend_ma = data["close"].rolling(self._params.ma_window).mean()
        is_bull = data["close"] > trend_ma

        raw_signal = (is_bull & (prob_up > self._params.prob_threshold)).astype(float)
        signal = raw_signal.where(trend_ma.notna() & prob_up.notna(), np.nan)
        signal.name = "momentum_gatekeeper_signal"
        return signal

    def save(self, path: str | Path) -> None:
        """Persist MomentumGatekeeper config + nested DirectionalClassifier.

        Feature-pipeline hyperparams live in this config alongside strategy
        hyperparams — the pipeline is stateless (no scaler to round-trip),
        it's rebuilt on load from the same ctor kwargs. Device preference is
        NOT persisted: the classifier subdir captures whatever device was
        resolved at fit time, and on load we defer to the classifier's own
        device re-resolution.
        """
        metadata = self._assert_fitted_with_metadata()
        # ``_classifier`` is set atomically with metadata in train() — assert for mypy.
        assert self._classifier is not None

        classifier = self._classifier
        pipeline_scaler = self._pipeline.scaler
        if pipeline_scaler is None:
            raise RuntimeError(
                "MomentumGatekeeperStrategy.save() found an unfitted feature "
                "pipeline; fix by calling strategy.train(train_data) before save()."
            )

        def write_weights(root: Path) -> None:
            classifier.save(root / CLASSIFIER_SUBDIR)
            save_standard_scaler(pipeline_scaler, root / PIPELINE_SCALER_JSON)

        save_model_skeleton(
            path,
            config=self._ctor_kwargs_as_json(),
            training_metadata=metadata,
            write_weights=write_weights,
        )

    def _ctor_kwargs_as_json(self) -> dict[str, object]:
        """Snapshot of this strategy's constructor kwargs as JSON-ready values.

        Delegates tuple→list + Enum→value conversions to
        ``frozen_params_to_json``; ``device`` is dropped (re-resolved on load
        via ``select_xgboost_device()``). ``feature_columns`` is overwritten
        with ``_resolved_feature_columns`` — the list the classifier was
        actually fit on (the ctor's ``feature_columns`` kwarg, when ``None``,
        resolves to the pipeline's full column set at ``train()`` time).
        Post-fit the two are equivalent.
        """
        d = frozen_params_to_json(self._params, omit=("device",))
        d["feature_columns"] = list(self._resolved_feature_columns)
        return d

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a trained MomentumGatekeeperStrategy from ``path``.

        Narrow the strategy's ``config.json`` into ctor kwargs BEFORE loading
        the classifier + pipeline scaler — a corrupt composite config
        fast-fails with a named-field error rather than crashing deep inside
        a sub-loader.
        """
        root = Path(path)
        config = json_io.read_dict(root / CONFIG_JSON)
        metadata = json_io.read_dict(root / METADATA_JSON)
        resolved = json_io.get_str_list(config, "feature_columns")

        instance = cls(
            ma_window=json_io.get_int(config, "ma_window"),
            prob_threshold=json_io.get_float(config, "prob_threshold"),
            feature_columns=resolved,
            rsi_period=json_io.get_int(config, "rsi_period"),
            macd_fast=json_io.get_int(config, "macd_fast"),
            macd_slow=json_io.get_int(config, "macd_slow"),
            macd_signal=json_io.get_int(config, "macd_signal"),
            vol_window=json_io.get_int(config, "vol_window"),
            ma_ratio_window=json_io.get_int(config, "ma_ratio_window"),
            short_return_period=json_io.get_int(config, "short_return_period"),
            long_return_period=json_io.get_int(config, "long_return_period"),
            n_estimators=json_io.get_int(config, "n_estimators"),
            learning_rate=json_io.get_float(config, "learning_rate"),
            max_depth=json_io.get_int(config, "max_depth"),
            subsample=json_io.get_float(config, "subsample"),
            colsample_bytree=json_io.get_float(config, "colsample_bytree"),
            val_split_ratio=json_io.get_float(config, "val_split_ratio"),
            interval=Interval(json_io.get_str(config, "interval")),
        )
        instance._classifier = DirectionalClassifier.load(root / CLASSIFIER_SUBDIR)
        # Replace the pipeline's unfitted scaler with the loaded fitted one.
        # The pipeline itself is stateless aside from this scaler — reusing
        # the ctor-built pipeline means hyperparams (rsi_period, macd_*, …)
        # stay in one place.
        instance._pipeline.scaler = load_standard_scaler(root / PIPELINE_SCALER_JSON)
        instance._resolved_feature_columns = resolved
        instance._set_fitted_with_metadata(TrainingMetadata.from_dict(metadata))
        return instance

    @property
    def name(self) -> str:
        return "MomentumGatekeeper"

    @property
    def required_warmup_bars(self) -> int:
        return max(self._params.ma_window, self._pipeline.hard_nan_warmup_bars)

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        """Expose strategy + owned classifier metadata for the deep leakage check.

        When ``directional_classifier`` was pretrained-injected, the
        classifier entry gets ``is_pretrained=True`` so the walk-forward
        orchestrator enforces the strict-no-overlap invariant against the
        fold's train window (not just its test window).
        """
        classifier_meta = (
            self._classifier.training_metadata if self._classifier is not None else None
        )
        return self._build_strategy_plus_leaf_metadata(
            LEAF_KEY_DIRECTIONAL_CLASSIFIER,
            collect_metadata(("classifier", classifier_meta)),
        )

    @staticmethod
    def suggest_params(trial: optuna.trial.BaseTrial) -> dict[str, object]:
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
