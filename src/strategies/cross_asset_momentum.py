"""Cross-asset momentum strategy: XGBoost on lagged returns of feature tickers.

Methodology adapted from Rapach, Strauss, Tu, and Zhou (2019), "Industry return
predictability: A machine learning approach", *Journal of Financial Economics*
135(2). The paper uses LASSO over lagged cross-industry returns to predict each
industry's next-month return; we retain the cross-asset-lagged-returns → ML
predictor → directional-trade pipeline but substitute gradient-boosted trees
(Krauss, Do, and Huck 2017, *European Journal of Operational Research* 259) for
the linear predictor and operate on user-configured ``lags`` × ``feature_tickers``
rather than the paper's fixed 1-month horizon over 30 industries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
import pandas as pd

from src.core import json_io
from src.core.logging import get_logger
from src.core.persistence import (
    CLASSIFIER_SUBDIR,
    CONFIG_JSON,
    METADATA_JSON,
    frozen_params_to_json,
    save_model_skeleton,
)
from src.core.registry import strategy_registry
from src.core.temporal import (
    TrackedMetadata,
    TrainingMetadata,
    collect_metadata,
)
from src.core.types import Device, Interval
from src.core.utils import align_features_for_directional_target, compute_log_returns
from src.models.xgboost_classifier import DirectionalClassifier
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna

logger = get_logger(__name__)

_THRESHOLD_LOWER_BOUND = 0.5
_THRESHOLD_UPPER_BOUND = 1.0

_LOG_RETURN_WARMUP = 1


@dataclass(frozen=True)
class _CrossAssetMomentumConfig:
    """Frozen snapshot of every ``CrossAssetMomentumStrategy.__init__`` kwarg.

    ``feature_tickers`` and ``lags`` are tuples (not lists) so ``frozen=True``
    actually guarantees immutability. Field names MUST mirror the ctor param
    names — :func:`tests.conftest.assert_params_match_constructor` drift-guards
    this.
    """

    primary_ticker: str
    feature_tickers: tuple[str, ...]
    lags: tuple[int, ...]
    direction_threshold: float
    n_estimators: int
    learning_rate: float
    max_depth: int
    subsample: float
    colsample_bytree: float
    val_split_ratio: float
    device: Device | None
    interval: Interval


def _derive_feature_columns(feature_tickers: Sequence[str], lags: Sequence[int]) -> list[str]:
    """Compute the deterministic feature-column names for ``(tickers × lags)``.

    Single source of truth so ctor validation and ``train`` feature-frame
    construction cannot drift. Order: outer loop over tickers (input
    order), inner loop over lags (input order).
    """
    return [f"lag{lag}_{ticker}" for ticker in feature_tickers for lag in lags]


@strategy_registry.register("CrossAssetMomentum")
class CrossAssetMomentumStrategy(IStrategy):
    """Single-asset traded, multi-asset feature directional momentum.

    Pipeline:
      1. For each ``(feature_ticker, lag)`` pair, compute
         ``log_return(close_<ticker>).shift(lag)`` from the wide
         ``<ohlcv>_<TICKER>`` frame.
      2. ``DirectionalClassifier`` (XGBoost) predicts P(next close of the
         primary ticker > current close).
      3. Three-way signal: ``+1`` iff ``p_up > direction_threshold``,
         ``-1`` iff ``p_up < 1 - direction_threshold``, ``0`` otherwise.
         Warmup bars before the longest lag has a non-NaN value remain NaN.
    """

    is_multi_feature_strategy: ClassVar[bool] = True
    uses_xgboost: ClassVar[bool] = True

    def __init__(
        self,
        primary_ticker: str,
        feature_tickers: Sequence[str],
        lags: Sequence[int] = (1, 5, 21),
        direction_threshold: float = 0.55,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        val_split_ratio: float = 0.2,
        device: Device | None = None,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if not primary_ticker:
            raise ValueError(
                "primary_ticker must be a non-empty string; fix by passing the "
                "ticker symbol of the asset to trade (e.g. 'SPY')."
            )
        if not feature_tickers:
            raise ValueError(
                "feature_tickers must be a non-empty sequence; fix by listing "
                "at least one ticker whose lagged returns predict the primary "
                "(typical: a basket of cross-asset peers like ['QQQ', 'IWM'])."
            )
        feature_tickers_tup = tuple(feature_tickers)
        if len(set(feature_tickers_tup)) != len(feature_tickers_tup):
            raise ValueError(
                f"feature_tickers contains duplicates: {list(feature_tickers_tup)}; "
                f"fix by passing each ticker exactly once."
            )
        if not lags:
            raise ValueError(
                "lags must be a non-empty sequence; fix by listing at least one "
                "positive integer (typical for daily bars: [1, 5, 21] = 1d/1w/1m)."
            )
        lags_tup = tuple(lags)
        if any(lag <= 0 for lag in lags_tup):
            raise ValueError(
                f"lags must be strictly positive integers, got {list(lags_tup)}; "
                f"fix by removing any lag <= 0 (lag 0 would leak the same-bar "
                f"return into the predictor)."
            )
        if len(set(lags_tup)) != len(lags_tup):
            raise ValueError(
                f"lags contains duplicates: {list(lags_tup)}; fix by passing each lag exactly once."
            )
        if not (_THRESHOLD_LOWER_BOUND <= direction_threshold < _THRESHOLD_UPPER_BOUND):
            raise ValueError(
                f"direction_threshold must be in "
                f"[{_THRESHOLD_LOWER_BOUND}, {_THRESHOLD_UPPER_BOUND}), "
                f"got {direction_threshold}; fix by passing a value at least "
                f"0.5 (so the long/short masks don't overlap) and strictly "
                f"below 1.0 (typical: 0.55 for a moderate confidence gate)."
            )

        feature_columns = _derive_feature_columns(feature_tickers_tup, lags_tup)
        self._classifier: DirectionalClassifier | None = None

        self._params = _CrossAssetMomentumConfig(
            primary_ticker=primary_ticker,
            feature_tickers=feature_tickers_tup,
            lags=lags_tup,
            direction_threshold=direction_threshold,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            val_split_ratio=val_split_ratio,
            device=device,
            interval=interval,
        )
        self._feature_columns = feature_columns

    def _build_feature_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute lagged log-return features for every (ticker, lag) pair.

        Resulting columns follow the deterministic order of
        :func:`_derive_feature_columns`. Leading rows are NaN (warmup) — the
        caller drops them before training and ignores them at inference.
        """
        cols: dict[str, pd.Series[float]] = {}
        for ticker in self._params.feature_tickers:
            log_ret = compute_log_returns(data[f"close_{ticker}"])
            for lag in self._params.lags:
                cols[f"lag{lag}_{ticker}"] = log_ret.shift(lag)
        return pd.DataFrame(cols)

    def train(
        self,
        train_data: pd.DataFrame,
        *,
        checkpoint_path: Path | None = None,
        **kwargs: object,
    ) -> None:
        """Fit the directional classifier on lagged cross-asset returns."""
        logger.info("%s train: %d bars", type(self).__name__, len(train_data))
        features = self._build_feature_frame(train_data)
        primary_close = train_data[f"close_{self._params.primary_ticker}"]
        features_ready, target_ready = align_features_for_directional_target(
            features, primary_close
        )

        self._classifier = DirectionalClassifier(
            feature_columns=list(self._feature_columns),
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
            TrainingMetadata.from_fit(
                train_data, self._params.interval, tuple(self._feature_columns)
            )
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Produce {-1, 0, +1} signals; warmup bars (any lag still NaN) stay NaN."""
        self._assert_fitted_with_metadata()
        if self._classifier is None:
            raise RuntimeError(
                "CrossAssetMomentumStrategy.generate_signals() invoked with no "
                "classifier wired; fix by re-running train(train_data)."
            )

        features = self._build_feature_frame(data)
        valid_mask = features.notna().all(axis=1)

        signal = pd.Series(np.nan, index=data.index, name="cross_asset_momentum_signal")
        if valid_mask.any():
            prob_up = self._classifier.predict_proba(features.loc[valid_mask]).to_numpy()
            t = self._params.direction_threshold
            sig_values = np.where(prob_up > t, 1.0, np.where(prob_up < 1.0 - t, -1.0, 0.0))
            signal.loc[valid_mask] = sig_values
        return signal

    def save(self, path: str | Path) -> None:
        """Persist strategy config + nested DirectionalClassifier under ``path``."""
        metadata = self._assert_fitted_with_metadata()
        assert self._classifier is not None
        classifier = self._classifier

        def write_weights(root: Path) -> None:
            classifier.save(root / CLASSIFIER_SUBDIR)

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
        via ``select_xgboost_device()`` inside ``DirectionalClassifier``).
        """
        return frozen_params_to_json(self._params, omit=("device",))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a trained CrossAssetMomentumStrategy from ``path``."""
        root = Path(path)
        config = json_io.read_dict(root / CONFIG_JSON)
        metadata = json_io.read_dict(root / METADATA_JSON)

        instance = cls(
            primary_ticker=json_io.get_str(config, "primary_ticker"),
            feature_tickers=json_io.get_str_list(config, "feature_tickers"),
            lags=json_io.get_int_list(config, "lags"),
            direction_threshold=json_io.get_float(config, "direction_threshold"),
            n_estimators=json_io.get_int(config, "n_estimators"),
            learning_rate=json_io.get_float(config, "learning_rate"),
            max_depth=json_io.get_int(config, "max_depth"),
            subsample=json_io.get_float(config, "subsample"),
            colsample_bytree=json_io.get_float(config, "colsample_bytree"),
            val_split_ratio=json_io.get_float(config, "val_split_ratio"),
            interval=Interval(json_io.get_str(config, "interval")),
        )
        instance._classifier = DirectionalClassifier.load(root / CLASSIFIER_SUBDIR)
        instance._set_fitted_with_metadata(TrainingMetadata.from_dict(metadata))
        return instance

    @property
    def name(self) -> str:
        return "CrossAssetMomentum"

    @property
    def primary_ticker(self) -> str:
        return self._params.primary_ticker

    @property
    def required_warmup_bars(self) -> int:
        return max(self._params.lags) + _LOG_RETURN_WARMUP

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        """Expose strategy + owned classifier metadata for the deep leakage check."""
        classifier_meta = (
            self._classifier.training_metadata if self._classifier is not None else None
        )
        return collect_metadata(
            ("strategy", self.training_metadata),
            ("classifier", classifier_meta),
        )

    @staticmethod
    def suggest_params(trial: optuna.trial.BaseTrial) -> dict[str, object]:
        """Optuna search space for CrossAssetMomentum hyperparameters.

        ``feature_tickers`` and ``lags`` stay fixed at YAML level — tuning the
        lag schedule per trial would change the classifier's input dimension,
        producing cross-trial Sharpe comparisons that aren't apples-to-apples.
        """
        return {
            "direction_threshold": trial.suggest_float(
                "cross_asset_direction_threshold", 0.50, 0.70
            ),
            "n_estimators": trial.suggest_int("cross_asset_n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("cross_asset_lr", 1e-3, 3e-1, log=True),
            "max_depth": trial.suggest_int("cross_asset_max_depth", 3, 8),
            "subsample": trial.suggest_float("cross_asset_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("cross_asset_colsample_bytree", 0.5, 1.0),
        }
