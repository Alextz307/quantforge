"""XGBoost directional classifier for predicting price direction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

import pandas as pd
import xgboost as xgb

from src.core import json_io
from src.core.device import select_xgboost_device
from src.core.logging import get_logger
from src.core.persistence import (
    CONFIG_JSON,
    METADATA_JSON,
    MODEL_UBJ,
    save_model_skeleton,
)
from src.core.registry import classifier_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Device, Interval
from src.core.utils import validate_open_unit_interval
from src.models.interface import IClassifier

if TYPE_CHECKING:
    import optuna

logger = get_logger(__name__)


@classifier_registry.register("xgboost_directional")
class DirectionalClassifier(IClassifier):
    """XGBoost classifier for predicting next-bar price direction.

    Uses early stopping on a temporal validation split (trailing
    ``val_split_ratio`` fraction of training data, default 20%).
    """

    def __init__(
        self,
        feature_columns: list[str],
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        early_stopping_rounds: int = 10,
        objective: str = "binary:logistic",
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        val_split_ratio: float = 0.2,
        update_n_estimators: int = 10,
        device: Device | None = None,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if not feature_columns:
            raise ValueError("DirectionalClassifier requires a non-empty feature_columns list")
        validate_open_unit_interval(val_split_ratio, "val_split_ratio")
        if update_n_estimators < 1:
            raise ValueError(f"update_n_estimators must be >= 1, got {update_n_estimators}")
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._early_stopping_rounds = early_stopping_rounds
        self._objective = objective
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._val_split_ratio = val_split_ratio
        self._update_n_estimators = update_n_estimators
        self._device = select_xgboost_device(device)
        self._interval = interval

        self._fitted = False
        self._model: xgb.XGBClassifier | None = None
        self._feature_columns: list[str] = list(feature_columns)
        self._best_iteration: int = 0
        self._eval_results: dict[str, dict[str, list[float]]] = {}
        self._training_metadata: TrainingMetadata | None = None

    def fit(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        **kwargs: object,
    ) -> None:
        """Fit XGBoost with early stopping on temporal validation split.

        Args:
            train_data: Feature DataFrame with DatetimeIndex.
            target: Binary target (1 = up, 0 = down).
            **kwargs: Unused (reserved for Optuna Trial passthrough).
        """
        self._model = xgb.XGBClassifier(
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            max_depth=self._max_depth,
            objective=self._objective,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            # `hist` is required on CUDA and is also the recommended CPU backend.
            tree_method="hist",
            device=self._device,
        )

        features = train_data[self._feature_columns]

        split_idx = int(len(features) * (1.0 - self._val_split_ratio))
        x_train = features.iloc[:split_idx]
        y_train = target.iloc[:split_idx]
        x_val = features.iloc[split_idx:]
        y_val = target.iloc[split_idx:]

        if len(x_val) < 2:
            # Not enough data for validation — train on all data without early stopping
            self._model.fit(features, target, verbose=False)
        else:
            self._model.set_params(early_stopping_rounds=self._early_stopping_rounds)
            self._model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                verbose=False,
            )
            try:
                self._best_iteration = int(self._model.best_iteration)
            except AttributeError:
                self._best_iteration = self._n_estimators
            results = self._model.evals_result()
            self._eval_results = {k: dict(v) for k, v in results.items()}

        self._fitted = True
        self._training_metadata = TrainingMetadata.from_fit(
            train_data, self._interval, tuple(self._feature_columns)
        )

    def predict_proba(self, data: pd.DataFrame) -> pd.Series:
        """Predict probability of upward move.

        Args:
            data: Feature DataFrame with same columns as training data.

        Returns:
            Series of probabilities in [0, 1].
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("DirectionalClassifier.predict_proba() called before fit()")

        proba = self._model.predict_proba(data[self._feature_columns])
        return pd.Series(proba[:, 1], index=data.index, name="up_prob")

    def update(
        self,
        new_data: pd.DataFrame,
        target: pd.Series,
        **kwargs: object,
    ) -> None:
        """Continue-boost: append ``update_n_estimators`` trees to the existing booster.

        Uses XGBoost's native ``xgb_model=<existing booster>`` continue-boosting
        API — no scaler or feature-schema refit. The existing booster is the
        starting point; ``update_n_estimators`` additional rounds are trained
        on ``new_data``. See :meth:`IClassifier.update` for the shared contract.
        """
        metadata = self._assert_fitted_with_metadata(caller="update")
        # ``_model`` is set atomically with metadata in fit() — assert for mypy.
        assert self._model is not None
        new_metadata = metadata.extend_from(new_data)

        existing_booster = self._model.get_booster()
        refreshed = xgb.XGBClassifier(
            n_estimators=self._update_n_estimators,
            learning_rate=self._learning_rate,
            max_depth=self._max_depth,
            objective=self._objective,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            tree_method="hist",
            device=self._device,
        )
        refreshed.fit(
            new_data[self._feature_columns],
            target,
            xgb_model=existing_booster,
            verbose=False,
        )
        self._model = refreshed
        self._training_metadata = new_metadata

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict binary class labels (0 = down, 1 = up).

        Args:
            data: Feature DataFrame with same columns as training data.

        Returns:
            Series of binary predictions.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("DirectionalClassifier.predict() called before fit()")

        preds = self._model.predict(data[self._feature_columns])
        return pd.Series(preds, index=data.index, name="direction")

    def save(self, path: str | Path) -> None:
        """Persist classifier config + booster to ``path``.

        Uses XGBoost's native UBJSON format (``model.ubj``) for the booster —
        the most version-stable option available. Device is NOT persisted; it
        is re-resolved on load via ``select_xgboost_device()``.
        """
        metadata = self._assert_fitted_with_metadata(caller="save")
        # ``_model`` is set atomically with metadata in fit() — assert for mypy.
        assert self._model is not None
        model = self._model

        def write_weights(root: Path) -> None:
            # XGBClassifier.save_model requires ``_estimator_type`` (a sklearn-side
            # attribute that's inconsistently populated across xgboost versions).
            # The booster-level save is stable: ``XGBClassifier.load_model`` accepts
            # a booster-only UBJ on the reconstruct side. ``best_iteration`` is
            # baked into the booster itself, so no separate weights.json is needed.
            model.get_booster().save_model(str(root / MODEL_UBJ))

        save_model_skeleton(
            path,
            config={
                "feature_columns": list(self._feature_columns),
                "n_estimators": self._n_estimators,
                "learning_rate": self._learning_rate,
                "max_depth": self._max_depth,
                "early_stopping_rounds": self._early_stopping_rounds,
                "objective": self._objective,
                "subsample": self._subsample,
                "colsample_bytree": self._colsample_bytree,
                "val_split_ratio": self._val_split_ratio,
                "update_n_estimators": self._update_n_estimators,
                "interval": self._interval.value,
            },
            training_metadata=metadata,
            write_weights=write_weights,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a fitted DirectionalClassifier from ``path``.

        Only ``device`` is threaded through to the fresh XGBClassifier — every
        other hyperparameter is baked into the booster UBJ and re-setting them
        on the wrapper would just be discarded the moment ``load_model`` runs.
        """
        root = Path(path)
        config = json_io.read_dict(root / CONFIG_JSON)
        metadata = json_io.read_dict(root / METADATA_JSON)

        instance = cls(
            feature_columns=json_io.get_str_list(config, "feature_columns"),
            n_estimators=json_io.get_int(config, "n_estimators"),
            learning_rate=json_io.get_float(config, "learning_rate"),
            max_depth=json_io.get_int(config, "max_depth"),
            early_stopping_rounds=json_io.get_int(config, "early_stopping_rounds"),
            objective=json_io.get_str(config, "objective"),
            subsample=json_io.get_float(config, "subsample"),
            colsample_bytree=json_io.get_float(config, "colsample_bytree"),
            val_split_ratio=json_io.get_float(config, "val_split_ratio"),
            update_n_estimators=json_io.get_int(config, "update_n_estimators"),
            interval=Interval(json_io.get_str(config, "interval")),
        )
        model = xgb.XGBClassifier(device=instance._device)
        model.load_model(str(root / MODEL_UBJ))
        instance._model = model
        # ``best_iteration`` is only populated on the booster when early
        # stopping fired during fit. ``hasattr`` already swallows the
        # ``AttributeError`` XGBoost raises otherwise, so no try/except needed.
        instance._best_iteration = (
            int(model.best_iteration)
            if hasattr(model, "best_iteration")
            else instance._n_estimators
        )
        instance._training_metadata = TrainingMetadata.from_dict(metadata)
        instance._fitted = True
        return instance

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for XGBoost hyperparameters."""
        return {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
            "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
            "objective": trial.suggest_categorical(
                "xgb_objective", ["binary:logistic", "binary:hinge"]
            ),
        }
