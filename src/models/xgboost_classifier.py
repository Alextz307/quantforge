"""XGBoost directional classifier for predicting price direction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import xgboost as xgb

from src.core.registry import classifier_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.models.interface import IClassifier

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


@classifier_registry.register("xgboost_directional")
class DirectionalClassifier(IClassifier):
    """XGBoost classifier for predicting next-bar price direction.

    Uses early stopping on a temporal validation split (last 20% of
    training data) to prevent overfitting.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        early_stopping_rounds: int = 10,
        objective: str = "binary:logistic",
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        interval: Interval = Interval.DAILY,
    ) -> None:
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._early_stopping_rounds = early_stopping_rounds
        self._objective = objective
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._interval = interval

        self._fitted = False
        self._model: xgb.XGBClassifier | None = None
        self._feature_columns: list[str] = []
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
        self._feature_columns = list(train_data.columns)

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
        )

        # 80/20 temporal split for early stopping
        split_idx = int(len(train_data) * 0.8)
        x_train = train_data.iloc[:split_idx]
        y_train = target.iloc[:split_idx]
        x_val = train_data.iloc[split_idx:]
        y_val = target.iloc[split_idx:]

        if len(x_val) < 2:
            # Not enough data for validation — train on all data without early stopping
            self._model.fit(train_data, target, verbose=False)
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
        # Column 1 is the probability of class 1 (up)
        return pd.Series(proba[:, 1], index=data.index, name="up_prob")

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
