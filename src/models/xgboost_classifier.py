"""XGBoost directional classifier for predicting price direction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import pandas as pd
import xgboost as xgb

from src.core import json_io
from src.core.device import select_xgboost_device
from src.core.logging import get_logger
from src.core.persistence import (
    BEST_ITERATION_UBJ,
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

_PROGRESS_LOG_INTERVAL = 10
# XGBClassifier names the validation eval set ``"validation_0"`` when a single
# eval_set is passed to .fit() — checkpoint saves are gated on improvements to
# this set so train-set monotone improvement doesn't trigger a save every round.
_VALIDATION_EVAL_NAME = "validation_0"


class _ProgressAndCheckpointCallback(xgb.callback.TrainingCallback):
    """Log eval metrics every N rounds and dump the booster on val improvement.

    Encapsulates two responsibilities so we add one callback to the booster
    rather than two: every ``log_interval`` boosting rounds an INFO line lands
    with the latest eval metric for each (data, metric) pair, and a
    ``best_iteration.ubj`` snapshot is written into ``checkpoint_path``
    whenever the **validation** metric improves on the best-seen value. Train-
    set improvements are tracked for the log line but never trigger a save —
    train logloss is monotone in boosting rounds, so saving on it would
    persist the latest booster every round, defeating "best-on-val" semantics.
    """

    def __init__(
        self,
        *,
        log_interval: int,
        checkpoint_path: Path | None,
    ) -> None:
        super().__init__()
        self._log_interval = log_interval
        self._checkpoint_path = checkpoint_path
        # Lower-is-better: track the lowest seen validation value per metric.
        # Only validation entries land here — train entries don't gate saves
        # and their best is unused, so the data_name dimension is dropped.
        self._best: dict[str, float] = {}

    # ``model`` is typed ``Any`` upstream in XGBoost (Booster | CVPack) — we
    # must mirror that to satisfy LSP. We only call ``save_model`` which both
    # variants implement.
    def after_iteration(
        self,
        model: Any,
        epoch: int,
        evals_log: dict[str, dict[str, list[float] | list[tuple[float, float]]]],
    ) -> bool:
        val_improved = False
        should_log = (epoch + 1) % self._log_interval == 0
        # Build the human-readable "data-metric=value" parts only on log
        # rounds so non-log rounds don't pay for the f-string formatting.
        latest: list[str] = []
        for data_name, metrics in evals_log.items():
            is_validation = data_name == _VALIDATION_EVAL_NAME
            # Non-validation entries are tracked only to surface them in the
            # log line — skip them entirely on quiet rounds since save gating
            # depends on validation only.
            if not is_validation and not should_log:
                continue
            for metric_name, history in metrics.items():
                last = history[-1]
                # ``cv`` returns (mean, std) tuples; standard fit returns floats.
                value = float(last[0]) if isinstance(last, tuple) else float(last)
                if should_log:
                    latest.append(f"{data_name}-{metric_name}={value:.4f}")
                if is_validation:
                    prev_best = self._best.get(metric_name)
                    if prev_best is None or value < prev_best:
                        self._best[metric_name] = value
                        val_improved = True

        if should_log:
            logger.info("round %d %s", epoch + 1, " ".join(latest))

        if val_improved and self._checkpoint_path is not None:
            # ``str()`` because XGBoost's save_model accepts only str paths.
            model.save_model(str(self._checkpoint_path / BEST_ITERATION_UBJ))

        # Returning False signals "do not stop early" — XGBoost's own
        # early-stopping callback handles termination separately.
        return False


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
        device: Device | None = None,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if not feature_columns:
            raise ValueError(
                "DirectionalClassifier requires a non-empty feature_columns list; "
                "fix by passing the explicit list of feature names the classifier "
                "should consume (e.g. ['rsi_14', 'macd_hist'])."
            )
        validate_open_unit_interval(val_split_ratio, "val_split_ratio")
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._early_stopping_rounds = early_stopping_rounds
        self._objective = objective
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._val_split_ratio = val_split_ratio
        self._device = select_xgboost_device(device)
        self._interval = interval

        self._model: xgb.XGBClassifier | None = None
        self._feature_columns: list[str] = list(feature_columns)
        self._best_iteration: int = 0
        self._eval_results: dict[str, dict[str, list[float]]] = {}

    def fit(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        *,
        checkpoint_path: Path | None = None,
        **kwargs: object,
    ) -> None:
        """Fit XGBoost with early stopping on temporal validation split.

        Args:
            train_data: Feature DataFrame with DatetimeIndex.
            target: Binary target (1 = up, 0 = down).
            checkpoint_path: When set, write the booster to
                ``<checkpoint_path>/<BEST_ITERATION_UBJ>`` after every round
                where the **validation** metric improves. The directory is
                created on first write so a mid-fit interrupt leaves the
                best-so-far booster recoverable. No-op when the dataset is
                too short for a validation split.
            **kwargs: Unused (reserved for Optuna Trial passthrough).
        """
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

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
            callback = _ProgressAndCheckpointCallback(
                log_interval=_PROGRESS_LOG_INTERVAL,
                checkpoint_path=checkpoint_path,
            )
            # XGBClassifier expects callbacks via set_params, not fit(); the
            # fit-time keyword is reserved for the deprecated XGBoost-1.x form.
            self._model.set_params(callbacks=[callback])
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

        self._set_fitted_with_metadata(
            TrainingMetadata.from_fit(train_data, self._interval, tuple(self._feature_columns))
        )

    def predict_proba(self, data: pd.DataFrame) -> pd.Series:
        """Predict probability of upward move.

        Args:
            data: Feature DataFrame with same columns as training data.

        Returns:
            Series of probabilities in [0, 1].
        """
        self._assert_fitted_with_metadata()
        if self._model is None:
            raise RuntimeError(
                "DirectionalClassifier.predict_proba() invoked with no booster "
                "wired; fix by re-running classifier.fit(train_data, target)."
            )

        proba = self._model.predict_proba(data[self._feature_columns])
        return pd.Series(proba[:, 1], index=data.index, name="up_prob")

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict binary class labels (0 = down, 1 = up).

        Args:
            data: Feature DataFrame with same columns as training data.

        Returns:
            Series of binary predictions.
        """
        self._assert_fitted_with_metadata()
        if self._model is None:
            raise RuntimeError(
                "DirectionalClassifier.predict() invoked with no booster "
                "wired; fix by re-running classifier.fit(train_data, target)."
            )

        preds = self._model.predict(data[self._feature_columns])
        return pd.Series(preds, index=data.index, name="direction")

    def save(self, path: str | Path) -> None:
        """Persist classifier config + booster to ``path``.

        Uses XGBoost's native UBJSON format (``model.ubj``) for the booster —
        the most version-stable option available. Device is NOT persisted; it
        is re-resolved on load via ``select_xgboost_device()``.
        """
        metadata = self._assert_fitted_with_metadata()
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
        instance._set_fitted_with_metadata(TrainingMetadata.from_dict(metadata))
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
