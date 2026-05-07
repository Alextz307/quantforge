"""Hybrid return predictor: ARMA mean forecast + LSTM residual correction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.core import json_io
from src.core.exceptions import guard_scaler_fit_once
from src.core.logging import get_logger, log_stage
from src.core.persistence import (
    ARMA_SUBDIR,
    CONFIG_JSON,
    LSTM_SUBDIR,
    METADATA_JSON,
    SCALER_JSON,
    frozen_params_to_json,
    load_standard_scaler,
    save_model_skeleton,
    save_standard_scaler,
)
from src.core.registry import model_registry
from src.core.temporal import TrackedMetadata, TrainingMetadata, collect_metadata
from src.core.types import Device, InformationCriterion, Interval, LossFunction
from src.core.utils import compute_log_returns
from src.models._hybrid_warmup import drop_feature_warmup
from src.models.arma import ARMAPredictor
from src.models.interface import IPredictor
from src.models.lstm import LSTMPredictor

logger = get_logger(__name__)

if TYPE_CHECKING:
    import optuna


@dataclass(frozen=True)
class _HybridReturnConfig:
    """Frozen snapshot of every ``HybridReturnModel.__init__`` kwarg.

    One source of truth for save/load + drift-guard tests. Field names MUST
    mirror the ctor param names.
    """

    feature_columns: tuple[str, ...]
    arma_p_max: int
    arma_q_max: int
    arma_information_criterion: InformationCriterion
    lstm_hidden_dim: int
    lstm_num_layers: int
    lstm_dropout: float
    lstm_lookback: int
    lstm_lr: float
    lstm_epochs: int
    lstm_loss_fn: LossFunction
    lstm_patience: int
    lstm_batch_size: int
    lstm_val_split_ratio: float
    lstm_device: Device | None
    lstm_amp: bool
    interval: Interval


@model_registry.register("hybrid_return")
class HybridReturnModel(IPredictor):
    """ARMA + LSTM hybrid: ARMA provides a one-step-ahead conditional mean
    forecast of log returns; an LSTM corrects the residual between the
    realized return and the ARMA forecast. Final output is
    ``arma_forecast + lstm_residual`` (no clipping — returns can be negative).
    """

    def __init__(
        self,
        *,
        feature_columns: list[str],
        arma_p_max: int = 5,
        arma_q_max: int = 5,
        arma_information_criterion: InformationCriterion = InformationCriterion.AIC,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
        lstm_lookback: int = 30,
        lstm_lr: float = 1e-3,
        lstm_epochs: int = 100,
        lstm_loss_fn: LossFunction = LossFunction.MSE,
        lstm_patience: int = 10,
        lstm_batch_size: int = 32,
        lstm_val_split_ratio: float = 0.2,
        lstm_device: Device | None = None,
        lstm_amp: bool = False,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if not feature_columns:
            raise ValueError(
                "HybridReturnModel requires a non-empty feature_columns list; "
                "fix by passing the explicit feature names the LSTM residual "
                "leaf should consume."
            )

        self._params = _HybridReturnConfig(
            feature_columns=tuple(feature_columns),
            arma_p_max=arma_p_max,
            arma_q_max=arma_q_max,
            arma_information_criterion=arma_information_criterion,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_dropout,
            lstm_lookback=lstm_lookback,
            lstm_lr=lstm_lr,
            lstm_epochs=lstm_epochs,
            lstm_loss_fn=lstm_loss_fn,
            lstm_patience=lstm_patience,
            lstm_batch_size=lstm_batch_size,
            lstm_val_split_ratio=lstm_val_split_ratio,
            lstm_device=lstm_device,
            lstm_amp=lstm_amp,
            interval=interval,
        )
        self._feature_columns: list[str] = list(feature_columns)

        self._arma = ARMAPredictor(
            p_max=arma_p_max,
            q_max=arma_q_max,
            information_criterion=arma_information_criterion,
            interval=interval,
        )
        self._lstm = LSTMPredictor(
            feature_columns=self._feature_columns,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            lookback=lstm_lookback,
            lr=lstm_lr,
            epochs=lstm_epochs,
            loss_fn=lstm_loss_fn,
            patience=lstm_patience,
            batch_size=lstm_batch_size,
            val_split_ratio=lstm_val_split_ratio,
            device=lstm_device,
            amp=lstm_amp,
            interval=interval,
        )

        self._scaler: StandardScaler | None = None

    @property
    def params(self) -> _HybridReturnConfig:
        """Frozen snapshot of every ctor kwarg — public so composites can
        sync their own passthrough-params bundle off a pretrained leaf
        without reaching into private state."""
        return self._params

    def fit(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        *,
        checkpoint_path: Path | None = None,
        **kwargs: object,
    ) -> None:
        """Fit ARMA on log returns, then fit LSTM on return residuals.

        Args:
            train_data: DataFrame with ``close`` column and all
                ``feature_columns``. DatetimeIndex required.
            target: Log returns series (may include leading NaN — it is
                dropped internally). Same convention as
                ``compute_log_returns(train_data["close"])``.
            checkpoint_path: Forwarded to ``LSTMPredictor.fit`` for best-state
                checkpointing of the residual-correction leaf.
            **kwargs: Forwarded to LSTM fit — supports Optuna ``trial``.
        """
        guard_scaler_fit_once(self._scaler, "HybridReturnModel")

        target_clean = target.dropna()
        with log_stage(logger, "HybridReturn [stage=arma]", n=len(target_clean)):
            self._arma.fit(train_data.loc[target_clean.index], target_clean)
            arma_train_pred = self._arma.predict(train_data, returns=target_clean)
            residuals = (target_clean - arma_train_pred).dropna()

        self._scaler = StandardScaler()
        feature_frame = train_data.loc[residuals.index, self._feature_columns]
        feature_frame, residuals = drop_feature_warmup(
            feature_frame, residuals, label="HybridReturn"
        )
        scaled_values = self._scaler.fit_transform(feature_frame)
        scaled_features = pd.DataFrame(
            scaled_values,
            index=residuals.index,
            columns=self._feature_columns,
        )

        with log_stage(
            logger,
            "HybridReturn [stage=lstm]",
            n=len(residuals),
            features=len(self._feature_columns),
        ):
            self._lstm.fit(scaled_features, residuals, checkpoint_path=checkpoint_path, **kwargs)

        self._set_fitted_with_metadata(
            TrainingMetadata.from_fit(
                train_data, self._params.interval, tuple(self._feature_columns)
            )
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Produce hybrid return forecasts aligned with ``data.index``.

        First ``lstm_lookback`` rows inherit NaN from the LSTM component.
        """
        # `_scaler is None` check narrows the type for mypy; once the
        # metadata slot is set the scaler is always set too (assigned
        # together inside `fit()`).
        self._assert_fitted_with_metadata()
        if self._scaler is None:
            raise RuntimeError(
                "HybridReturnModel.predict() invoked with no scaler wired; "
                "fix by re-running model.fit(train_data, target) (or load() "
                "from disk)."
            )

        log_returns = compute_log_returns(data["close"]).dropna()
        arma_pred = self._arma.predict(data, returns=log_returns)

        scaled_features = self._scale_to_frame(data[self._feature_columns])

        lstm_residual = self._lstm.predict(scaled_features)
        final_return = arma_pred + lstm_residual
        final_return.name = "hybrid_return"
        return final_return

    def predict_single(self, recent_window: pd.DataFrame) -> float:
        """Single hybrid return forecast from a recent window."""
        self._assert_fitted_with_metadata()
        return float(self.predict(recent_window).iloc[-1])

    def _scale_to_frame(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Transform ``feature_frame`` through the fitted scaler and rewrap as a
        DataFrame the LSTM can consume. Callers must ensure ``training_metadata``
        is set first (the ``cast`` is safe under that precondition).
        """
        scaler = cast(StandardScaler, self._scaler)
        scaled = scaler.transform(feature_frame)
        return pd.DataFrame(scaled, index=feature_frame.index, columns=self._feature_columns)

    def save(self, path: str | Path) -> None:
        """Persist HybridReturn to ``path`` as ``<path>/arma/`` +
        ``<path>/lstm/`` + ``<path>/scaler.json`` + config + metadata.

        Every ctor kwarg is persisted in the composite's own ``config.json``.
        """
        metadata = self._assert_fitted_with_metadata()
        # ``_scaler`` is set atomically with metadata in fit() — assert for mypy.
        assert self._scaler is not None

        scaler = self._scaler

        def write_weights(root: Path) -> None:
            self._arma.save(root / ARMA_SUBDIR)
            self._lstm.save(root / LSTM_SUBDIR)
            save_standard_scaler(scaler, root / SCALER_JSON)

        save_model_skeleton(
            path,
            config=self._ctor_kwargs_as_json(),
            training_metadata=metadata,
            write_weights=write_weights,
        )

    def _ctor_kwargs_as_json(self) -> dict[str, object]:
        """Snapshot of this composite's constructor kwargs as JSON-ready values.

        ``frozen_params_to_json`` handles the tuple→list + Enum→value
        conversions uniformly; ``lstm_device`` is dropped so the saved JSON
        doesn't pin a device that may not exist on the loading machine. The
        drift guard in ``tests/integration/test_strategy_save_load.py``
        verifies the output keys match ``__init__``'s parameter names.
        """
        return frozen_params_to_json(self._params, omit=("lstm_device",))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a fitted HybridReturnModel from ``path``.

        Construct the composite instance from its own ``config.json`` BEFORE
        loading sub-models — a corrupt composite config fast-fails with a
        named-field error, without wasting I/O on the ARMA/LSTM subdirs.
        """
        root = Path(path)
        config = json_io.read_dict(root / CONFIG_JSON)
        metadata = json_io.read_dict(root / METADATA_JSON)

        instance = cls(
            feature_columns=json_io.get_str_list(config, "feature_columns"),
            arma_p_max=json_io.get_int(config, "arma_p_max"),
            arma_q_max=json_io.get_int(config, "arma_q_max"),
            arma_information_criterion=InformationCriterion(
                json_io.get_str(config, "arma_information_criterion")
            ),
            lstm_hidden_dim=json_io.get_int(config, "lstm_hidden_dim"),
            lstm_num_layers=json_io.get_int(config, "lstm_num_layers"),
            lstm_dropout=json_io.get_float(config, "lstm_dropout"),
            lstm_lookback=json_io.get_int(config, "lstm_lookback"),
            lstm_lr=json_io.get_float(config, "lstm_lr"),
            lstm_epochs=json_io.get_int(config, "lstm_epochs"),
            lstm_loss_fn=LossFunction(json_io.get_str(config, "lstm_loss_fn")),
            lstm_patience=json_io.get_int(config, "lstm_patience"),
            lstm_batch_size=json_io.get_int(config, "lstm_batch_size"),
            lstm_val_split_ratio=json_io.get_float(config, "lstm_val_split_ratio"),
            lstm_amp=json_io.get_bool(config, "lstm_amp"),
            interval=Interval(json_io.get_str(config, "interval")),
        )
        instance._arma = ARMAPredictor.load(root / ARMA_SUBDIR)
        instance._lstm = LSTMPredictor.load(root / LSTM_SUBDIR)
        instance._scaler = load_standard_scaler(root / SCALER_JSON)
        instance._set_fitted_with_metadata(TrainingMetadata.from_dict(metadata))
        return instance

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        """Expose hybrid + owned ARMA + LSTM metadata for the deep leakage check."""
        return collect_metadata(
            ("hybrid_return", self._training_metadata),
            ("arma", self._arma.training_metadata),
            ("lstm", self._lstm.training_metadata),
        )

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space combining ARMA and LSTM hyperparameters."""
        return {
            "arma_p_max": trial.suggest_int("hybrid_ret_arma_p_max", 1, 5),
            "arma_q_max": trial.suggest_int("hybrid_ret_arma_q_max", 1, 5),
            "arma_information_criterion": InformationCriterion(
                trial.suggest_categorical(
                    "hybrid_ret_arma_ic", [e.value for e in InformationCriterion]
                )
            ),
            "lstm_hidden_dim": trial.suggest_int("hybrid_ret_lstm_hidden_dim", 32, 128),
            "lstm_num_layers": trial.suggest_int("hybrid_ret_lstm_num_layers", 1, 3),
            "lstm_dropout": trial.suggest_float("hybrid_ret_lstm_dropout", 0.0, 0.5),
            "lstm_lookback": trial.suggest_int("hybrid_ret_lstm_lookback", 10, 60),
            "lstm_lr": trial.suggest_float("hybrid_ret_lstm_lr", 1e-4, 1e-2, log=True),
            "lstm_loss_fn": LossFunction(
                trial.suggest_categorical(
                    "hybrid_ret_lstm_loss_fn", [e.value for e in LossFunction]
                )
            ),
        }
