"""Hybrid volatility predictor: GARCH conditional variance + LSTM residual correction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.core import json_io
from src.core.exceptions import guard_scaler_fit_once
from src.core.persistence import (
    CONFIG_JSON,
    GARCH_SUBDIR,
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
from src.core.types import Device, Interval, LossFunction
from src.core.utils import compute_log_returns
from src.models.garch import GARCHPredictor
from src.models.interface import IPredictor
from src.models.lstm import LSTMPredictor

if TYPE_CHECKING:
    import optuna


@dataclass(frozen=True)
class _HybridVolConfig:
    """Frozen snapshot of every ``HybridVolatilityModel.__init__`` kwarg.

    One source of truth for save/load + drift-guard tests. ``feature_columns``
    is a tuple so ``frozen=True`` actually guarantees immutability (a list
    field would still be mutable). Field names MUST mirror the ctor param
    names — the drift test catches misalignment.
    """

    feature_columns: tuple[str, ...]
    garch_p_max: int
    garch_q_max: int
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
    min_vol: float
    interval: Interval


@model_registry.register("hybrid_volatility")
class HybridVolatilityModel(IPredictor):
    """GARCH + LSTM hybrid: GARCH provides the base conditional-variance
    forecast, an LSTM corrects the residual between realized volatility and
    the GARCH forecast. The final output is ``garch_vol + lstm_residual``
    clipped to ``min_vol``.
    """

    def __init__(
        self,
        *,
        feature_columns: list[str],
        garch_p_max: int = 5,
        garch_q_max: int = 5,
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
        min_vol: float = 1e-3,
        interval: Interval = Interval.DAILY,
    ) -> None:
        if not feature_columns:
            raise ValueError(
                "HybridVolatilityModel requires a non-empty feature_columns "
                "list; fix by passing the explicit feature names the LSTM "
                "residual leaf should consume."
            )

        self._params = _HybridVolConfig(
            feature_columns=tuple(feature_columns),
            garch_p_max=garch_p_max,
            garch_q_max=garch_q_max,
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
            min_vol=min_vol,
            interval=interval,
        )
        self._feature_columns: list[str] = list(feature_columns)

        self._garch = GARCHPredictor(
            p_max=garch_p_max,
            q_max=garch_q_max,
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
            interval=interval,
        )

        self._scaler: StandardScaler | None = None

    @property
    def params(self) -> _HybridVolConfig:
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
        """Fit GARCH on log returns, then fit LSTM on realized-vol residuals.

        Args:
            train_data: DataFrame with ``close`` column and all
                ``feature_columns``. DatetimeIndex required.
            target: Annualized realized volatility series aligned with
                ``train_data`` (caller computes via e.g. Garman-Klass).
            checkpoint_path: Forwarded to ``LSTMPredictor.fit`` for best-state
                checkpointing of the residual-correction leaf.
            **kwargs: Forwarded to LSTM fit — supports Optuna ``trial``.
        """
        guard_scaler_fit_once(self._scaler, "HybridVolatilityModel")

        log_returns = compute_log_returns(train_data["close"]).dropna()
        self._garch.fit(train_data.loc[log_returns.index], log_returns)

        garch_train_vol = self._garch.predict(train_data, returns=log_returns)
        residuals = (target - garch_train_vol).dropna()

        self._scaler = StandardScaler()
        feature_frame = train_data.loc[residuals.index, self._feature_columns]
        scaled_values = self._scaler.fit_transform(feature_frame)
        scaled_features = pd.DataFrame(
            scaled_values,
            index=residuals.index,
            columns=self._feature_columns,
        )

        self._lstm.fit(scaled_features, residuals, checkpoint_path=checkpoint_path, **kwargs)

        self._set_fitted_with_metadata(
            TrainingMetadata.from_fit(
                train_data, self._params.interval, tuple(self._feature_columns)
            )
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Produce hybrid volatility forecasts aligned with ``data.index``.

        First ``lstm_lookback`` rows inherit NaN from the LSTM component.
        Output is clipped to ``min_vol``.
        """
        # `_scaler is None` check narrows the type for mypy; once the
        # metadata slot is set the scaler is always set too (assigned
        # together inside `fit()`).
        self._assert_fitted_with_metadata()
        if self._scaler is None:
            raise RuntimeError(
                "HybridVolatilityModel.predict() invoked with no scaler "
                "wired; fix by re-running model.fit(train_data, target) "
                "(or load() from disk)."
            )

        log_returns = compute_log_returns(data["close"]).dropna()
        garch_vol = self._garch.predict(data, returns=log_returns)

        scaled_features = self._scale_to_frame(data[self._feature_columns])

        lstm_residual = self._lstm.predict(scaled_features)
        # Clip floor: vol is non-negative by definition; a large negative LSTM
        # residual can drive `garch_vol + residual` below zero on noisy data.
        final_vol = (garch_vol + lstm_residual).clip(lower=self._params.min_vol)
        final_vol.name = "hybrid_vol"
        return final_vol

    def _scale_to_frame(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Transform ``feature_frame`` through the fitted scaler and rewrap as a
        DataFrame the LSTM can consume. Callers must ensure ``training_metadata``
        is set first (the ``cast`` is safe under that precondition; ``_scaler``
        is assigned alongside the metadata commit inside ``fit()``).
        """
        scaler = cast(StandardScaler, self._scaler)
        scaled = scaler.transform(feature_frame)
        return pd.DataFrame(scaled, index=feature_frame.index, columns=self._feature_columns)

    def predict_single(self, recent_window: pd.DataFrame) -> float:
        """Single hybrid-vol forecast from a recent window."""
        self._assert_fitted_with_metadata()
        return float(self.predict(recent_window).iloc[-1])

    def save(self, path: str | Path) -> None:
        """Persist HybridVolatility to ``path`` as ``<path>/garch/`` +
        ``<path>/lstm/`` + ``<path>/scaler.json`` + config + metadata.

        Every ctor kwarg is persisted in the composite's own ``config.json``
        — we don't reach into leaf private state on load (see
        black-box-composition rule). The leaf directories exist solely to
        round-trip the fitted weights.
        """
        metadata = self._assert_fitted_with_metadata()
        # ``_scaler`` is set atomically with metadata in fit() — assert for mypy.
        assert self._scaler is not None

        scaler = self._scaler

        def write_weights(root: Path) -> None:
            self._garch.save(root / GARCH_SUBDIR)
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
        """Reconstruct a fitted HybridVolatilityModel from ``path``.

        Construct the composite instance from its own ``config.json`` BEFORE
        loading sub-models — a corrupt composite config fast-fails with a
        named-field error, without wasting I/O on the GARCH/LSTM subdirs.
        """
        root = Path(path)
        config = json_io.read_dict(root / CONFIG_JSON)
        metadata = json_io.read_dict(root / METADATA_JSON)

        instance = cls(
            feature_columns=json_io.get_str_list(config, "feature_columns"),
            garch_p_max=json_io.get_int(config, "garch_p_max"),
            garch_q_max=json_io.get_int(config, "garch_q_max"),
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
            min_vol=json_io.get_float(config, "min_vol"),
            interval=Interval(json_io.get_str(config, "interval")),
        )
        instance._garch = GARCHPredictor.load(root / GARCH_SUBDIR)
        instance._lstm = LSTMPredictor.load(root / LSTM_SUBDIR)
        instance._scaler = load_standard_scaler(root / SCALER_JSON)
        instance._set_fitted_with_metadata(TrainingMetadata.from_dict(metadata))
        return instance

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        """Expose hybrid + owned GARCH + LSTM metadata for the deep leakage check."""
        return collect_metadata(
            ("hybrid_volatility", self._training_metadata),
            ("garch", self._garch.training_metadata),
            ("lstm", self._lstm.training_metadata),
        )

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space combining GARCH and LSTM hyperparameters."""
        return {
            "garch_p_max": trial.suggest_int("hybrid_vol_garch_p_max", 1, 5),
            "garch_q_max": trial.suggest_int("hybrid_vol_garch_q_max", 1, 5),
            "lstm_hidden_dim": trial.suggest_int("hybrid_vol_lstm_hidden_dim", 32, 128),
            "lstm_num_layers": trial.suggest_int("hybrid_vol_lstm_num_layers", 1, 3),
            "lstm_dropout": trial.suggest_float("hybrid_vol_lstm_dropout", 0.0, 0.5),
            "lstm_lookback": trial.suggest_int("hybrid_vol_lstm_lookback", 10, 60),
            "lstm_lr": trial.suggest_float("hybrid_vol_lstm_lr", 1e-4, 1e-2, log=True),
            "lstm_loss_fn": LossFunction(
                trial.suggest_categorical(
                    "hybrid_vol_lstm_loss_fn", [e.value for e in LossFunction]
                )
            ),
        }
