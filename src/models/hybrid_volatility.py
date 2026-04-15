"""Hybrid volatility predictor: GARCH conditional variance + LSTM residual correction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.core.exceptions import guard_scaler_fit_once
from src.core.registry import model_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Device, Interval, LossFunction
from src.core.utils import compute_log_returns
from src.models.garch import GARCHPredictor
from src.models.interface import IPredictor
from src.models.lstm import LSTMPredictor

if TYPE_CHECKING:
    import optuna


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
            raise ValueError("HybridVolatilityModel requires a non-empty feature_columns list")

        self._feature_columns: list[str] = list(feature_columns)
        self._min_vol = min_vol
        self._interval = interval

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
        self._fitted = False
        self._training_metadata: TrainingMetadata | None = None

    def fit(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        **kwargs: object,
    ) -> None:
        """Fit GARCH on log returns, then fit LSTM on realized-vol residuals.

        Args:
            train_data: DataFrame with ``close`` column and all
                ``feature_columns``. DatetimeIndex required.
            target: Annualized realized volatility series aligned with
                ``train_data`` (caller computes via e.g. Garman-Klass).
            **kwargs: Forwarded to LSTM fit — supports Optuna ``trial``.
        """
        guard_scaler_fit_once(self._scaler, "HybridVolatilityModel")

        log_returns = compute_log_returns(train_data["close"]).dropna()
        self._garch.fit(train_data.loc[log_returns.index], log_returns)

        garch_train_vol = self._garch.predict(train_data)
        residuals = (target - garch_train_vol).dropna()

        self._scaler = StandardScaler()
        feature_frame = train_data.loc[residuals.index, self._feature_columns]
        scaled_values = self._scaler.fit_transform(feature_frame)
        scaled_features = pd.DataFrame(
            scaled_values,
            index=residuals.index,
            columns=self._feature_columns,
        )

        self._lstm.fit(scaled_features, residuals, **kwargs)

        self._fitted = True
        self._training_metadata = TrainingMetadata.from_fit(
            train_data, self._interval, tuple(self._feature_columns)
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Produce hybrid volatility forecasts aligned with ``data.index``.

        First ``lstm_lookback`` rows inherit NaN from the LSTM component.
        Output is clipped to ``min_vol``.
        """
        # `_scaler is None` check narrows the type for mypy; once `_fitted`
        # is True the scaler is always set (assigned together inside `fit()`).
        if not self._fitted or self._scaler is None:
            raise RuntimeError("HybridVolatilityModel.predict() called before fit()")

        # TODO(Phase 6): GARCH.predict() recomputes log returns from data["close"]
        # — already computed during fit(). Add a leaf `predict_from_returns()` fast
        # path to skip the recomputation in walk-forward / Optuna hot loops.
        garch_vol = self._garch.predict(data)

        # TODO(Phase 6): wrapping the scaled ndarray as a DataFrame just so LSTM
        # can call `.values` again is wasted allocation per fold. Add
        # `LSTMPredictor.predict_array(scaled, index)` to accept ndarray directly.
        scaled_values = self._scaler.transform(data[self._feature_columns])
        scaled_features = pd.DataFrame(
            scaled_values,
            index=data.index,
            columns=self._feature_columns,
        )

        lstm_residual = self._lstm.predict(scaled_features)
        # Clip floor: vol is non-negative by definition; a large negative LSTM
        # residual can drive `garch_vol + residual` below zero on noisy data.
        final_vol = (garch_vol + lstm_residual).clip(lower=self._min_vol)
        final_vol.name = "hybrid_vol"
        return final_vol

    def predict_single(self, recent_window: pd.DataFrame) -> float:
        """Single hybrid-vol forecast from a recent window."""
        if not self._fitted:
            raise RuntimeError("HybridVolatilityModel.predict_single() called before fit()")
        return float(self.predict(recent_window).iloc[-1])

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
