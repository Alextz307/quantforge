"""Hybrid return predictor: ARMA mean forecast + LSTM residual correction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.core.exceptions import guard_scaler_fit_once
from src.core.registry import model_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Device, InformationCriterion, Interval, LossFunction
from src.models.arma import ARMAPredictor
from src.models.interface import IPredictor
from src.models.lstm import LSTMPredictor

if TYPE_CHECKING:
    import optuna


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
        interval: Interval = Interval.DAILY,
    ) -> None:
        if not feature_columns:
            raise ValueError("HybridReturnModel requires a non-empty feature_columns list")

        self._feature_columns: list[str] = list(feature_columns)
        self._interval = interval

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
        """Fit ARMA on log returns, then fit LSTM on return residuals.

        Args:
            train_data: DataFrame with ``close`` column and all
                ``feature_columns``. DatetimeIndex required.
            target: Log returns series (may include leading NaN — it is
                dropped internally). Same convention as
                ``compute_log_returns(train_data["close"])``.
            **kwargs: Forwarded to LSTM fit — supports Optuna ``trial``.
        """
        guard_scaler_fit_once(self._scaler, "HybridReturnModel")

        target_clean = target.dropna()
        self._arma.fit(train_data.loc[target_clean.index], target_clean)

        arma_train_pred = self._arma.predict(train_data)
        residuals = (target_clean - arma_train_pred).dropna()

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
        """Produce hybrid return forecasts aligned with ``data.index``.

        First ``lstm_lookback`` rows inherit NaN from the LSTM component.
        """
        # `_scaler is None` check narrows the type for mypy; once `_fitted`
        # is True the scaler is always set (assigned together inside `fit()`).
        if not self._fitted or self._scaler is None:
            raise RuntimeError("HybridReturnModel.predict() called before fit()")

        # TODO(Phase 6): ARMA.predict() recomputes log returns from data["close"]
        # — already computed during fit(). Add a leaf `predict_from_returns()` fast
        # path to skip the recomputation in walk-forward / Optuna hot loops.
        arma_pred = self._arma.predict(data)

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
        final_return = arma_pred + lstm_residual
        final_return.name = "hybrid_return"
        return final_return

    def predict_single(self, recent_window: pd.DataFrame) -> float:
        """Single hybrid return forecast from a recent window."""
        if not self._fitted:
            raise RuntimeError("HybridReturnModel.predict_single() called before fit()")
        return float(self.predict(recent_window).iloc[-1])

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
