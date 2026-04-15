"""ARMA return predictor using pmdarima for automatic order selection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pmdarima as pm

from src.core.registry import model_registry
from src.core.temporal import TrainingMetadata
from src.core.types import InformationCriterion, Interval
from src.core.utils import compute_log_returns
from src.models.interface import IPredictor

if TYPE_CHECKING:
    import optuna
    from pmdarima.arima import ARIMA

logger = logging.getLogger(__name__)


@model_registry.register("arma")
class ARMAPredictor(IPredictor):
    """ARMA predictor with automatic order selection via AIC.

    Uses pmdarima's auto_arima with d=0 (returns are stationary).
    One-step-ahead forecasting uses fixed parameters — no re-estimation.
    """

    def __init__(
        self,
        p_max: int = 5,
        q_max: int = 5,
        d: int = 0,
        information_criterion: InformationCriterion = InformationCriterion.AIC,
        interval: Interval = Interval.DAILY,
    ) -> None:
        self._p_max = p_max
        self._q_max = q_max
        self._d = d
        self._information_criterion = information_criterion
        self._interval = interval

        self._fitted = False
        self._model: ARIMA | None = None
        self._best_order: tuple[int, int, int] = (0, 0, 0)
        self._training_metadata: TrainingMetadata | None = None

    def _run_auto_arima(self, values: np.ndarray[tuple[int], np.dtype[np.float64]]) -> ARIMA:
        """Run auto_arima with configured parameters."""
        model: ARIMA = pm.auto_arima(
            values,
            start_p=0,
            start_q=0,
            max_p=self._p_max,
            max_q=self._q_max,
            d=self._d,
            seasonal=False,
            stepwise=True,
            information_criterion=self._information_criterion,
            suppress_warnings=True,
            error_action="ignore",
        )
        return model

    def tune(self, returns: pd.Series) -> tuple[int, int]:
        """Find best (p, q) order via auto_arima.

        Args:
            returns: Log returns series.

        Returns:
            Best (p, q) pair.
        """
        model = self._run_auto_arima(np.asarray(returns.values, dtype=np.float64))
        order = model.order
        return order[0], order[2]

    def fit(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        **kwargs: object,
    ) -> None:
        """Fit ARMA on training returns.

        Args:
            train_data: DataFrame with DatetimeIndex (used for metadata).
            target: Log returns series to fit on.
            **kwargs: Unused (reserved for Optuna Trial passthrough).
        """
        self._model = self._run_auto_arima(np.asarray(target.values, dtype=np.float64))
        self._best_order = self._model.order

        logger.info("ARMA fit: best order %s", self._best_order)

        self._fitted = True
        self._training_metadata = TrainingMetadata.from_fit(
            train_data, self._interval, ("returns",)
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """One-step-ahead forecasts using fixed ARMA parameters.

        Computes log returns internally from close prices via
        ``compute_log_returns()``. The caller's ``target`` passed to
        ``fit()`` must use the same log-return convention.

        Does NOT re-estimate parameters.

        Args:
            data: DataFrame with 'close' column and DatetimeIndex.

        Returns:
            Series of one-step-ahead return forecasts.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("ARMAPredictor.predict() called before fit()")

        returns = compute_log_returns(data["close"]).dropna()
        values = returns.values

        n = len(values)
        predictions = np.empty(n)

        fitted_vals = self._model.predict_in_sample()
        n_fitted = len(fitted_vals)

        if n <= n_fitted:
            predictions[:n] = fitted_vals[:n]
        else:
            predictions[:n_fitted] = fitted_vals
            n_oos = n - n_fitted
            oos_forecasts = self._model.predict(n_periods=n_oos)
            predictions[n_fitted:] = oos_forecasts

        # Offset by 1: the first row of `data` has no log return (it's NaN), so
        # `predictions` starts at data.index[1], not data.index[0].
        arr = np.full(len(data), np.nan)
        arr[1 : 1 + len(predictions)] = predictions
        return pd.Series(arr, index=data.index, name="arma_forecast").ffill()

    def predict_single(self, recent_window: pd.DataFrame) -> float:
        """Predict single one-step-ahead return forecast."""
        if not self._fitted or self._model is None:
            raise RuntimeError("ARMAPredictor.predict_single() called before fit()")

        forecast = self._model.predict(n_periods=1)
        return float(forecast[0])

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for ARMA hyperparameters."""
        return {
            "p_max": trial.suggest_int("arma_p_max", 1, 5),
            "q_max": trial.suggest_int("arma_q_max", 1, 5),
            "information_criterion": InformationCriterion(
                trial.suggest_categorical("arma_ic", [e.value for e in InformationCriterion])
            ),
        }
