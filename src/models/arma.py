"""ARMA return predictor using pmdarima for automatic order selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA as SMARIMA

from src.core.persistence import (
    CONFIG_JSON,
    ENDOG_NPY,
    METADATA_JSON,
    WEIGHTS_JSON,
    ensure_model_dir,
    json_get_float_list,
    json_get_int,
    json_get_int_list,
    json_get_str,
    read_json_dict,
    write_json,
)
from src.core.registry import model_registry
from src.core.temporal import TrainingMetadata
from src.core.types import InformationCriterion, Interval
from src.core.utils import compute_log_returns
from src.models.interface import IPredictor

if TYPE_CHECKING:
    import optuna
    from pmdarima.arima import ARIMA

logger = logging.getLogger(__name__)


class _ARMAModel(Protocol):
    """Subset of the pmdarima ``ARIMA`` surface that ``ARMAPredictor`` uses.

    ``_StatsmodelsARMAAdapter`` satisfies this protocol post-load so the rest of
    the predictor doesn't care whether the underlying model is pmdarima (fresh
    fit) or statsmodels-backed (reconstructed from JSON).
    """

    order: tuple[int, int, int]

    def predict_in_sample(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...
    def predict(self, n_periods: int) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...


class _StatsmodelsARMAAdapter:
    """Minimal pmdarima-ARIMA-compatible adapter backed by statsmodels.

    Constructed by ``ARMAPredictor.load()``. ``predict_in_sample`` returns the
    statsmodels fitted values as a plain numpy array; ``predict(n_periods)``
    returns the numpy array of out-of-sample forecasts. Matches the shapes
    ``ARMAPredictor.predict`` already expects from pmdarima.

    ``_results`` holds the filtered SARIMAX state; ``order`` is public to match
    the pmdarima ``ARIMA.order`` surface. Every other input is function-local
    — the statsmodels results object already retains the endog and params it
    needs, so stashing them on ``self`` would just duplicate memory.
    """

    def __init__(
        self,
        endog: np.ndarray[tuple[int], np.dtype[np.float64]],
        order: tuple[int, int, int],
        params: np.ndarray[tuple[int], np.dtype[np.float64]],
        trend: str,
    ) -> None:
        self.order = order
        # ``statsmodels.tsa.arima.model.ARIMA`` auto-adds a constant when ``d=0``
        # and ``trend=None``. Callers persist ``'n'`` for "no trend" and ``'c'``
        # for "with intercept" so the filter params and the model shape match.
        model = SMARIMA(endog, order=order, trend=trend)
        self._results = model.filter(params)

    def predict_in_sample(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        fitted = self._results.fittedvalues
        return np.asarray(fitted, dtype=np.float64)

    def predict(self, n_periods: int) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        forecast = self._results.forecast(steps=n_periods)
        return np.asarray(forecast, dtype=np.float64)


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
        self._model: _ARMAModel | None = None
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

    def save(self, path: str | Path) -> None:
        """Persist fitted ARMA params + training endog to ``path``.

        The training endog is persisted because statsmodels needs it to
        reconstruct the filter state on load — without it, ``predict_in_sample``
        cannot recover the fitted values. It's written as a numpy ``.npy`` file
        (binary, pickle-free for float arrays) rather than JSON to keep the
        on-disk size manageable on large training windows.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("ARMAPredictor.save() called before fit()")
        if self._training_metadata is None:
            raise RuntimeError("ARMAPredictor.save() missing training metadata")

        # pmdarima's ARIMA stores the statsmodels result in ``arima_res_``. That
        # holds the fitted params + the training endog + the trend spec.
        pm_model = cast("ARIMA", self._model)
        arima_res = pm_model.arima_res_
        trend_raw = arima_res.model.trend
        # pmdarima represents "no trend" as ``None``; statsmodels' ``ARIMA``
        # treats ``None`` as "auto" (adds a constant when d=0), so on load we
        # persist ``'n'`` to mean "no trend" explicitly.
        trend = trend_raw if isinstance(trend_raw, str) else "n"

        root = ensure_model_dir(path)
        write_json(
            root / CONFIG_JSON,
            {
                "p_max": self._p_max,
                "q_max": self._q_max,
                "d": self._d,
                "information_criterion": self._information_criterion.value,
                "interval": self._interval.value,
            },
        )
        write_json(
            root / WEIGHTS_JSON,
            {
                "order": list(self._best_order),
                "params": np.asarray(arima_res.params, dtype=np.float64).tolist(),
                "trend": trend,
            },
        )
        endog = np.asarray(arima_res.model.endog, dtype=np.float64)
        np.save(root / ENDOG_NPY, endog, allow_pickle=False)
        write_json(root / METADATA_JSON, self._training_metadata.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a fitted ARMAPredictor from ``path``.

        The loaded ``_model`` is a statsmodels-backed adapter rather than a
        pmdarima ARIMA — pmdarima has no pickle-free round-trip. The adapter
        exposes the same ``.order`` / ``.predict_in_sample`` / ``.predict``
        surface, so ``ARMAPredictor.predict()`` doesn't care.
        """
        root = Path(path)
        config = read_json_dict(root / CONFIG_JSON)
        weights = read_json_dict(root / WEIGHTS_JSON)
        metadata = read_json_dict(root / METADATA_JSON)

        instance = cls(
            p_max=json_get_int(config, "p_max"),
            q_max=json_get_int(config, "q_max"),
            d=json_get_int(config, "d"),
            information_criterion=InformationCriterion(
                json_get_str(config, "information_criterion")
            ),
            interval=Interval(json_get_str(config, "interval")),
        )
        order_ints = json_get_int_list(weights, "order")
        if len(order_ints) != 3:
            raise ValueError(f"ARMA order must be a 3-element list, got length {len(order_ints)}")
        order: tuple[int, int, int] = (order_ints[0], order_ints[1], order_ints[2])

        params = np.asarray(json_get_float_list(weights, "params"), dtype=np.float64)
        trend = json_get_str(weights, "trend")
        endog = np.load(root / ENDOG_NPY, allow_pickle=False).astype(np.float64, copy=False)

        instance._best_order = order
        instance._model = _StatsmodelsARMAAdapter(endog, order, params, trend)
        instance._training_metadata = TrainingMetadata.from_dict(metadata)
        instance._fitted = True
        return instance

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
