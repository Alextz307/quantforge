"""ARMA return predictor using pmdarima for automatic order selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA as SMARIMA

from src.core import json_io
from src.core.persistence import (
    CONFIG_JSON,
    ENDOG_NPY,
    METADATA_JSON,
    WEIGHTS_JSON,
    save_model_skeleton,
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


class _StatsmodelsARMAAdapter:
    """Pmdarima-ARIMA-compatible adapter backed by statsmodels.

    Produced by every ``ARMAPredictor`` write path (``fit``, ``update``,
    ``load``) so the rest of the predictor can treat ``self._model`` as a
    single surface. ``predict_in_sample()`` returns the statsmodels fitted
    values as a plain numpy array; ``predict(n_periods)`` returns the numpy
    array of out-of-sample forecasts — both match the shapes
    ``ARMAPredictor.predict`` expected from the original pmdarima path.

    Public attributes (``order``, ``trend``, ``endog``, ``params``) are the
    single source of truth for ``save()`` and ``update()``. ``_results`` is
    the filtered SARIMAX state used for in-sample fitted values and
    out-of-sample forecasts.
    """

    def __init__(
        self,
        endog: np.ndarray[tuple[int], np.dtype[np.float64]],
        order: tuple[int, int, int],
        params: np.ndarray[tuple[int], np.dtype[np.float64]],
        trend: str,
    ) -> None:
        self.order = order
        self.trend = trend
        self.endog = endog
        self.params = params
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
        # ``self._model`` is always a ``_StatsmodelsARMAAdapter`` once fit. Its
        # public attributes (``order``, ``trend``, ``endog``, ``params``) are
        # the single source of truth for ``update()`` and ``save()``; pmdarima
        # is only used transiently inside ``fit()`` to pick the order.
        self._model: _StatsmodelsARMAAdapter | None = None
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
        model = self._run_auto_arima(np.asarray(returns, dtype=np.float64))
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
        endog = np.asarray(target, dtype=np.float64)
        pm_model = self._run_auto_arima(endog)
        order = pm_model.order
        arima_res = pm_model.arima_res_
        # pmdarima represents "no trend" as ``None``; statsmodels' ``ARIMA``
        # treats ``None`` as "auto" (adds a constant when d=0), so we normalize
        # to the explicit ``'n'`` here and carry that through save/load.
        trend_raw = arima_res.model.trend
        trend = trend_raw if isinstance(trend_raw, str) else "n"
        params = np.asarray(arima_res.params, dtype=np.float64)

        # Wrap into the adapter so every post-fit code path (predict, update,
        # save) sees the same ``_StatsmodelsARMAAdapter`` surface.
        self._model = _StatsmodelsARMAAdapter(endog, order, params, trend)

        logger.info("ARMA fit: best order %s", order)

        self._fitted = True
        self._training_metadata = TrainingMetadata.from_fit(
            train_data, self._interval, ("returns",)
        )

    def predict(
        self,
        data: pd.DataFrame,
        *,
        returns: pd.Series | None = None,
    ) -> pd.Series:
        """One-step-ahead forecasts using fixed ARMA parameters.

        Does NOT re-estimate parameters.

        Args:
            data: DataFrame with 'close' column and DatetimeIndex — used for
                output index alignment.
            returns: Optional pre-computed log returns (dropna'd). Its index
                must be a subset of ``data.index`` — typically
                ``data.index[1:]`` when derived from the same frame, though
                composites may pass a sub-range. Skips the internal
                ``compute_log_returns(data["close"])`` derivation when
                provided; defaults to deriving returns from ``data["close"]``.

        Returns:
            Series of one-step-ahead return forecasts.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("ARMAPredictor.predict() called before fit()")

        caller_returns = returns
        if caller_returns is None:
            returns_clean = compute_log_returns(data["close"]).dropna()
        else:
            returns_clean = caller_returns

        n = len(returns_clean)
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

        # Fast path for the canonical "returns indexed at data.index[1:]" case
        # — positional slice-assign is O(N); reindex is O(N log N) hash. Both
        # the ``returns=None`` default and composite callers that pass
        # ``compute_log_returns(data["close"]).dropna()`` land here.
        if caller_returns is None or returns_clean.index.equals(data.index[1:]):
            arr = np.full(len(data), np.nan)
            arr[1 : 1 + len(predictions)] = predictions
            return pd.Series(arr, index=data.index, name="arma_forecast").ffill()

        # Caller-provided returns index an arbitrary subset of data.index;
        # fall back to label alignment to stay correct.
        forecast = pd.Series(predictions, index=returns_clean.index, name="arma_forecast")
        return forecast.reindex(data.index).ffill()

    def predict_single(self, recent_window: pd.DataFrame) -> float:
        """Predict single one-step-ahead return forecast."""
        if not self._fitted or self._model is None:
            raise RuntimeError("ARMAPredictor.predict_single() called before fit()")

        forecast = self._model.predict(n_periods=1)
        return float(forecast[0])

    def update(
        self,
        new_data: pd.DataFrame,
        target: pd.Series,
        **kwargs: object,
    ) -> None:
        """Fixed-order refit on the training window extended by ``new_data``.

        Skips ``pmdarima.auto_arima`` — re-runs statsmodels MLE on the combined
        endog with the cached ``(p, d, q)`` and trend spec. The order stays
        frozen; coefficients move. The refitted model is stored as a
        ``_StatsmodelsARMAAdapter`` (same shape as post-``load()``) so
        ``predict()`` stays indifferent to fit-vs-update origin. See
        :meth:`IPredictor.update` for the shared contract.
        """
        if not self._fitted or self._model is None or self._training_metadata is None:
            raise RuntimeError("ARMAPredictor.update() called before fit()")

        new_metadata = self._training_metadata.extend_from(new_data)

        new_endog = np.asarray(target, dtype=np.float64)
        combined = np.concatenate([self._model.endog, new_endog])
        sm_result = SMARIMA(combined, order=self._model.order, trend=self._model.trend).fit()
        refitted_params = np.asarray(sm_result.params, dtype=np.float64)

        # Commit: pure assignments below cannot raise.
        self._model = _StatsmodelsARMAAdapter(
            combined, self._model.order, refitted_params, self._model.trend
        )
        self._training_metadata = new_metadata

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

        adapter = self._model

        def write_weights(root: Path) -> None:
            json_io.write(
                root / WEIGHTS_JSON,
                {
                    "order": list(adapter.order),
                    "params": adapter.params.tolist(),
                    "trend": adapter.trend,
                },
            )
            np.save(root / ENDOG_NPY, adapter.endog, allow_pickle=False)

        save_model_skeleton(
            path,
            config={
                "p_max": self._p_max,
                "q_max": self._q_max,
                "d": self._d,
                "information_criterion": self._information_criterion.value,
                "interval": self._interval.value,
            },
            training_metadata=self._training_metadata,
            write_weights=write_weights,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a fitted ARMAPredictor from ``path``.

        The loaded ``_model`` is a ``_StatsmodelsARMAAdapter`` — same shape
        as the post-fit and post-update paths.
        """
        root = Path(path)
        config = json_io.read_dict(root / CONFIG_JSON)
        weights = json_io.read_dict(root / WEIGHTS_JSON)
        metadata = json_io.read_dict(root / METADATA_JSON)

        instance = cls(
            p_max=json_io.get_int(config, "p_max"),
            q_max=json_io.get_int(config, "q_max"),
            d=json_io.get_int(config, "d"),
            information_criterion=InformationCriterion(
                json_io.get_str(config, "information_criterion")
            ),
            interval=Interval(json_io.get_str(config, "interval")),
        )
        order_ints = json_io.get_int_list(weights, "order")
        if len(order_ints) != 3:
            raise ValueError(f"ARMA order must be a 3-element list, got length {len(order_ints)}")
        order: tuple[int, int, int] = (order_ints[0], order_ints[1], order_ints[2])

        params = np.asarray(json_io.get_float_list(weights, "params"), dtype=np.float64)
        trend = json_io.get_str(weights, "trend")
        endog = np.load(root / ENDOG_NPY, allow_pickle=False).astype(np.float64, copy=False)

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
