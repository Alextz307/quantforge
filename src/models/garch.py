"""GARCH volatility predictor using the arch library."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from arch import arch_model

from src.core.registry import model_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.core.utils import compute_log_returns
from src.models.interface import IPredictor

if TYPE_CHECKING:
    import optuna
    from arch.univariate.base import ARCHModelResult

logger = logging.getLogger(__name__)

_SCALE_FACTOR = 100.0


@model_registry.register("garch")
class GARCHPredictor(IPredictor):
    """GARCH(p,q) volatility predictor with AIC-based order selection.

    Fits a GARCH model on returns scaled x100 (arch library convention),
    then produces conditional volatility forecasts using fixed parameters.
    """

    def __init__(
        self,
        p_max: int = 5,
        q_max: int = 5,
        interval: Interval = Interval.DAILY,
    ) -> None:
        self._p_max = p_max
        self._q_max = q_max
        self._interval = interval

        self._fitted = False
        self._best_p = 0
        self._best_q = 0
        self._omega = 0.0
        self._alpha: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([])
        self._beta: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([])
        self._train_mu = 0.0
        self._train_backcast = 0.0
        self._training_metadata: TrainingMetadata | None = None

    def tune(self, returns: pd.Series) -> tuple[int, int]:
        """Grid search over (p,q) in [1, p_max] x [1, q_max] using AIC.

        Args:
            returns: Raw (unscaled) log returns with DatetimeIndex.

        Returns:
            Best (p, q) pair.
        """
        _, best_p, best_q = self._grid_search(returns * _SCALE_FACTOR)
        return best_p, best_q

    def _grid_search(self, scaled: pd.Series) -> tuple[ARCHModelResult, int, int]:
        """Run AIC grid search and return (fitted_result, best_p, best_q)."""
        best_aic = math.inf
        best_p, best_q = 1, 1
        best_result: ARCHModelResult | None = None

        for p in range(1, self._p_max + 1):
            for q in range(1, self._q_max + 1):
                try:
                    model = arch_model(scaled, vol="GARCH", p=p, q=q, dist="skewt", mean="Zero")
                    result = model.fit(disp="off", show_warning=False)
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_p, best_q = p, q
                        best_result = result
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    continue

        if best_result is None:
            # All (p,q) combos failed — fall back to GARCH(1,1)
            fallback = arch_model(scaled, vol="GARCH", p=1, q=1, dist="skewt", mean="Zero")
            best_result = fallback.fit(disp="off", show_warning=False)
            best_p, best_q = 1, 1

        logger.info("GARCH tune: best (p=%d, q=%d) with AIC=%.2f", best_p, best_q, best_aic)
        return best_result, best_p, best_q

    def fit(
        self,
        train_data: pd.DataFrame,
        target: pd.Series,
        **kwargs: object,
    ) -> None:
        """Fit GARCH on training returns.

        Args:
            train_data: DataFrame with DatetimeIndex (used for metadata).
            target: Log returns series to fit on.
            **kwargs: Unused (reserved for Optuna Trial passthrough).
        """
        scaled = target * _SCALE_FACTOR

        result, self._best_p, self._best_q = self._grid_search(scaled)

        self._omega = float(result.params["omega"])
        self._alpha = np.array(
            [float(result.params[f"alpha[{i + 1}]"]) for i in range(self._best_p)]
        )
        self._beta = np.array([float(result.params[f"beta[{i + 1}]"]) for i in range(self._best_q)])
        self._train_mu = float(scaled.mean())
        cond_vol = result.conditional_volatility
        self._train_backcast = float(
            cond_vol.iloc[0] ** 2  # type: ignore[union-attr]
        )
        self._fitted = True

        self._training_metadata = TrainingMetadata.from_fit(
            train_data, self._interval, ("returns",)
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Produce annualized conditional volatility series.

        Computes log returns internally from close prices via
        ``compute_log_returns()``. The caller's ``target`` passed to
        ``fit()`` must use the same log-return convention.

        Args:
            data: DataFrame with 'close' column and DatetimeIndex.

        Returns:
            Series of annualized volatility forecasts.
        """
        if not self._fitted:
            raise RuntimeError("GARCHPredictor.predict() called before fit()")

        log_returns = compute_log_returns(data["close"])
        scaled = log_returns.dropna() * _SCALE_FACTOR

        cond_var = self._manual_garch_filter(np.asarray(scaled.values, dtype=np.float64))
        cond_vol_daily = np.sqrt(cond_var) / _SCALE_FACTOR

        ann_factor = math.sqrt(self._interval.annualization_factor())
        cond_vol_annual = cond_vol_daily * ann_factor

        # Align with original index: first row has NaN return, use ffill
        result = pd.Series(np.nan, index=data.index, name="garch_vol")
        result.iloc[1 : 1 + len(cond_vol_annual)] = cond_vol_annual.tolist()
        return result.ffill()

    def predict_single(self, recent_window: pd.DataFrame) -> float:
        """Predict a single annualized volatility value from recent data."""
        vol_series = self.predict(recent_window)
        return float(vol_series.iloc[-1])

    def generate_vol_series(self, returns: pd.Series) -> pd.Series:
        """Convenience: run the GARCH filter on a returns series.

        Args:
            returns: Raw (unscaled) log returns.

        Returns:
            Annualized conditional volatility series.
        """
        if not self._fitted:
            raise RuntimeError("GARCHPredictor.generate_vol_series() called before fit()")

        scaled = returns * _SCALE_FACTOR
        cond_var = self._manual_garch_filter(np.asarray(scaled.values, dtype=np.float64))
        cond_vol_daily = np.sqrt(cond_var) / _SCALE_FACTOR

        ann_factor = math.sqrt(self._interval.annualization_factor())
        result = pd.Series(
            cond_vol_daily * ann_factor,
            index=returns.index,
            name="garch_vol",
        )
        return result

    def _manual_garch_filter(
        self, scaled_returns: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Recursive GARCH(p,q) filter using fixed parameters.

        sigma^2[t] = omega + sum(alpha[i] * e^2[t-i]) + sum(beta[j] * sigma^2[t-j])

        For t < max(p, q), missing past values use backcast as substitute.

        Args:
            scaled_returns: Returns already multiplied by SCALE_FACTOR.

        Returns:
            Array of conditional variance values (scaled).
        """
        n = len(scaled_returns)
        sigma2 = np.empty(n)
        p = self._best_p
        q = self._best_q

        for t in range(n):
            var_t = self._omega

            for i in range(p):
                if t - i - 1 >= 0:
                    e2 = (scaled_returns[t - i - 1] - self._train_mu) ** 2
                else:
                    e2 = self._train_backcast
                var_t += self._alpha[i] * e2

            for j in range(q):
                if t - j - 1 >= 0:
                    var_t += self._beta[j] * sigma2[t - j - 1]
                else:
                    var_t += self._beta[j] * self._train_backcast

            sigma2[t] = max(var_t, 1e-12)

        return sigma2

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for GARCH hyperparameters."""
        return {
            "p_max": trial.suggest_int("garch_p_max", 1, 5),
            "q_max": trial.suggest_int("garch_q_max", 1, 5),
        }
