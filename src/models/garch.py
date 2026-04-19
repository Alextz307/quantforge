"""GARCH volatility predictor using the arch library."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import pandas as pd
from arch import arch_model

import quant_engine
from src.core import json_io
from src.core.persistence import (
    CONFIG_JSON,
    METADATA_JSON,
    TRAIN_RETURNS_NPY,
    WEIGHTS_JSON,
    save_model_skeleton,
)
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
        self._garch_params: quant_engine.GarchParams | None = None
        # Cache of the unscaled training returns (values only). ``update()``
        # concatenates this with new returns and re-fits with fixed (p,q),
        # skipping the AIC grid search.
        self._train_returns: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([])
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

        # Scalar / numpy views are retained alongside the cached ``GarchParams``
        # so "frozen-params" invariant tests can read individual fields.
        self._omega = float(result.params["omega"])
        self._alpha = np.array(
            [float(result.params[f"alpha[{i + 1}]"]) for i in range(self._best_p)]
        )
        self._beta = np.array([float(result.params[f"beta[{i + 1}]"]) for i in range(self._best_q)])
        self._train_mu = float(scaled.mean())
        cond_vol = cast(pd.Series, result.conditional_volatility)
        self._train_backcast = float(cond_vol.iloc[0] ** 2)
        self._garch_params = quant_engine.GarchParams(
            omega=self._omega,
            alpha=self._alpha.tolist(),
            beta=self._beta.tolist(),
            mu=self._train_mu,
            backcast=self._train_backcast,
        )
        self._train_returns = np.asarray(target.values, dtype=np.float64)
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
        arr = np.full(len(data), np.nan)
        arr[1 : 1 + len(cond_vol_annual)] = cond_vol_annual
        return pd.Series(arr, index=data.index, name="garch_vol").ffill()

    def predict_single(self, recent_window: pd.DataFrame) -> float:
        """Predict a single annualized volatility value from recent data."""
        vol_series = self.predict(recent_window)
        return float(vol_series.iloc[-1])

    def update(
        self,
        new_data: pd.DataFrame,
        target: pd.Series,
        **kwargs: object,
    ) -> None:
        """Warm-start refit on the training window extended by ``new_data``.

        Skips the AIC grid search — re-fits only ``(best_p, best_q)`` on the
        combined training + new-bar returns, typically ~10-25× faster than a
        full ``fit()`` when ``p_max == q_max == 5``. See :meth:`IPredictor.update`
        for the shared transactional-validation + metadata contract.
        """
        if not self._fitted or self._training_metadata is None:
            raise RuntimeError("GARCHPredictor.update() called before fit()")

        new_metadata = self._training_metadata.extend_from(new_data)

        new_returns = np.asarray(target.values, dtype=np.float64)
        combined = np.concatenate([self._train_returns, new_returns])
        scaled = combined * _SCALE_FACTOR

        model = arch_model(
            scaled,
            vol="GARCH",
            p=self._best_p,
            q=self._best_q,
            dist="skewt",
            mean="Zero",
        )
        result = model.fit(disp="off", show_warning=False)

        # Commit: pure assignments below cannot raise.
        self._omega = float(result.params["omega"])
        self._alpha = np.array(
            [float(result.params[f"alpha[{i + 1}]"]) for i in range(self._best_p)]
        )
        self._beta = np.array([float(result.params[f"beta[{i + 1}]"]) for i in range(self._best_q)])
        self._train_mu = float(scaled.mean())
        # ``result.conditional_volatility`` is a pd.Series when the input was a
        # Series and a plain ndarray when the input was an array. ``update()``
        # passes a concatenated ndarray, so index via ``[0]`` instead of the
        # Series-only ``.iloc[0]`` accessor used by ``fit()``.
        self._train_backcast = float(np.asarray(result.conditional_volatility)[0] ** 2)
        self._garch_params = quant_engine.GarchParams(
            omega=self._omega,
            alpha=self._alpha.tolist(),
            beta=self._beta.tolist(),
            mu=self._train_mu,
            backcast=self._train_backcast,
        )
        self._train_returns = combined
        self._training_metadata = new_metadata

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
        """Run the GARCH(p,q) recursion via the C++ filter on the cached params."""
        # Non-None by contract: callers guard on ``self._fitted``.
        params = cast(quant_engine.GarchParams, self._garch_params)
        return quant_engine.garch_filter(scaled_returns, params)

    def save(self, path: str | Path) -> None:
        """Persist fitted GARCH params to ``path`` as a directory.

        ``best_p`` and ``best_q`` are NOT persisted — they are always equal to
        ``len(alpha)`` and ``len(beta)`` respectively, so storing them would
        introduce a silent consistency failure point.
        """
        if not self._fitted:
            raise RuntimeError("GARCHPredictor.save() called before fit()")
        if self._training_metadata is None:
            raise RuntimeError("GARCHPredictor.save() missing training metadata")

        def write_weights(root: Path) -> None:
            json_io.write(
                root / WEIGHTS_JSON,
                {
                    "omega": self._omega,
                    "alpha": self._alpha.tolist(),
                    "beta": self._beta.tolist(),
                    "mu": self._train_mu,
                    "backcast": self._train_backcast,
                },
            )
            # Training returns are cached for warm-start ``update()``. Stored as
            # float64 ``.npy`` rather than JSON to keep long training windows
            # lean on disk (2000 bars ≈ 16KB here vs. ~40KB as JSON digits).
            np.save(root / TRAIN_RETURNS_NPY, self._train_returns, allow_pickle=False)

        save_model_skeleton(
            path,
            config={
                "p_max": self._p_max,
                "q_max": self._q_max,
                "interval": self._interval.value,
            },
            training_metadata=self._training_metadata,
            write_weights=write_weights,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reconstruct a fitted GARCHPredictor from ``path``."""
        root = Path(path)
        config = json_io.read_dict(root / CONFIG_JSON)
        weights = json_io.read_dict(root / WEIGHTS_JSON)
        metadata = json_io.read_dict(root / METADATA_JSON)

        instance = cls(
            p_max=json_io.get_int(config, "p_max"),
            q_max=json_io.get_int(config, "q_max"),
            interval=Interval(json_io.get_str(config, "interval")),
        )
        alpha = json_io.get_float_list(weights, "alpha")
        beta = json_io.get_float_list(weights, "beta")
        instance._omega = json_io.get_float(weights, "omega")
        instance._alpha = np.asarray(alpha, dtype=np.float64)
        instance._beta = np.asarray(beta, dtype=np.float64)
        instance._train_mu = json_io.get_float(weights, "mu")
        instance._train_backcast = json_io.get_float(weights, "backcast")
        instance._best_p = len(alpha)
        instance._best_q = len(beta)
        # ``GarchParams`` takes independent copies so a future caller that
        # mutates ``_alpha``/``_beta`` ndarrays can't silently divergence the
        # cached pybind11 struct. The list() copy is O(p+q), typically ≤10
        # elements.
        instance._garch_params = quant_engine.GarchParams(
            omega=instance._omega,
            alpha=list(alpha),
            beta=list(beta),
            mu=instance._train_mu,
            backcast=instance._train_backcast,
        )
        instance._train_returns = np.load(root / TRAIN_RETURNS_NPY, allow_pickle=False).astype(
            np.float64, copy=False
        )
        instance._training_metadata = TrainingMetadata.from_dict(metadata)
        instance._fitted = True
        return instance

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, object]:
        """Optuna search space for GARCH hyperparameters."""
        return {
            "p_max": trial.suggest_int("garch_p_max", 1, 5),
            "q_max": trial.suggest_int("garch_q_max", 1, 5),
        }
