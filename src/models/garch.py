"""GARCH volatility predictor using the arch library."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import pandas as pd
from arch import arch_model

import quant_engine
from src.core import json_io
from src.core.logging import get_logger
from src.core.persistence import (
    CONFIG_JSON,
    METADATA_JSON,
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

logger = get_logger(__name__)

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
        *,
        checkpoint_path: Path | None = None,  # noqa: ARG002
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
        """Produce annualized conditional volatility series.

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
            Series of annualized volatility forecasts.
        """
        if not self._fitted:
            raise RuntimeError("GARCHPredictor.predict() called before fit()")

        caller_returns = returns
        if caller_returns is None:
            returns_clean = compute_log_returns(data["close"]).dropna()
        else:
            returns_clean = caller_returns

        scaled = np.asarray(returns_clean, dtype=np.float64) * _SCALE_FACTOR
        cond_var = self._manual_garch_filter(scaled)
        cond_vol_daily = np.sqrt(cond_var) / _SCALE_FACTOR
        cond_vol_annual = cond_vol_daily * math.sqrt(self._interval.annualization_factor())

        # Fast path for the canonical "returns indexed at data.index[1:]" case:
        # one NaN-filled leading row, every remaining bar has one entry.
        # Positional slice-assign is O(N); reindex is O(N log N) hash. Both
        # the ``returns=None`` default and composite callers that pass
        # ``compute_log_returns(data["close"]).dropna()`` land here.
        if caller_returns is None or returns_clean.index.equals(data.index[1:]):
            arr = np.full(len(data), np.nan)
            arr[1 : 1 + len(cond_vol_annual)] = cond_vol_annual
            return pd.Series(arr, index=data.index, name="garch_vol").ffill()

        # Caller-provided returns index an arbitrary subset of data.index;
        # fall back to label alignment to stay correct.
        vol_series = pd.Series(cond_vol_annual, index=returns_clean.index, name="garch_vol")
        return vol_series.reindex(data.index).ffill()

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
        cond_var = self._manual_garch_filter(np.asarray(scaled, dtype=np.float64))
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
        metadata = self._assert_fitted_with_metadata(caller="save")

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

        save_model_skeleton(
            path,
            config={
                "p_max": self._p_max,
                "q_max": self._q_max,
                "interval": self._interval.value,
            },
            training_metadata=metadata,
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
