"""Feature engineering pipeline with anti-leakage scaling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import quant_engine
from src.core.exceptions import guard_scaler_fit_once
from src.core.registry import feature_registry
from src.features.interface import IFeaturePipeline

logger = logging.getLogger(__name__)


def _compute_rsi(close: pd.Series[float], period: int = 14) -> pd.Series[float]:
    """Compute RSI via the C++ binding (Wilder's smoothing)."""
    values = quant_engine.RSI(period).compute(np.asarray(close, dtype=np.float64))
    return pd.Series(values, index=close.index)


def _compute_macd(
    close: pd.Series[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series[float], pd.Series[float], pd.Series[float]]:
    """Compute MACD line, signal line, and histogram via the C++ binding."""
    result = quant_engine.MACD(fast, slow, signal).compute_all(np.asarray(close, dtype=np.float64))
    macd_line = pd.Series(result.macd_line, index=close.index)
    signal_line = pd.Series(result.signal_line, index=close.index)
    histogram = pd.Series(result.histogram, index=close.index)
    return macd_line, signal_line, histogram


@feature_registry.register("standard")
class FeatureEngineeringPipeline(IFeaturePipeline):
    """Standard feature pipeline with anti-leakage scaling.

    Computes return-based, volatility, and technical features from
    OHLCV data.  StandardScaler is fit ONCE on training data — a
    second ``fit()`` raises ``LeakageError``.

    Leading NaN from warmup periods is preserved (never back-filled).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        vol_window: int = 20,
        ma_ratio_window: int = 20,
        short_return_period: int = 5,
        long_return_period: int = 21,
    ) -> None:
        if short_return_period < 2 or long_return_period <= short_return_period:
            raise ValueError(
                f"require 2 <= short_return_period < long_return_period, "
                f"got short={short_return_period}, long={long_return_period}"
            )

        self._rsi_period = rsi_period
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._vol_window = vol_window
        self._ma_ratio_window = ma_ratio_window
        self._short_return_period = short_return_period
        self._long_return_period = long_return_period

        self._scaler: StandardScaler | None = None

    @property
    def hard_nan_warmup_bars(self) -> int:
        """Count of leading NaN bars across all features.

        C++ MACD signal line emits NaN for the first ``slow + signal - 2`` bars.
        """
        macd_signal_warmup = self._macd_slow + self._macd_signal - 2
        return max(
            self._long_return_period,
            self._vol_window,
            self._ma_ratio_window,
            self._rsi_period,
            macd_signal_warmup,
        )

    def _compute_raw_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all raw features before scaling."""
        close: pd.Series[float] = data["close"]

        features = pd.DataFrame(index=data.index)

        features["return_1d"] = close.pct_change(1)
        features[f"return_{self._short_return_period}d"] = close.pct_change(
            self._short_return_period
        )
        features[f"return_{self._long_return_period}d"] = close.pct_change(self._long_return_period)
        features[f"vol_{self._vol_window}"] = features["return_1d"].rolling(self._vol_window).std()

        sma: pd.Series[float] = close.rolling(self._ma_ratio_window).mean()
        features["ma_ratio"] = close / sma

        features[f"rsi_{self._rsi_period}"] = _compute_rsi(close, self._rsi_period)

        macd, signal, hist = _compute_macd(
            close, self._macd_fast, self._macd_slow, self._macd_signal
        )
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_hist"] = hist

        return features

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit scaler on training features.

        Raises:
            LeakageError: If called more than once.
        """
        guard_scaler_fit_once(self._scaler, "FeatureEngineeringPipeline")

        features = self._compute_raw_features(train_data)
        self._fit_scaler(features)

    def _fit_scaler(self, features: pd.DataFrame) -> None:
        """Fit StandardScaler on non-NaN rows of pre-computed features.

        Precondition: callers must invoke ``guard_scaler_fit_once`` first.
        """
        guard_scaler_fit_once(self._scaler, "FeatureEngineeringPipeline")
        self._scaler = StandardScaler()
        valid_mask = features.notna().all(axis=1)
        if valid_mask.any():
            self._scaler.fit(features.loc[valid_mask])

        logger.info("Feature pipeline fitted on %d valid rows", int(valid_mask.sum()))

    def fit_transform(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform in one pass (avoids double feature computation)."""
        guard_scaler_fit_once(self._scaler, "FeatureEngineeringPipeline")

        features = self._compute_raw_features(train_data)
        self._fit_scaler(features)
        assert self._scaler is not None

        valid_mask = features.notna().all(axis=1)
        if valid_mask.any():
            features.loc[valid_mask] = self._scaler.transform(features.loc[valid_mask])

        return features

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler.

        Leading NaN from warmup periods is preserved.

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if self._scaler is None:
            raise RuntimeError("FeatureEngineeringPipeline.transform() called before fit()")

        features = self._compute_raw_features(data)

        valid_mask = features.notna().all(axis=1)
        if valid_mask.any():
            features.loc[valid_mask] = self._scaler.transform(features.loc[valid_mask])

        return features
