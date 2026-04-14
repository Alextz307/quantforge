"""Feature engineering pipeline with anti-leakage scaling."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.core.exceptions import guard_scaler_fit_once
from src.core.registry import feature_registry
from src.features.interface import IFeaturePipeline

logger = logging.getLogger(__name__)


def _compute_rsi(close: pd.Series[float], period: int = 14) -> pd.Series[float]:
    """Compute RSI using rolling-window average of gains and losses.

    # TODO(Phase 4): replace with C++ binding
    """
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    avg_gain: pd.Series[float] = gains.rolling(window=period, min_periods=period).mean()
    avg_loss: pd.Series[float] = losses.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi: pd.Series[float] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _compute_macd(
    close: pd.Series[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series[float], pd.Series[float], pd.Series[float]]:
    """Compute MACD line, signal line, and histogram.

    # TODO(Phase 4): replace with C++ binding
    """
    ema_fast: pd.Series[float] = close.ewm(span=fast, adjust=False).mean()
    ema_slow: pd.Series[float] = close.ewm(span=slow, adjust=False).mean()
    macd_line: pd.Series[float] = ema_fast - ema_slow
    signal_line: pd.Series[float] = macd_line.ewm(span=signal, adjust=False).mean()
    histogram: pd.Series[float] = macd_line - signal_line
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
    ) -> None:
        self._rsi_period = rsi_period
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._vol_window = vol_window

        self._scaler: StandardScaler | None = None

    def _compute_raw_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all raw features before scaling."""
        close: pd.Series[float] = data["close"]

        features = pd.DataFrame(index=data.index)

        features["return_1d"] = close.pct_change(1)
        features["return_5d"] = close.pct_change(5)
        features["return_21d"] = close.pct_change(21)
        features[f"vol_{self._vol_window}"] = features["return_1d"].rolling(self._vol_window).std()

        sma: pd.Series[float] = close.rolling(self._vol_window).mean()
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
