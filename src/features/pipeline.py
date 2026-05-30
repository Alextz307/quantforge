"""
Feature engineering pipeline with anti-leakage scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import quant_engine
from src.core.constants import (
    ADX_PERIOD,
    BOLLINGER_NUM_STD,
    BOLLINGER_PERIOD,
    GARMAN_KLASS_WINDOW,
    OBV_ZSCORE_WINDOW,
    OHLCV_COLUMNS,
    ROC_QUARTER_PERIOD,
    VOLUME_ZSCORE_WINDOW,
)
from src.core.exceptions import guard_scaler_fit_once
from src.core.logging import get_logger
from src.core.registry import feature_registry
from src.features.interface import IFeaturePipeline

logger = get_logger(__name__)


def _compute_rsi(close: pd.Series[float], period: int = 14) -> pd.Series[float]:
    """
    Compute RSI via the C++ binding (Wilder's smoothing).
    """

    values = quant_engine.RSI(period).compute(np.asarray(close, dtype=np.float64))
    return pd.Series(values, index=close.index)


def _compute_macd(
    close: pd.Series[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series[float], pd.Series[float], pd.Series[float]]:
    """
    Compute MACD line, signal line, and histogram via the C++ binding.
    """

    result = quant_engine.MACD(fast, slow, signal).compute_all(np.asarray(close, dtype=np.float64))
    macd_line = pd.Series(result.macd_line, index=close.index)
    signal_line = pd.Series(result.signal_line, index=close.index)
    histogram = pd.Series(result.histogram, index=close.index)
    return macd_line, signal_line, histogram


def _compute_garman_klass(
    open_: pd.Series[float],
    high: pd.Series[float],
    low: pd.Series[float],
    close: pd.Series[float],
    window: int,
) -> pd.Series[float]:
    """
    Annualized Garman-Klass volatility via the C++ estimator.

    Trailing rolling window; the first ``window - 1`` bars are NaN.
    """

    values = quant_engine.GarmanKlass(window).compute(
        np.asarray(open_, dtype=np.float64),
        np.asarray(high, dtype=np.float64),
        np.asarray(low, dtype=np.float64),
        np.asarray(close, dtype=np.float64),
    )
    return pd.Series(values, index=close.index)


def _compute_bollinger_pctb(
    close: pd.Series[float],
    period: int,
    num_std: float,
) -> pd.Series[float]:
    """
    Bollinger %B = (close - lower) / (upper - lower) via the C++ bands.

    Bands carry a ``period - 1`` NaN warmup; %B inherits it. Zero-width
    bands (degenerate flat windows) yield NaN rather than a divide blow-up.
    """

    result = quant_engine.BollingerBands(period, num_std).compute_all(
        np.asarray(close, dtype=np.float64)
    )
    upper = pd.Series(result.upper, index=close.index)
    lower = pd.Series(result.lower, index=close.index)
    width = (upper - lower).replace(0.0, np.nan)
    return (close - lower) / width


def _compute_adx(
    high: pd.Series[float],
    low: pd.Series[float],
    close: pd.Series[float],
    period: int,
) -> pd.Series[float]:
    """
    Wilder's ADX (trend strength) via forward-only EWM smoothing.

    Directional movement and true range use only bars <= t (``diff`` /
    ``shift`` are backward; ``ewm(adjust=False)`` is a forward recursion).
    The first ``2 * period - 1`` bars are NaN-masked: ADX needs one
    Wilder pass for the directional indices and a second for the average,
    so earlier values are not warmed up.
    """

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / period
    atr = true_range.ewm(alpha=alpha, adjust=False).mean().replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    adx.iloc[: 2 * period - 1] = np.nan
    return adx


@feature_registry.register("standard")
class FeatureEngineeringPipeline(IFeaturePipeline):
    """
    Standard feature pipeline with anti-leakage scaling.

    Computes return-based, volatility, and technical features from
    OHLCV data.  StandardScaler is fit ONCE on training data - a
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
        roc_period: int = ROC_QUARTER_PERIOD,
        gk_window: int = GARMAN_KLASS_WINDOW,
        bb_period: int = BOLLINGER_PERIOD,
        bb_num_std: float = BOLLINGER_NUM_STD,
        adx_period: int = ADX_PERIOD,
        volume_zscore_window: int = VOLUME_ZSCORE_WINDOW,
        obv_zscore_window: int = OBV_ZSCORE_WINDOW,
        keep_ohlc: bool = False,
    ) -> None:
        if short_return_period < 2 or long_return_period <= short_return_period:
            raise ValueError(
                f"require 2 <= short_return_period < long_return_period, "
                f"got short={short_return_period}, long={long_return_period}; "
                f"fix by raising long_return_period above short_return_period "
                f"(typical: short=5, long=21)."
            )

        self._rsi_period = rsi_period
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._vol_window = vol_window
        self._ma_ratio_window = ma_ratio_window
        self._short_return_period = short_return_period
        self._long_return_period = long_return_period
        self._roc_period = roc_period
        self._gk_window = gk_window
        self._bb_period = bb_period
        self._bb_num_std = bb_num_std
        self._adx_period = adx_period
        self._volume_zscore_window = volume_zscore_window
        self._obv_zscore_window = obv_zscore_window
        # When True, transform() emits raw OHLCV (un-scaled) alongside
        # the scaled engineered features. Strategies like ReturnForecast
        # and VolatilityTargeting depend on both being in the same frame.
        self._keep_ohlc = keep_ohlc

        self._scaler: StandardScaler | None = None

    @property
    def scaler(self) -> StandardScaler | None:
        """
        The fitted ``StandardScaler``, or ``None`` before ``fit()``.

        Exposed so callers that persist the pipeline's state (e.g.
        ``MomentumGatekeeperStrategy.save``) can round-trip the scaler
        through the public API instead of reaching into ``_scaler``.
        """

        return self._scaler

    @scaler.setter
    def scaler(self, value: StandardScaler) -> None:
        """
        Replace the fitted scaler with a loaded one.

        Used by ``load()`` paths that reconstruct the pipeline from a
        persisted scaler rather than re-fitting. The loaded scaler is
        assumed fitted - sklearn will raise on the first ``transform()``
        call otherwise.
        """

        self._scaler = value

    @property
    def hard_nan_warmup_bars(self) -> int:
        """
        Count of leading NaN bars across all features.

        The longest warmup wins: ``roc_period`` (default 63) usually
        dominates, ahead of the MACD signal line (``slow + signal - 2``)
        and ADX (``2 * adx_period - 1``).
        """

        macd_signal_warmup = self._macd_slow + self._macd_signal - 2
        adx_warmup = 2 * self._adx_period - 1
        return max(
            self._long_return_period,
            self._vol_window,
            self._ma_ratio_window,
            self._rsi_period,
            macd_signal_warmup,
            self._roc_period,
            self._gk_window,
            self._bb_period,
            adx_warmup,
            self._volume_zscore_window,
            self._obv_zscore_window,
        )

    def _compute_raw_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all raw features before scaling.

        Requires full OHLCV: the range / gap / volume / Garman-Klass /
        ADX features read open / high / low / volume, not just close.
        """

        missing = [c for c in OHLCV_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"FeatureEngineeringPipeline requires OHLCV columns; missing {missing} "
                f"(have {list(data.columns)}). The range / gap / volume / Garman-Klass "
                f"features need full OHLCV - pass a frame with "
                f"open / high / low / close / volume."
            )

        open_: pd.Series[float] = data["open"]
        high: pd.Series[float] = data["high"]
        low: pd.Series[float] = data["low"]
        close: pd.Series[float] = data["close"]
        volume: pd.Series[float] = data["volume"]

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

        features[f"roc_{self._roc_period}"] = close.pct_change(self._roc_period)
        features["garman_klass"] = _compute_garman_klass(open_, high, low, close, self._gk_window)
        features["intraday_range"] = (high - low) / close
        features["overnight_gap"] = open_ / close.shift(1) - 1.0
        features["bb_pctb"] = _compute_bollinger_pctb(close, self._bb_period, self._bb_num_std)
        features[f"adx_{self._adx_period}"] = _compute_adx(high, low, close, self._adx_period)

        vol_mean = volume.rolling(self._volume_zscore_window).mean()
        # Flat-volume window -> std 0; guard like the bands/ATR so the z-score
        # is NaN, not a divide-by-zero.
        vol_std = volume.rolling(self._volume_zscore_window).std().replace(0.0, np.nan)
        features["volume_zscore"] = (volume - vol_mean) / vol_std

        direction = pd.Series(
            np.sign(np.asarray(close.diff(), dtype=np.float64)), index=close.index
        )
        obv = (direction * volume).cumsum()
        obv_mean = obv.rolling(self._obv_zscore_window).mean()
        obv_std = obv.rolling(self._obv_zscore_window).std().replace(0.0, np.nan)
        features["obv_z"] = (obv - obv_mean) / obv_std

        return features

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Fit scaler on training features.

        Raises:
            LeakageError: If called more than once.
        """

        guard_scaler_fit_once(self._scaler, "FeatureEngineeringPipeline")

        features = self._compute_raw_features(train_data)
        self._fit_scaler(features)

    def _fit_scaler(self, features: pd.DataFrame) -> None:
        """
        Fit StandardScaler on non-NaN rows of pre-computed features.

        Precondition: callers must invoke ``guard_scaler_fit_once`` first.

        Raises:
            ValueError: If no row survives the feature warmup (every row has
                at least one NaN feature). Fails fast at fit time rather than
                leaving an unfitted scaler that surfaces a confusing
                ``NotFittedError`` later in ``transform()``.
        """

        valid_mask = features.notna().all(axis=1)
        if not valid_mask.any():
            raise ValueError(
                f"FeatureEngineeringPipeline.fit(): no rows survived feature warmup. "
                f"Got {len(features)} rows but every row has at least one NaN feature. "
                f"The training frame needs more than {self.hard_nan_warmup_bars} bars of "
                f"non-degenerate OHLCV (e.g. constant volume makes volume_zscore all-NaN)."
            )

        self._scaler = StandardScaler()
        self._scaler.fit(features.loc[valid_mask])
        logger.info("Feature pipeline fitted on %d valid rows", int(valid_mask.sum()))

    def fit_transform(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform in one pass (avoids double feature computation).
        """

        guard_scaler_fit_once(self._scaler, "FeatureEngineeringPipeline")

        features = self._compute_raw_features(train_data)
        self._fit_scaler(features)
        assert self._scaler is not None

        self._apply_scaler_in_place(features)
        return self._maybe_attach_ohlc(features, train_data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scaler.

        Leading NaN from warmup periods is preserved.

        Raises:
            RuntimeError: If called before ``fit()``.
        """

        if self._scaler is None:
            raise RuntimeError(
                "FeatureEngineeringPipeline.transform() called before fit(); "
                "fix by calling pipeline.fit(train_data) first."
            )

        features = self._compute_raw_features(data)
        self._apply_scaler_in_place(features)
        return self._maybe_attach_ohlc(features, data)

    def _maybe_attach_ohlc(self, features: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenate raw OHLCV onto ``features`` when ``keep_ohlc=True``.
        """

        if not self._keep_ohlc:
            return features
        present = [c for c in OHLCV_COLUMNS if c in source.columns]
        if not present:
            return features
        return pd.concat([source[present], features], axis=1)

    def _apply_scaler_in_place(self, features: pd.DataFrame) -> None:
        """
        Scale non-NaN rows in place. Leading warmup NaNs are preserved.
        """

        assert self._scaler is not None
        valid_mask = features.notna().all(axis=1)
        if valid_mask.any():
            features_valid = features.loc[valid_mask]
            features.loc[valid_mask] = self._scaler.transform(features_valid)
