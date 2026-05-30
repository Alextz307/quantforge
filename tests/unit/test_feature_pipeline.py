"""
Tests for FeatureEngineeringPipeline.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.features.pipeline import FeatureEngineeringPipeline, _compute_macd, _compute_rsi
from tests.conftest import make_synthetic_close_df, make_synthetic_ohlcv_df

PIPELINE_ROW_COUNT = 200
TRAIN_TEST_SPLIT_INDEX = 150

# roc_63 has the longest hard-NaN warmup of any feature.
LONGEST_WARMUP_BARS = 63

# Must mirror FeatureEngineeringPipeline.__init__ defaults.
DEFAULT_RSI_PERIOD = 14
DEFAULT_VOL_WINDOW = 20
DEFAULT_ROC_PERIOD = 63
DEFAULT_ADX_PERIOD = 14

RSI_LOWER_BOUND = 0
RSI_UPPER_BOUND = 100
ADX_LOWER_BOUND = 0.0
ADX_UPPER_BOUND = 100.0

DEFAULT_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_21d",
    f"vol_{DEFAULT_VOL_WINDOW}",
    "ma_ratio",
    f"rsi_{DEFAULT_RSI_PERIOD}",
    "macd",
    "macd_signal",
    "macd_hist",
    f"roc_{DEFAULT_ROC_PERIOD}",
    "garman_klass",
    "intraday_range",
    "overnight_gap",
    "bb_pctb",
    f"adx_{DEFAULT_ADX_PERIOD}",
    "volume_zscore",
    "obv_z",
]

CUSTOM_RSI_PERIOD = 10
CUSTOM_MACD_FAST = 8
CUSTOM_MACD_SLOW = 17
CUSTOM_MACD_SIGNAL = 5
CUSTOM_VOL_WINDOW = 10

SCALED_MEAN_TOLERANCE = 0.1
MACD_HIST_ATOL = 1e-10

# A clean monotone uptrend drives DX (hence ADX) toward 100: all directional
# movement is positive, so -DI collapses to 0 and |+DI - -DI| / (+DI + -DI) = 1.
MONOTONE_RAMP_ROWS = 120
MONOTONE_STEP = 0.5
ADX_TREND_FLOOR = 90.0

OVERNIGHT_GAP_ROWS = 120
OVERNIGHT_GAP_STD = 0.01
OVERNIGHT_GAP_SEED = 7

# Below the longest (roc_63) warmup, so every row carries a NaN feature.
BELOW_WARMUP_ROWS = 40
# A flat-volume stretch longer than the z-score window (forces rolling std 0).
FLAT_VOLUME_START = 50
FLAT_VOLUME_LEN = 40

# Causality cut points (< PIPELINE_ROW_COUNT) and the gapped-frame seed.
CAUSALITY_PREFIX_LENS = (100, 150)
CAUSALITY_GAP_SEED = 11

# (column, leading-NaN count) for each feature's no-fill warmup region.
RETURN_1D_WARMUP = 1
RETURN_5D_WARMUP = 5
RETURN_21D_WARMUP = 21
OVERNIGHT_GAP_WARMUP = 1
ADX_WARMUP = 2 * DEFAULT_ADX_PERIOD - 1
WARMUP_CASES = (
    ("return_1d", RETURN_1D_WARMUP),
    ("return_5d", RETURN_5D_WARMUP),
    ("return_21d", RETURN_21D_WARMUP),
    (f"vol_{DEFAULT_VOL_WINDOW}", DEFAULT_VOL_WINDOW),
    ("overnight_gap", OVERNIGHT_GAP_WARMUP),
    (f"adx_{DEFAULT_ADX_PERIOD}", ADX_WARMUP),
    (f"roc_{DEFAULT_ROC_PERIOD}", DEFAULT_ROC_PERIOD),
)


def _gapped_ohlcv_df(n_rows: int, seed: int) -> pd.DataFrame:
    """
    OHLCV with genuine overnight gaps (open != prior close).

    The shared fixture sets open == prior close, making overnight_gap
    identically 0; injecting gaps keeps every feature non-degenerate so the
    causality and variance checks are not vacuous.
    """

    base = make_synthetic_ohlcv_df(n_rows=n_rows, seed=seed)
    rng = np.random.default_rng(seed)
    gap = rng.normal(0.0, OVERNIGHT_GAP_STD, n_rows)
    open_with_gap = base["close"].shift(1).to_numpy() * (1.0 + gap)
    open_with_gap[0] = base["close"].iloc[0]
    base["open"] = open_with_gap
    base["high"] = base[["open", "high", "close"]].max(axis=1)
    base["low"] = base[["open", "low", "close"]].min(axis=1)
    return base


@pytest.fixture
def pipeline_df() -> pd.DataFrame:
    return make_synthetic_ohlcv_df(n_rows=PIPELINE_ROW_COUNT)


@pytest.fixture
def fitted_pipeline(pipeline_df: pd.DataFrame) -> FeatureEngineeringPipeline:
    p = FeatureEngineeringPipeline()
    p.fit(pipeline_df)
    return p


class TestFeatureEngineeringPipeline:
    def test_transform_before_fit_raises(self, pipeline_df: pd.DataFrame) -> None:
        p = FeatureEngineeringPipeline()
        with pytest.raises(RuntimeError, match="before fit"):
            p.transform(pipeline_df)

    def test_fit_twice_raises_leakage(self, pipeline_df: pd.DataFrame) -> None:
        p = FeatureEngineeringPipeline()
        p.fit(pipeline_df)
        with pytest.raises(LeakageError):
            p.fit(pipeline_df)

    def test_fit_transform_works(self, pipeline_df: pd.DataFrame) -> None:
        p = FeatureEngineeringPipeline()
        result = p.fit_transform(pipeline_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(pipeline_df)

    def test_output_columns(
        self, fitted_pipeline: FeatureEngineeringPipeline, pipeline_df: pd.DataFrame
    ) -> None:
        result = fitted_pipeline.transform(pipeline_df)
        assert list(result.columns) == DEFAULT_FEATURE_COLUMNS

    def test_missing_ohlcv_raises(self) -> None:
        """
        The pipeline needs full OHLCV; a close-only frame fails fast.
        """

        close_only = make_synthetic_close_df(n_rows=PIPELINE_ROW_COUNT)
        p = FeatureEngineeringPipeline()
        with pytest.raises(ValueError, match="OHLCV"):
            p.fit(close_only)

    def test_fit_below_warmup_raises(self) -> None:
        """
        A frame shorter than the longest warmup leaves every row with a NaN
        feature; fit must fail fast rather than leave an unfitted scaler that
        only surfaces a confusing NotFittedError later in transform().
        """

        short = make_synthetic_ohlcv_df(n_rows=BELOW_WARMUP_ROWS)
        with pytest.raises(ValueError, match="warmup"):
            FeatureEngineeringPipeline().fit(short)

    def test_flat_volume_window_is_nan_not_inf(self, pipeline_df: pd.DataFrame) -> None:
        """
        A flat-volume stretch makes the rolling std 0; volume_zscore must
        yield NaN (warmup-style), never inf or a divide-by-zero warning.
        """

        df = pipeline_df.copy()
        vol = df["volume"].to_numpy().copy()
        flat_end = FLAT_VOLUME_START + FLAT_VOLUME_LEN
        vol[FLAT_VOLUME_START:flat_end] = vol[FLAT_VOLUME_START]
        df["volume"] = vol

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            raw = FeatureEngineeringPipeline()._compute_raw_features(df)

        vz = raw["volume_zscore"]
        assert not np.isinf(vz.to_numpy()).any()
        assert vz.iloc[FLAT_VOLUME_START + DEFAULT_VOL_WINDOW : flat_end].isna().all()

    def test_leading_nan_preserved(
        self, fitted_pipeline: FeatureEngineeringPipeline, pipeline_df: pd.DataFrame
    ) -> None:
        """
        Leading NaN from the longest warmup must NOT be back-filled.
        """

        result = fitted_pipeline.transform(pipeline_df)
        assert result[f"roc_{DEFAULT_ROC_PERIOD}"].iloc[:LONGEST_WARMUP_BARS].isna().all()

    @pytest.mark.parametrize("column,warmup", WARMUP_CASES)
    def test_warmup_rows_are_nan(self, pipeline_df: pd.DataFrame, column: str, warmup: int) -> None:
        """
        Each feature's warmup region stays NaN - never zero- or back-filled.

        The causality test treats 0 == 0 as a match, so a fillna(0) on a
        mid-horizon feature's warmup would slip past it; this pins the
        leading-NaN region per feature on the raw (pre-scale) output.
        """

        raw = FeatureEngineeringPipeline()._compute_raw_features(pipeline_df)
        assert raw[column].iloc[:warmup].isna().all(), column

    @pytest.mark.parametrize("column", DEFAULT_FEATURE_COLUMNS)
    @pytest.mark.parametrize("prefix_len", CAUSALITY_PREFIX_LENS)
    def test_features_are_causal(self, column: str, prefix_len: int) -> None:
        """
        No feature at row t may read any bar > t.

        A feature is causal iff computing it on a truncated prefix yields
        the same values (on the overlapping rows) as computing it on the
        full series. Runs on a gapped frame (so overnight_gap is not the
        degenerate all-zero column the shared fixture produces), sweeps two
        cut points, and asserts the compared overlap actually varies so a
        constant or all-NaN column cannot pass vacuously. Holds the C++
        Garman-Klass / Bollinger features and the cumulative OBV to the same
        standard as the pandas ones.
        """

        df = _gapped_ohlcv_df(PIPELINE_ROW_COUNT, CAUSALITY_GAP_SEED)
        p = FeatureEngineeringPipeline()
        full = p._compute_raw_features(df)
        prefix = p._compute_raw_features(df.iloc[:prefix_len])

        full_col = full[column].iloc[:prefix_len]
        prefix_col = prefix[column]
        assert prefix_col.dropna().nunique() > 1, f"{column} degenerate; causality check vacuous"
        both_nan = full_col.isna() & prefix_col.isna()
        assert bool(((full_col == prefix_col) | both_nan).all()), column

    def test_garman_klass_nonnegative(self, pipeline_df: pd.DataFrame) -> None:
        raw = FeatureEngineeringPipeline()._compute_raw_features(pipeline_df)
        assert (raw["garman_klass"].dropna() >= 0.0).all()

    def test_intraday_range_nonnegative(self, pipeline_df: pd.DataFrame) -> None:
        raw = FeatureEngineeringPipeline()._compute_raw_features(pipeline_df)
        assert (raw["intraday_range"].dropna() >= 0.0).all()

    def test_adx_within_bounds(self, pipeline_df: pd.DataFrame) -> None:
        raw = FeatureEngineeringPipeline()._compute_raw_features(pipeline_df)
        adx = raw[f"adx_{DEFAULT_ADX_PERIOD}"].dropna()
        assert (adx >= ADX_LOWER_BOUND).all()
        assert (adx <= ADX_UPPER_BOUND).all()

    def test_adx_saturates_on_clean_trend(self) -> None:
        """
        A pure uptrend has only positive directional movement, so DX (and
        thus the Wilder-smoothed ADX) approaches 100. Locks the
        ewm(adjust=False) smoothing convention behaviorally.
        """

        idx = pd.bdate_range(start="2021-01-04", periods=MONOTONE_RAMP_ROWS, freq="B")
        close = pd.Series(
            np.arange(MONOTONE_RAMP_ROWS, dtype=np.float64) * MONOTONE_STEP + 100.0,
            index=idx,
        )
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + MONOTONE_STEP,
                "low": close - MONOTONE_STEP,
                "close": close,
                "volume": np.full(MONOTONE_RAMP_ROWS, 1.0),
            },
            index=idx,
        )
        raw = FeatureEngineeringPipeline()._compute_raw_features(df)
        assert raw[f"adx_{DEFAULT_ADX_PERIOD}"].dropna().iloc[-1] >= ADX_TREND_FLOOR

    def test_bb_pctb_and_volume_features_finite(self, pipeline_df: pd.DataFrame) -> None:
        raw = FeatureEngineeringPipeline()._compute_raw_features(pipeline_df)
        for col in ("bb_pctb", "volume_zscore", "obv_z"):
            values = raw[col].dropna().to_numpy()
            assert np.isfinite(values).all(), col
            assert len(values) > 0, col

    def test_overnight_gap_has_variance(self) -> None:
        """
        On data with genuine overnight gaps, the feature varies (the shared
        OHLCV fixture sets open == prior close, so gaps are ~0 there).
        """

        base = _gapped_ohlcv_df(OVERNIGHT_GAP_ROWS, OVERNIGHT_GAP_SEED)
        raw = FeatureEngineeringPipeline()._compute_raw_features(base)
        assert raw["overnight_gap"].dropna().std() > 0.0

    def test_scaler_normalizes_training_data(self, pipeline_df: pd.DataFrame) -> None:
        """
        After scaling, valid training rows should have ~zero mean.
        """

        p = FeatureEngineeringPipeline()
        result = p.fit_transform(pipeline_df)
        valid = result["return_1d"].dropna()
        assert abs(valid.mean()) < SCALED_MEAN_TOLERANCE

    def test_transform_uses_training_stats(self, pipeline_df: pd.DataFrame) -> None:
        """
        Test data transformed using training scaler statistics.
        """

        train = pipeline_df.iloc[:TRAIN_TEST_SPLIT_INDEX]
        test = pipeline_df.iloc[TRAIN_TEST_SPLIT_INDEX:]

        p = FeatureEngineeringPipeline()
        p.fit(train)
        train_result = p.transform(train)
        test_result = p.transform(test)

        assert list(train_result.columns) == list(test_result.columns)
        assert len(test_result) == len(test)

    def test_registry_registration(self) -> None:
        from src.core.registry import feature_registry

        assert "standard" in feature_registry

    def test_custom_periods(self, pipeline_df: pd.DataFrame) -> None:
        """
        Pipeline works with non-default periods and names columns accordingly.
        """

        p = FeatureEngineeringPipeline(
            rsi_period=CUSTOM_RSI_PERIOD,
            macd_fast=CUSTOM_MACD_FAST,
            macd_slow=CUSTOM_MACD_SLOW,
            macd_signal=CUSTOM_MACD_SIGNAL,
            vol_window=CUSTOM_VOL_WINDOW,
        )
        result = p.fit_transform(pipeline_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(pipeline_df)
        assert f"rsi_{CUSTOM_RSI_PERIOD}" in result.columns
        assert f"vol_{CUSTOM_VOL_WINDOW}" in result.columns

    def test_keep_ohlc_passthrough(self) -> None:
        """
        When keep_ohlc=True, OHLCV survives both fit_transform and transform.
        """

        ohlcv = make_synthetic_ohlcv_df(n_rows=PIPELINE_ROW_COUNT, seed=42)
        p = FeatureEngineeringPipeline(keep_ohlc=True)
        train_result = p.fit_transform(ohlcv)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in train_result.columns, col
            assert (train_result[col] == ohlcv[col]).all(), col
        for engineered in DEFAULT_FEATURE_COLUMNS:
            assert engineered in train_result.columns

        held_out = ohlcv.iloc[len(ohlcv) // 2 :].copy()
        test_result = p.transform(held_out)
        assert (test_result["close"] == held_out["close"]).all()


class TestComputeRSI:
    def test_rsi_range(self, pipeline_df: pd.DataFrame) -> None:
        """
        RSI values must be in [0, 100].
        """

        rsi = _compute_rsi(pipeline_df["close"])
        valid = rsi.dropna()
        assert (valid >= RSI_LOWER_BOUND).all()
        assert (valid <= RSI_UPPER_BOUND).all()

    def test_rsi_has_leading_nan(self, pipeline_df: pd.DataFrame) -> None:
        rsi = _compute_rsi(pipeline_df["close"], period=DEFAULT_RSI_PERIOD)
        assert rsi.iloc[:DEFAULT_RSI_PERIOD].isna().all()


class TestComputeMACD:
    def test_macd_histogram_is_difference(self, pipeline_df: pd.DataFrame) -> None:
        """
        Histogram = MACD line - signal line.
        """

        macd, signal, hist = _compute_macd(pipeline_df["close"])
        np.testing.assert_allclose(
            np.asarray(hist.values, dtype=np.float64),
            np.asarray((macd - signal).values, dtype=np.float64),
            atol=MACD_HIST_ATOL,
        )
