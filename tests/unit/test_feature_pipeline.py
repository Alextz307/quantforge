"""
Tests for FeatureEngineeringPipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.features.pipeline import FeatureEngineeringPipeline, _compute_macd, _compute_rsi
from tests.conftest import make_synthetic_close_df

PIPELINE_ROW_COUNT = 200
TRAIN_TEST_SPLIT_INDEX = 150
RETURN_21D_WARMUP_BARS = 21

# Must mirror FeatureEngineeringPipeline.__init__ defaults.
DEFAULT_RSI_PERIOD = 14
DEFAULT_VOL_WINDOW = 20

RSI_LOWER_BOUND = 0
RSI_UPPER_BOUND = 100

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
]

CUSTOM_RSI_PERIOD = 10
CUSTOM_MACD_FAST = 8
CUSTOM_MACD_SLOW = 17
CUSTOM_MACD_SIGNAL = 5
CUSTOM_VOL_WINDOW = 10

SCALED_MEAN_TOLERANCE = 0.1
MACD_HIST_ATOL = 1e-10


@pytest.fixture
def pipeline_df() -> pd.DataFrame:
    return make_synthetic_close_df(n_rows=PIPELINE_ROW_COUNT)


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

    def test_leading_nan_preserved(
        self, fitted_pipeline: FeatureEngineeringPipeline, pipeline_df: pd.DataFrame
    ) -> None:
        """
        Leading NaN from warmup must NOT be back-filled.
        """

        result = fitted_pipeline.transform(pipeline_df)
        assert result["return_21d"].iloc[:RETURN_21D_WARMUP_BARS].isna().all()

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

        from tests.conftest import make_synthetic_ohlcv_df

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
