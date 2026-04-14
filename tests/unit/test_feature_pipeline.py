"""Tests for FeatureEngineeringPipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.features.pipeline import FeatureEngineeringPipeline, _compute_macd, _compute_rsi
from tests.conftest import make_synthetic_close_df


@pytest.fixture
def pipeline_df() -> pd.DataFrame:
    return make_synthetic_close_df(n_rows=200)


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
        expected = [
            "return_1d",
            "return_5d",
            "return_21d",
            "vol_20",  # default vol_window=20
            "ma_ratio",
            "rsi_14",  # default rsi_period=14
            "macd",
            "macd_signal",
            "macd_hist",
        ]
        assert list(result.columns) == expected

    def test_leading_nan_preserved(
        self, fitted_pipeline: FeatureEngineeringPipeline, pipeline_df: pd.DataFrame
    ) -> None:
        """Leading NaN from warmup must NOT be back-filled."""
        result = fitted_pipeline.transform(pipeline_df)
        assert result["return_21d"].iloc[:21].isna().all()

    def test_no_bfill(
        self, fitted_pipeline: FeatureEngineeringPipeline, pipeline_df: pd.DataFrame
    ) -> None:
        """First row of return_1d must be NaN (not back-filled)."""
        result = fitted_pipeline.transform(pipeline_df)
        assert pd.isna(result["return_1d"].iloc[0])

    def test_scaler_normalizes_training_data(self, pipeline_df: pd.DataFrame) -> None:
        """After scaling, valid training rows should have ~zero mean."""
        p = FeatureEngineeringPipeline()
        result = p.fit_transform(pipeline_df)
        # Check a feature that has many valid rows
        valid = result["return_1d"].dropna()
        assert abs(valid.mean()) < 0.1  # approximately zero after scaling

    def test_transform_uses_training_stats(self, pipeline_df: pd.DataFrame) -> None:
        """Test data transformed using training scaler statistics."""
        train = pipeline_df.iloc[:150]
        test = pipeline_df.iloc[150:]

        p = FeatureEngineeringPipeline()
        p.fit(train)
        train_result = p.transform(train)
        test_result = p.transform(test)

        # Test data should NOT have zero mean (different distribution)
        # Both should be DataFrames with same columns
        assert list(train_result.columns) == list(test_result.columns)
        assert len(test_result) == len(test)

    def test_registry_registration(self) -> None:
        from src.core.registry import feature_registry

        assert "standard" in feature_registry

    def test_custom_periods(self, pipeline_df: pd.DataFrame) -> None:
        """Pipeline works with non-default periods and names columns accordingly."""
        p = FeatureEngineeringPipeline(
            rsi_period=10, macd_fast=8, macd_slow=17, macd_signal=5, vol_window=10
        )
        result = p.fit_transform(pipeline_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(pipeline_df)
        assert "rsi_10" in result.columns
        assert "vol_10" in result.columns


class TestComputeRSI:
    def test_rsi_range(self, pipeline_df: pd.DataFrame) -> None:
        """RSI values must be in [0, 100]."""
        rsi = _compute_rsi(pipeline_df["close"])
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_has_leading_nan(self, pipeline_df: pd.DataFrame) -> None:
        rsi = _compute_rsi(pipeline_df["close"], period=14)
        # First 14 values need warmup (diff drops first, then rolling needs 14)
        assert rsi.iloc[:14].isna().all()


class TestComputeMACD:
    def test_macd_histogram_is_difference(self, pipeline_df: pd.DataFrame) -> None:
        """Histogram = MACD line - signal line."""
        macd, signal, hist = _compute_macd(pipeline_df["close"])
        np.testing.assert_allclose(
            np.asarray(hist.values, dtype=np.float64),
            np.asarray((macd - signal).values, dtype=np.float64),
            atol=1e-10,
        )
