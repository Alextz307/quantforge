"""Tests for anti-leakage validation decorators."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.core.contracts import no_future_data, no_nan_in_output, temporally_sorted
from src.core.exceptions import LeakageError


class TestNoFutureData:
    def test_passes_for_valid_output(self) -> None:
        @no_future_data
        def good_transform(df: pd.DataFrame) -> pd.DataFrame:
            return df * 2

        idx = pd.DatetimeIndex([datetime(2024, 1, i) for i in range(1, 6)])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        result = good_transform(df)
        assert len(result) == 5

    def test_catches_future_data(self) -> None:
        @no_future_data
        def leaky_transform(df: pd.DataFrame) -> pd.DataFrame:
            future_idx = pd.DatetimeIndex([datetime(2025, 1, i) for i in range(1, 4)])
            return pd.DataFrame({"val": [1.0, 2.0, 3.0]}, index=future_idx)

        idx = pd.DatetimeIndex([datetime(2024, 1, i) for i in range(1, 6)])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)

        with pytest.raises(LeakageError, match="future data"):
            leaky_transform(df)

    def test_passes_for_subset_of_input(self) -> None:
        @no_future_data
        def filter_transform(df: pd.DataFrame) -> pd.DataFrame:
            return df.iloc[:3]

        idx = pd.DatetimeIndex([datetime(2024, 1, i) for i in range(1, 6)])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        result = filter_transform(df)
        assert len(result) == 3

    def test_passes_for_non_dataframe_output(self) -> None:
        @no_future_data
        def scalar_output(df: pd.DataFrame) -> float:
            return df["close"].mean()

        idx = pd.DatetimeIndex([datetime(2024, 1, i) for i in range(1, 6)])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        result = scalar_output(df)
        assert result == 3.0


class TestTemporallySorted:
    def test_passes_for_sorted_data(self) -> None:
        @temporally_sorted
        def process(df: pd.DataFrame) -> pd.DataFrame:
            return df

        idx = pd.DatetimeIndex([datetime(2024, 1, i) for i in range(1, 6)])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        result = process(df)
        assert len(result) == 5

    def test_catches_unsorted_data(self) -> None:
        @temporally_sorted
        def process(df: pd.DataFrame) -> pd.DataFrame:
            return df

        idx = pd.DatetimeIndex([datetime(2024, 1, 5), datetime(2024, 1, 3), datetime(2024, 1, 1)])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)

        with pytest.raises(LeakageError, match="not temporally sorted"):
            process(df)

    def test_passes_for_non_dataframe_arg(self) -> None:
        @temporally_sorted
        def process_scalar(x: float) -> float:
            return x * 2

        result = process_scalar(5.0)
        assert result == 10.0


class TestNoNanInOutput:
    def test_passes_for_clean_output(self) -> None:
        @no_nan_in_output
        def clean_transform(df: pd.DataFrame) -> pd.DataFrame:
            return df * 2

        idx = pd.DatetimeIndex([datetime(2024, 1, i) for i in range(1, 4)])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
        result = clean_transform(df)
        assert len(result) == 3

    def test_catches_nan_in_dataframe(self) -> None:
        @no_nan_in_output
        def nan_transform(df: pd.DataFrame) -> pd.DataFrame:
            result = df.copy()
            result.iloc[0, 0] = np.nan
            return result

        idx = pd.DatetimeIndex([datetime(2024, 1, i) for i in range(1, 4)])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)

        with pytest.raises(ValueError, match="NaN"):
            nan_transform(df)

    def test_catches_nan_in_series(self) -> None:
        @no_nan_in_output
        def nan_series(df: pd.DataFrame) -> pd.Series:
            return pd.Series([1.0, np.nan, 3.0])

        idx = pd.DatetimeIndex([datetime(2024, 1, i) for i in range(1, 4)])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)

        with pytest.raises(ValueError, match="NaN"):
            nan_series(df)

    def test_passes_for_non_dataframe_output(self) -> None:
        @no_nan_in_output
        def scalar_output(df: pd.DataFrame) -> float:
            return 42.0

        idx = pd.DatetimeIndex([datetime(2024, 1, i) for i in range(1, 4)])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
        result = scalar_output(df)
        assert result == 42.0
