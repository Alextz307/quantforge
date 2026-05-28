"""
Tests for src.data.validator.validate_bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import DataQualityError
from src.data.validator import validate_bars
from tests.conftest import make_synthetic_ohlcv_df

TEST_ROWS = 50
BAD_IDX = 10
SECOND_BAD_IDX = 20
THIRD_BAD_IDX = 30
FOURTH_BAD_IDX = 40


@pytest.fixture
def good_df() -> pd.DataFrame:
    return make_synthetic_ohlcv_df(n_rows=TEST_ROWS)


class TestValidateBarsHappyPath:
    def test_accepts_synthetic_fixture(self, good_df: pd.DataFrame) -> None:
        validate_bars(good_df)


class TestValidateBarsPreconditions:
    def test_empty_frame_raises(self) -> None:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with pytest.raises(DataQualityError, match="empty"):
            validate_bars(empty)

    def test_missing_column_raises(self, good_df: pd.DataFrame) -> None:
        with pytest.raises(DataQualityError, match="missing required columns"):
            validate_bars(good_df.drop(columns=["volume"]))

    def test_non_datetime_index_raises(self, good_df: pd.DataFrame) -> None:
        reindexed = good_df.reset_index(drop=True)
        with pytest.raises(DataQualityError, match="DatetimeIndex"):
            validate_bars(reindexed)


class TestValidateBarsNonFinite:
    @pytest.mark.parametrize("col", ["open", "high", "low", "close", "volume"])
    def test_nan_in_any_ohlcv_col_raises(self, good_df: pd.DataFrame, col: str) -> None:
        bad = good_df.copy()
        bad.loc[bad.index[BAD_IDX], col] = np.nan
        with pytest.raises(DataQualityError, match="non-finite"):
            validate_bars(bad)

    @pytest.mark.parametrize("col", ["open", "high", "low", "close", "volume"])
    def test_inf_in_any_ohlcv_col_raises(self, good_df: pd.DataFrame, col: str) -> None:
        bad = good_df.copy()
        bad.loc[bad.index[BAD_IDX], col] = np.inf
        with pytest.raises(DataQualityError, match="non-finite"):
            validate_bars(bad)

    def test_nat_in_datetime_index_raises(self, good_df: pd.DataFrame) -> None:
        idx_with_nat = good_df.index.to_list()
        idx_with_nat[BAD_IDX] = pd.NaT
        bad = good_df.copy()
        bad.index = pd.DatetimeIndex(idx_with_nat)
        with pytest.raises(DataQualityError, match="NaT in index"):
            validate_bars(bad)


class TestValidateBarsSigns:
    @pytest.mark.parametrize("col", ["open", "high", "low", "close"])
    def test_non_positive_price_raises(self, good_df: pd.DataFrame, col: str) -> None:
        bad = good_df.copy()
        bad.loc[bad.index[BAD_IDX], col] = 0.0
        with pytest.raises(DataQualityError, match=f"{col} must be > 0"):
            validate_bars(bad)

    def test_negative_volume_raises(self, good_df: pd.DataFrame) -> None:
        bad = good_df.copy()
        bad.loc[bad.index[BAD_IDX], "volume"] = -1.0
        with pytest.raises(DataQualityError, match="volume must be >= 0"):
            validate_bars(bad)

    def test_zero_volume_allowed(self, good_df: pd.DataFrame) -> None:
        bar_ok = good_df.copy()
        bar_ok.loc[bar_ok.index[BAD_IDX], "volume"] = 0.0
        validate_bars(bar_ok)


class TestValidateBarsOHLCOrdering:
    def test_high_below_open_raises(self, good_df: pd.DataFrame) -> None:
        bad = good_df.copy()
        ts = bad.index[BAD_IDX]
        bad.loc[ts, "high"] = bad.loc[ts, "open"] - 1.0
        with pytest.raises(DataQualityError, match=r"high < max\(open, close\)"):
            validate_bars(bad)

    def test_high_below_close_raises(self, good_df: pd.DataFrame) -> None:
        bad = good_df.copy()
        ts = bad.index[BAD_IDX]
        bad.loc[ts, "high"] = bad.loc[ts, "close"] - 1.0
        with pytest.raises(DataQualityError, match=r"high < max\(open, close\)"):
            validate_bars(bad)

    def test_low_above_open_raises(self, good_df: pd.DataFrame) -> None:
        bad = good_df.copy()
        ts = bad.index[BAD_IDX]
        bad.loc[ts, "low"] = bad.loc[ts, "open"] + 1.0
        with pytest.raises(DataQualityError, match=r"low > min\(open, close\)"):
            validate_bars(bad)

    def test_low_above_close_raises(self, good_df: pd.DataFrame) -> None:
        bad = good_df.copy()
        ts = bad.index[BAD_IDX]
        bad.loc[ts, "low"] = bad.loc[ts, "close"] + 1.0
        with pytest.raises(DataQualityError, match=r"low > min\(open, close\)"):
            validate_bars(bad)

    def test_high_below_low_raises(self, good_df: pd.DataFrame) -> None:
        bad = good_df.copy()
        ts = bad.index[BAD_IDX]
        h = bad.loc[ts, "high"]
        bad.loc[ts, "high"] = bad.loc[ts, "low"]
        bad.loc[ts, "low"] = h
        bad.loc[ts, "open"] = bad.loc[ts, "high"]
        bad.loc[ts, "close"] = bad.loc[ts, "low"]
        with pytest.raises(DataQualityError, match=r"high < max\(open, close\)|high < low"):
            validate_bars(bad)


class TestValidateBarsDuplicateTimestamps:
    def test_duplicate_index_raises(self, good_df: pd.DataFrame) -> None:
        bad = pd.concat([good_df, good_df.iloc[[BAD_IDX]]]).sort_index()
        with pytest.raises(DataQualityError, match="duplicate timestamps"):
            validate_bars(bad)


class TestValidateBarsErrorMessages:
    def test_error_lists_first_offenders_only(self, good_df: pd.DataFrame) -> None:
        bad = good_df.copy()
        bad.loc[bad.index[BAD_IDX], "open"] = -1.0
        bad.loc[bad.index[SECOND_BAD_IDX], "open"] = -1.0
        bad.loc[bad.index[THIRD_BAD_IDX], "open"] = -1.0
        bad.loc[bad.index[FOURTH_BAD_IDX], "open"] = -1.0
        with pytest.raises(DataQualityError) as excinfo:
            validate_bars(bad)
        assert str(bad.index[BAD_IDX]) in str(excinfo.value)
