"""
Day-boundary walk-forward tests (`snap_to_day=True`).

Honours the framework's intraday day-boundary rule: training cutoff must
land on a day close, even when the underlying bars are intraday. The gap
argument is reinterpreted as trading days of embargo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.temporal import (
    WalkForwardValidator,
    _first_bar_after_gap_days,
    _snap_train_end_backward,
)
from tests.conftest import make_daily_df

# Hourly bars starting at midnight -> every date has exactly 24 bars, which
# keeps the snap arithmetic trivial to verify by hand.
HOURLY_FREQ = "h"
HOURLY_START = "2020-01-01 00:00"
HOURLY_BARS_PER_DAY = 24
HOURLY_DAYS = 40
HOURLY_ROWS = HOURLY_DAYS * HOURLY_BARS_PER_DAY
HOURLY_SEED = 7
HOURLY_N_SPLITS = 3
HOURLY_TEST_SIZE = HOURLY_BARS_PER_DAY * 2

DAILY_ROWS = 1000
DAILY_START = "2018-01-02"
DAILY_N_SPLITS = 4
DAILY_TEST_SIZE = 60
DAILY_GAP = 5

HELPER_ROWS = 240
HELPER_TRAIN_END = HOURLY_BARS_PER_DAY
HELPER_GAP_DAYS = 3

HOLIDAY_PRE_BARS = 48
HOLIDAY_POST_BARS = 48


def _make_hourly_close_df(n_rows: int = HOURLY_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(HOURLY_SEED)
    idx = pd.date_range(start=HOURLY_START, periods=n_rows, freq=HOURLY_FREQ)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n_rows))
    return pd.DataFrame({"close": close}, index=idx)


@pytest.fixture
def helper_dates() -> pd.DatetimeIndex:
    idx = pd.date_range(HOURLY_START, periods=HELPER_ROWS, freq=HOURLY_FREQ)
    assert isinstance(idx, pd.DatetimeIndex)
    return idx.normalize()


class TestSnapToDayIntraday:
    def test_every_boundary_crosses_midnight(self) -> None:
        df = _make_hourly_close_df()
        wf = WalkForwardValidator(
            n_splits=HOURLY_N_SPLITS,
            test_size=HOURLY_TEST_SIZE,
            gap=0,
            snap_to_day=True,
        )
        for s in wf.split(df):
            train_last = s.train.index[-1]
            test_first = s.test.index[0]
            assert train_last.date() != test_first.date(), (
                f"Fold {s.fold_index}: train ended {train_last}, test started {test_first} - "
                "snap_to_day must force a day change across the boundary."
            )

    def test_train_ends_on_last_bar_of_its_day(self) -> None:
        df = _make_hourly_close_df()
        wf = WalkForwardValidator(
            n_splits=HOURLY_N_SPLITS,
            test_size=HOURLY_TEST_SIZE,
            gap=0,
            snap_to_day=True,
        )
        assert isinstance(df.index, pd.DatetimeIndex)
        dates = df.index.normalize()
        for s in wf.split(df):
            train_end_pos = df.index.get_loc(s.train.index[-1])
            assert isinstance(train_end_pos, int)
            if train_end_pos + 1 < len(df):
                assert dates[train_end_pos] < dates[train_end_pos + 1]


class TestSnapHelpers:
    """
    Direct tests for the snap helpers - isolates day-based arithmetic
    from the full WalkForwardValidator.split() geometric budget."""

    def test_gap_counts_distinct_dates_not_bars(self, helper_dates: pd.DatetimeIndex) -> None:
        first_bar_jan5 = HOURLY_BARS_PER_DAY * 4
        assert (
            _first_bar_after_gap_days(helper_dates, HELPER_TRAIN_END, HELPER_GAP_DAYS)
            == first_bar_jan5
        )

    def test_zero_gap_returns_first_bar_of_next_day(self, helper_dates: pd.DatetimeIndex) -> None:
        assert _first_bar_after_gap_days(helper_dates, HELPER_TRAIN_END, 0) == HOURLY_BARS_PER_DAY

    def test_snap_already_at_boundary_is_noop(self, helper_dates: pd.DatetimeIndex) -> None:
        assert _snap_train_end_backward(helper_dates, HOURLY_BARS_PER_DAY) == HOURLY_BARS_PER_DAY

    def test_snap_mid_day_walks_back(self, helper_dates: pd.DatetimeIndex) -> None:
        mid_day_two = HOURLY_BARS_PER_DAY + 12
        assert _snap_train_end_backward(helper_dates, mid_day_two) == HOURLY_BARS_PER_DAY

    def test_first_bar_helper_rejects_zero_train_end(self, helper_dates: pd.DatetimeIndex) -> None:
        with pytest.raises(ValueError, match="train_end >= 1"):
            _first_bar_after_gap_days(helper_dates, 0, 1)

    def test_holiday_gap_uses_observed_dates(self) -> None:
        first_block = pd.date_range("2020-01-06 00:00", periods=HOLIDAY_PRE_BARS, freq="h")
        second_block = pd.date_range("2020-01-13 00:00", periods=HOLIDAY_POST_BARS, freq="h")
        idx = first_block.append(second_block)
        assert isinstance(idx, pd.DatetimeIndex)
        dates = idx.normalize()
        result = _first_bar_after_gap_days(dates, HOLIDAY_PRE_BARS, 1)
        assert result == HOLIDAY_PRE_BARS + HOURLY_BARS_PER_DAY


class TestSnapToDaySlidingWindow:
    def test_sliding_preserves_window_size(self) -> None:
        df = _make_hourly_close_df()
        wf = WalkForwardValidator(
            n_splits=HOURLY_N_SPLITS,
            test_size=HOURLY_TEST_SIZE,
            gap=0,
            expanding=False,
            snap_to_day=True,
        )
        train_sizes = [len(s.train) for s in wf.split(df)]
        assert len(set(train_sizes)) == 1


class TestSnapToDayDaily:
    def test_noop_on_daily_data(self) -> None:
        """
        Every daily bar is already at a day close; snap must produce the
        same splits as the default path."""

        df = make_daily_df(DAILY_ROWS, start=DAILY_START)
        default_splits = list(
            WalkForwardValidator(
                n_splits=DAILY_N_SPLITS, test_size=DAILY_TEST_SIZE, gap=DAILY_GAP
            ).split(df)
        )
        snapped_splits = list(
            WalkForwardValidator(
                n_splits=DAILY_N_SPLITS,
                test_size=DAILY_TEST_SIZE,
                gap=DAILY_GAP,
                snap_to_day=True,
            ).split(df)
        )

        assert len(default_splits) == len(snapped_splits)
        for default, snapped in zip(default_splits, snapped_splits, strict=True):
            pd.testing.assert_frame_equal(default.train, snapped.train)
            pd.testing.assert_frame_equal(default.test, snapped.test)


class TestSnapToDayErrors:
    def test_single_distinct_date_raises(self) -> None:
        idx = pd.date_range(start="2020-01-02 00:00", periods=HOURLY_BARS_PER_DAY, freq="h")
        df = pd.DataFrame({"close": np.arange(HOURLY_BARS_PER_DAY, dtype=float)}, index=idx)
        wf = WalkForwardValidator(n_splits=1, test_size=5, gap=0, snap_to_day=True)
        with pytest.raises(ValueError, match="at least 2 distinct dates"):
            list(wf.split(df))

    def test_insufficient_future_days_raises(self) -> None:
        df = _make_hourly_close_df(n_rows=3 * HOURLY_BARS_PER_DAY)
        wf = WalkForwardValidator(n_splits=1, test_size=10, gap=5, snap_to_day=True)
        with pytest.raises(ValueError, match="snap_to_day"):
            list(wf.split(df))

    def test_last_fold_test_window_overrun_raises(self) -> None:
        """
        Snap pushes the last fold's test window past end-of-frame - must
        raise rather than silently truncate."""

        df = _make_hourly_close_df(n_rows=4 * HOURLY_BARS_PER_DAY)
        wf = WalkForwardValidator(n_splits=1, test_size=50, gap=1, snap_to_day=True)
        with pytest.raises(ValueError, match="past end-of-frame"):
            list(wf.split(df))
