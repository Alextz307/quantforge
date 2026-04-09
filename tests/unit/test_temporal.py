"""Tests for temporal validation: TemporalSplit and WalkForwardValidator."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.temporal import (
    PurgedGroupTimeSeriesSplit,
    TemporalSplit,
    WalkForwardValidator,
)


def _make_daily_df(n_rows: int, start: str = "2020-01-01") -> pd.DataFrame:
    """Create a simple DataFrame with DatetimeIndex for testing."""
    idx = pd.bdate_range(start=start, periods=n_rows, freq="B")
    return pd.DataFrame(
        {"close": range(n_rows), "volume": [1000] * n_rows},
        index=idx,
    )


class TestTemporalSplit:
    def test_valid_split(self) -> None:
        df = _make_daily_df(100)
        train = df.iloc[:50]
        test = df.iloc[55:]
        split = TemporalSplit(
            train=train,
            test=test,
            split_date=pd.Timestamp(test.index[0]),
            fold_index=0,
        )
        assert split.fold_index == 0
        assert len(split.train) == 50

    def test_rejects_overlapping_train_test(self) -> None:
        df = _make_daily_df(100)
        train = df.iloc[:60]
        test = df.iloc[50:]  # overlaps with train
        with pytest.raises(LeakageError, match="overlaps"):
            TemporalSplit(
                train=train,
                test=test,
                split_date=pd.Timestamp(test.index[0]),
            )

    def test_rejects_adjacent_train_test(self) -> None:
        """Train end == test start (same timestamp) should be rejected."""
        idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=10, freq="D"))
        data = pd.DataFrame({"v": range(10)}, index=idx)
        # Create overlap: train includes day 5, test starts at day 5
        train_overlap = data.iloc[:6]  # days 0-5
        test_overlap = data.iloc[5:]  # days 5-9 (day 5 is in both)
        with pytest.raises(LeakageError):
            TemporalSplit(
                train=train_overlap,
                test=test_overlap,
                split_date=pd.Timestamp(test_overlap.index[0]),
            )

    def test_rejects_empty_train(self) -> None:
        df = _make_daily_df(10)
        empty = df.iloc[:0]
        test = df.iloc[5:]
        with pytest.raises(ValueError, match="non-empty"):
            TemporalSplit(
                train=empty,
                test=test,
                split_date=pd.Timestamp(test.index[0]),
            )

    def test_rejects_empty_test(self) -> None:
        df = _make_daily_df(10)
        train = df.iloc[:5]
        empty = df.iloc[:0]
        with pytest.raises(ValueError, match="non-empty"):
            TemporalSplit(
                train=train,
                test=empty,
                split_date=pd.Timestamp("2024-01-01"),
            )

    def test_rejects_non_datetime_index(self) -> None:
        train = pd.DataFrame({"v": [1, 2, 3]})
        test = pd.DataFrame({"v": [4, 5, 6]})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            TemporalSplit(
                train=train,
                test=test,
                split_date=pd.Timestamp("2024-01-01"),
            )

    def test_frozen(self) -> None:
        df = _make_daily_df(100)
        split = TemporalSplit(
            train=df.iloc[:40],
            test=df.iloc[50:],
            split_date=pd.Timestamp(df.index[50]),
        )
        with pytest.raises(AttributeError):
            split.fold_index = 5  # type: ignore[misc]


class TestWalkForwardValidator:
    def test_correct_number_of_splits(self) -> None:
        df = _make_daily_df(2000)
        validator = WalkForwardValidator(n_splits=4, test_size=252, gap=5)
        splits = list(validator.split(df))
        assert len(splits) == 4

    def test_splits_have_correct_fold_indices(self) -> None:
        df = _make_daily_df(2000)
        validator = WalkForwardValidator(n_splits=3, test_size=200, gap=5)
        splits = list(validator.split(df))
        assert [s.fold_index for s in splits] == [0, 1, 2]

    def test_test_size_is_correct(self) -> None:
        df = _make_daily_df(2000)
        validator = WalkForwardValidator(n_splits=4, test_size=252, gap=5)
        splits = list(validator.split(df))
        for split in splits:
            assert len(split.test) == 252

    def test_gap_is_respected(self) -> None:
        df = _make_daily_df(2000)
        validator = WalkForwardValidator(n_splits=4, test_size=252, gap=10)
        splits = list(validator.split(df))
        for split in splits:
            train_end_idx = df.index.get_loc(split.train.index[-1])
            test_start_idx = df.index.get_loc(split.test.index[0])
            assert test_start_idx - train_end_idx > 10  # type: ignore[operator]

    def test_expanding_window_grows(self) -> None:
        df = _make_daily_df(2000)
        validator = WalkForwardValidator(n_splits=4, test_size=252, gap=5, expanding=True)
        splits = list(validator.split(df))
        train_sizes = [len(s.train) for s in splits]
        # Each subsequent training set should be larger
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_no_temporal_leakage(self) -> None:
        """Every split's train max < test min (guaranteed by TemporalSplit)."""
        df = _make_daily_df(2000)
        validator = WalkForwardValidator(n_splits=4, test_size=252, gap=5)
        splits = list(validator.split(df))
        for split in splits:
            assert split.train.index.max() < split.test.index.min()

    def test_insufficient_data_raises(self) -> None:
        df = _make_daily_df(50)
        validator = WalkForwardValidator(n_splits=4, test_size=252, gap=5)
        with pytest.raises(ValueError, match="too few|required"):
            list(validator.split(df))

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            WalkForwardValidator(n_splits=0)
        with pytest.raises(ValueError):
            WalkForwardValidator(test_size=0)
        with pytest.raises(ValueError):
            WalkForwardValidator(gap=-1)

    def test_single_split(self) -> None:
        df = _make_daily_df(500)
        validator = WalkForwardValidator(n_splits=1, test_size=100, gap=5)
        splits = list(validator.split(df))
        assert len(splits) == 1
        assert len(splits[0].test) == 100


class TestPurgedGroupTimeSeriesSplit:
    def test_correct_number_of_splits(self) -> None:
        df = _make_daily_df(1000)
        splitter = PurgedGroupTimeSeriesSplit(n_groups=5, embargo_pct=0.01)
        splits = list(splitter.split(df))
        assert len(splits) == splitter.n_folds  # n_groups - 1 = 4

    def test_embargo_removes_boundary_data(self) -> None:
        df = _make_daily_df(1000)
        splitter = PurgedGroupTimeSeriesSplit(n_groups=5, embargo_pct=0.02)
        splits = list(splitter.split(df))
        for split in splits:
            # Verify gap between train end and test start
            assert split.train.index.max() < split.test.index.min()

    def test_no_overlap(self) -> None:
        df = _make_daily_df(500)
        splitter = PurgedGroupTimeSeriesSplit(n_groups=5, embargo_pct=0.01)
        splits = list(splitter.split(df))
        for split in splits:
            train_dates = set(split.train.index)
            test_dates = set(split.test.index)
            assert train_dates.isdisjoint(test_dates)

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            PurgedGroupTimeSeriesSplit(n_groups=1)
        with pytest.raises(ValueError):
            PurgedGroupTimeSeriesSplit(embargo_pct=1.0)
        with pytest.raises(ValueError):
            PurgedGroupTimeSeriesSplit(embargo_pct=-0.1)

    def test_warns_when_fold_skipped(self) -> None:
        df = _make_daily_df(20)
        splitter = PurgedGroupTimeSeriesSplit(n_groups=2, embargo_pct=0.99)
        with pytest.warns(UserWarning, match="skipped"):
            splits = list(splitter.split(df))
        assert len(splits) < splitter.n_folds

    def test_too_few_rows(self) -> None:
        df = _make_daily_df(5)
        splitter = PurgedGroupTimeSeriesSplit(n_groups=10)
        with pytest.raises(ValueError, match="too few"):
            list(splitter.split(df))
