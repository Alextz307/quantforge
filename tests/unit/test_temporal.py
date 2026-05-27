"""Tests for temporal validation: TemporalSplit and WalkForwardValidator."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.temporal import (
    PurgedGroupTimeSeriesSplit,
    TemporalSplit,
    WalkForwardValidator,
    resolve_holdout_boundary,
)
from tests.conftest import make_daily_df

SMALL_DF_ROWS = 100
TRAIN_SLICE_END = 50
TEST_SLICE_START = 55
OVERLAP_TRAIN_END = 60
TINY_DF_ROWS = 10
SHORT_DF_ROWS = 5
SMALL_TEST_DF_ROWS = 3

# Adjacent-overlap test: train ends one row past where test starts so they
# share a single boundary row, which TemporalSplit must reject as a leak.
ADJACENT_DF_ROWS = 10
ADJACENT_DF_START = "2024-01-01"
ADJACENT_OVERLAP_INDEX = 5
ADJACENT_TRAIN_END = ADJACENT_OVERLAP_INDEX + 1
ADJACENT_TEST_START = ADJACENT_OVERLAP_INDEX

WF_LARGE_DF_ROWS = 2000
WF_DEFAULT_N_SPLITS = 4
WF_DEFAULT_TEST_SIZE = 252
WF_DEFAULT_GAP = 5
WF_LARGE_GAP = 10
WF_THREE_SPLITS = 3
WF_THREE_TEST_SIZE = 200
WF_INSUFFICIENT_DF_ROWS = 50
WF_SINGLE_SPLIT_DF_ROWS = 500
WF_SINGLE_TEST_SIZE = 100

PG_DF_ROWS = 1000
PG_OVERLAP_DF_ROWS = 500
PG_DEFAULT_N_GROUPS = 5
PG_SMALL_EMBARGO_PCT = 0.01
PG_MEDIUM_EMBARGO_PCT = 0.02
PG_TINY_DF_ROWS = 20
PG_TWO_GROUPS = 2
PG_AGGRESSIVE_EMBARGO_PCT = 0.99
PG_TINIEST_DF_ROWS = 5
PG_LARGE_N_GROUPS = 10


class TestTemporalSplit:
    def test_valid_split(self) -> None:
        df = make_daily_df(SMALL_DF_ROWS)
        train = df.iloc[:TRAIN_SLICE_END]
        test = df.iloc[TEST_SLICE_START:]
        split = TemporalSplit(
            train=train,
            test=test,
            split_date=pd.Timestamp(test.index[0]),
            fold_index=0,
        )
        assert split.fold_index == 0
        assert len(split.train) == TRAIN_SLICE_END

    def test_rejects_overlapping_train_test(self) -> None:
        df = make_daily_df(SMALL_DF_ROWS)
        train = df.iloc[:OVERLAP_TRAIN_END]
        test = df.iloc[TRAIN_SLICE_END:]
        with pytest.raises(LeakageError, match="overlaps"):
            TemporalSplit(
                train=train,
                test=test,
                split_date=pd.Timestamp(test.index[0]),
            )

    def test_rejects_adjacent_train_test(self) -> None:
        """Train end == test start (same timestamp) should be rejected."""

        idx = pd.DatetimeIndex(pd.date_range(ADJACENT_DF_START, periods=ADJACENT_DF_ROWS, freq="D"))
        data = pd.DataFrame({"v": range(ADJACENT_DF_ROWS)}, index=idx)
        train_overlap = data.iloc[:ADJACENT_TRAIN_END]
        test_overlap = data.iloc[ADJACENT_TEST_START:]
        with pytest.raises(LeakageError):
            TemporalSplit(
                train=train_overlap,
                test=test_overlap,
                split_date=pd.Timestamp(test_overlap.index[0]),
            )

    def test_rejects_empty_train(self) -> None:
        df = make_daily_df(TINY_DF_ROWS)
        empty = df.iloc[:0]
        test = df.iloc[SHORT_DF_ROWS:]
        with pytest.raises(ValueError, match="non-empty"):
            TemporalSplit(
                train=empty,
                test=test,
                split_date=pd.Timestamp(test.index[0]),
            )

    def test_rejects_empty_test(self) -> None:
        df = make_daily_df(TINY_DF_ROWS)
        train = df.iloc[:SHORT_DF_ROWS]
        empty = df.iloc[:0]
        with pytest.raises(ValueError, match="non-empty"):
            TemporalSplit(
                train=train,
                test=empty,
                split_date=pd.Timestamp(ADJACENT_DF_START),
            )

    def test_rejects_non_datetime_index(self) -> None:
        train = pd.DataFrame({"v": [1, 2, 3]})
        test = pd.DataFrame({"v": [4, 5, 6]})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            TemporalSplit(
                train=train,
                test=test,
                split_date=pd.Timestamp(ADJACENT_DF_START),
            )

    def test_frozen(self) -> None:
        df = make_daily_df(SMALL_DF_ROWS)
        split = TemporalSplit(
            train=df.iloc[:40],
            test=df.iloc[TRAIN_SLICE_END:],
            split_date=pd.Timestamp(df.index[TRAIN_SLICE_END]),
        )
        with pytest.raises(AttributeError):
            split.fold_index = 5  # type: ignore[misc]


class TestWalkForwardValidator:
    def test_correct_number_of_splits(self) -> None:
        df = make_daily_df(WF_LARGE_DF_ROWS)
        validator = WalkForwardValidator(
            n_splits=WF_DEFAULT_N_SPLITS, test_size=WF_DEFAULT_TEST_SIZE, gap=WF_DEFAULT_GAP
        )
        splits = list(validator.split(df))
        assert len(splits) == WF_DEFAULT_N_SPLITS

    def test_splits_have_correct_fold_indices(self) -> None:
        df = make_daily_df(WF_LARGE_DF_ROWS)
        validator = WalkForwardValidator(
            n_splits=WF_THREE_SPLITS, test_size=WF_THREE_TEST_SIZE, gap=WF_DEFAULT_GAP
        )
        splits = list(validator.split(df))
        assert [s.fold_index for s in splits] == list(range(WF_THREE_SPLITS))

    def test_test_size_is_correct(self) -> None:
        df = make_daily_df(WF_LARGE_DF_ROWS)
        validator = WalkForwardValidator(
            n_splits=WF_DEFAULT_N_SPLITS, test_size=WF_DEFAULT_TEST_SIZE, gap=WF_DEFAULT_GAP
        )
        splits = list(validator.split(df))
        for split in splits:
            assert len(split.test) == WF_DEFAULT_TEST_SIZE

    def test_gap_is_respected(self) -> None:
        df = make_daily_df(WF_LARGE_DF_ROWS)
        validator = WalkForwardValidator(
            n_splits=WF_DEFAULT_N_SPLITS, test_size=WF_DEFAULT_TEST_SIZE, gap=WF_LARGE_GAP
        )
        splits = list(validator.split(df))
        for split in splits:
            train_end_idx = df.index.get_loc(split.train.index[-1])
            test_start_idx = df.index.get_loc(split.test.index[0])
            assert test_start_idx - train_end_idx > WF_LARGE_GAP  # type: ignore[operator]

    def test_expanding_window_grows(self) -> None:
        df = make_daily_df(WF_LARGE_DF_ROWS)
        validator = WalkForwardValidator(
            n_splits=WF_DEFAULT_N_SPLITS,
            test_size=WF_DEFAULT_TEST_SIZE,
            gap=WF_DEFAULT_GAP,
            expanding=True,
        )
        splits = list(validator.split(df))
        train_sizes = [len(s.train) for s in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_no_temporal_leakage(self) -> None:
        """Every split's train max < test min (guaranteed by TemporalSplit)."""

        df = make_daily_df(WF_LARGE_DF_ROWS)
        validator = WalkForwardValidator(
            n_splits=WF_DEFAULT_N_SPLITS, test_size=WF_DEFAULT_TEST_SIZE, gap=WF_DEFAULT_GAP
        )
        splits = list(validator.split(df))
        for split in splits:
            assert split.train.index.max() < split.test.index.min()

    def test_insufficient_data_raises(self) -> None:
        df = make_daily_df(WF_INSUFFICIENT_DF_ROWS)
        validator = WalkForwardValidator(
            n_splits=WF_DEFAULT_N_SPLITS, test_size=WF_DEFAULT_TEST_SIZE, gap=WF_DEFAULT_GAP
        )
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
        df = make_daily_df(WF_SINGLE_SPLIT_DF_ROWS)
        validator = WalkForwardValidator(
            n_splits=1, test_size=WF_SINGLE_TEST_SIZE, gap=WF_DEFAULT_GAP
        )
        splits = list(validator.split(df))
        assert len(splits) == 1
        assert len(splits[0].test) == WF_SINGLE_TEST_SIZE


class TestPurgedGroupTimeSeriesSplit:
    def test_correct_number_of_splits(self) -> None:
        df = make_daily_df(PG_DF_ROWS)
        splitter = PurgedGroupTimeSeriesSplit(
            n_groups=PG_DEFAULT_N_GROUPS, embargo_pct=PG_SMALL_EMBARGO_PCT
        )
        splits = list(splitter.split(df))
        assert len(splits) == splitter.n_folds

    def test_embargo_removes_boundary_data(self) -> None:
        df = make_daily_df(PG_DF_ROWS)
        splitter = PurgedGroupTimeSeriesSplit(
            n_groups=PG_DEFAULT_N_GROUPS, embargo_pct=PG_MEDIUM_EMBARGO_PCT
        )
        splits = list(splitter.split(df))
        for split in splits:
            assert split.train.index.max() < split.test.index.min()

    def test_no_overlap(self) -> None:
        df = make_daily_df(PG_OVERLAP_DF_ROWS)
        splitter = PurgedGroupTimeSeriesSplit(
            n_groups=PG_DEFAULT_N_GROUPS, embargo_pct=PG_SMALL_EMBARGO_PCT
        )
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
        df = make_daily_df(PG_TINY_DF_ROWS)
        splitter = PurgedGroupTimeSeriesSplit(
            n_groups=PG_TWO_GROUPS, embargo_pct=PG_AGGRESSIVE_EMBARGO_PCT
        )
        with pytest.warns(UserWarning, match="skipped"):
            splits = list(splitter.split(df))
        assert len(splits) < splitter.n_folds

    def test_too_few_rows(self) -> None:
        df = make_daily_df(PG_TINIEST_DF_ROWS)
        splitter = PurgedGroupTimeSeriesSplit(n_groups=PG_LARGE_N_GROUPS)
        with pytest.raises(ValueError, match="too few"):
            list(splitter.split(df))


# Fixture sizes chosen so (N, pct) arithmetic is exact: 100 * 0.85 = 85 with no
# float rounding noise.
HB_DF_ROWS = 100
HB_PCT = 0.15
HB_CUTOFF = 85
HB_TINY_DF_ROWS = 10
HB_TINY_PCT_ALL = 0.99
# Pct so small that (1 - pct) rounds to exactly 1.0 in double precision, so
# int(n * (1-pct)) == n. Exercises the sub-per-bar-fraction edge case.
HB_FLOAT_EPS_PCT = 1e-20


class TestResolveHoldoutBoundary:
    """Exercises tripwire #2 of the holdout contract: boundary resolution.

    The resolver is the single canonical place that turns the config's knobs
    (pct or pinned timestamp) into the absolute timestamp the runner uses
    to slice. Every failure mode documented on the helper is covered here.
    """

    def test_returns_none_when_neither_knob_set(self) -> None:
        df = make_daily_df(HB_DF_ROWS)
        assert resolve_holdout_boundary(df) is None

    def test_pct_returns_cutoff_index_timestamp(self) -> None:
        df = make_daily_df(HB_DF_ROWS)
        boundary = resolve_holdout_boundary(df, holdout_pct=HB_PCT)
        assert boundary == df.index[HB_CUTOFF]

    def test_pct_split_dev_and_holdout_are_strictly_temporal(self) -> None:
        """The returned boundary, used as documented, yields a clean split."""

        df = make_daily_df(HB_DF_ROWS)
        boundary = resolve_holdout_boundary(df, holdout_pct=HB_PCT)
        assert boundary is not None
        dev = df[df.index < boundary]
        holdout = df[df.index >= boundary]
        assert len(dev) + len(holdout) == len(df)
        assert dev.index.max() < holdout.index.min()

    def test_pct_composes_cleanly_with_temporal_split(self) -> None:
        """Tripwire #3: TemporalSplit accepts the resolved boundary."""

        df = make_daily_df(HB_DF_ROWS)
        boundary = resolve_holdout_boundary(df, holdout_pct=HB_PCT)
        assert boundary is not None
        split = TemporalSplit(
            train=df[df.index < boundary],
            test=df[df.index >= boundary],
            split_date=boundary,
        )
        assert len(split.train) == HB_CUTOFF
        assert len(split.test) == HB_DF_ROWS - HB_CUTOFF

    def test_pinned_timestamp_returned_when_in_df(self) -> None:
        df = make_daily_df(HB_DF_ROWS)
        pinned = df.index[HB_CUTOFF]
        boundary = resolve_holdout_boundary(df, holdout_start=pinned)
        assert boundary == pinned

    def test_pinned_timestamp_not_in_df_raises_leakage(self) -> None:
        """Tripwire #2: data drift detection.

        A pinned timestamp that is no longer present in the fetched data
        means the vendor adjusted / added / removed a bar since the
        boundary was recorded in a manifest. Returning a nearest-neighbour
        would silently shift bars across the dev / holdout line — the
        exact leakage vector we're defending against.
        """

        df = make_daily_df(HB_DF_ROWS)
        phantom = pd.Timestamp(df.index[0]) - pd.Timedelta(days=1)
        with pytest.raises(LeakageError, match="not present in the fetched data"):
            resolve_holdout_boundary(df, holdout_start=phantom)

    def test_both_knobs_set_raises(self) -> None:
        """Defense in depth: even if config validation is bypassed, the
        helper refuses ambiguous input."""

        df = make_daily_df(HB_DF_ROWS)
        with pytest.raises(ValueError, match="at most one of holdout_pct / holdout_start"):
            resolve_holdout_boundary(
                df, holdout_pct=HB_PCT, holdout_start=pd.Timestamp(df.index[HB_CUTOFF])
            )

    def test_pct_too_large_empties_dev_raises(self) -> None:
        df = make_daily_df(HB_TINY_DF_ROWS)
        with pytest.raises(ValueError, match="empty dev region"):
            resolve_holdout_boundary(df, holdout_pct=HB_TINY_PCT_ALL)

    def test_pct_too_small_empties_holdout_raises(self) -> None:
        """Sub-per-bar fractions round to cutoff == len(df)."""

        df = make_daily_df(HB_TINY_DF_ROWS)
        with pytest.raises(ValueError, match="empty holdout region"):
            resolve_holdout_boundary(df, holdout_pct=HB_FLOAT_EPS_PCT)

    def test_non_datetime_index_raises(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            resolve_holdout_boundary(df, holdout_pct=HB_PCT)

    def test_pinned_timestamp_resolves_same_boundary_as_matching_pct(self) -> None:
        """Two runs of the SAME boundary — one via pct, one via pinned timestamp
        derived from the pct's first run — yield identical results. This is
        how the manifest round-trip is supposed to work in practice: dev run
        records the derived timestamp, holdout eval reads it back pinned."""

        df = make_daily_df(HB_DF_ROWS)
        via_pct = resolve_holdout_boundary(df, holdout_pct=HB_PCT)
        assert via_pct is not None
        via_pinned = resolve_holdout_boundary(df, holdout_start=via_pct)
        assert via_pct == via_pinned
