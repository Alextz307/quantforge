"""Tests for TemporalTripleSplit and TrainingMetadata."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.temporal import TemporalTripleSplit, TrainingMetadata
from src.core.types import Interval
from tests.conftest import make_daily_df

# TripleSplit common parameters
TRIPLE_SMALL_DF_ROWS = 100
TRIPLE_DEFAULT_DF_ROWS = 200
TRIPLE_LARGE_DF_ROWS = 500
TRIPLE_TINY_DF_ROWS = 10
DEFAULT_VAL_PCT = 0.15
DEFAULT_HOLDOUT_PCT = 0.15
DEFAULT_GAP = 5
INVALID_PCT_TOO_LARGE_VAL = 0.6
INVALID_PCT_TOO_LARGE_HOLDOUT = 0.5

# TripleSplit overlap construction (raw indices)
OVERLAP_TRAIN_TO_VAL_TRAIN_END = 60  # > validation_start to overlap train-val
OVERLAP_TRAIN_TO_VAL_VAL_START = 50
OVERLAP_TRAIN_TO_VAL_VAL_END = 80
OVERLAP_TRAIN_TO_VAL_HOLDOUT_START = 85

OVERLAP_VAL_TO_HOLDOUT_TRAIN_END = 30
OVERLAP_VAL_TO_HOLDOUT_VAL_START = 40
OVERLAP_VAL_TO_HOLDOUT_VAL_END = 70
OVERLAP_VAL_TO_HOLDOUT_HOLDOUT_START = 65  # < val_end to overlap val-holdout

EMPTY_TRAIN_VAL_START = 40
EMPTY_TRAIN_VAL_END = 60
EMPTY_TRAIN_HOLDOUT_START = 70

NON_DATETIME_PLAIN_ROWS = 50
NON_DATETIME_VAL_START = 20
NON_DATETIME_VAL_END = 35
NON_DATETIME_HOLDOUT_START = 40

# TrainingMetadata sample
META_TRAIN_START = "2020-01-02"
META_TRAIN_END = "2023-06-30"
META_FIT_TIMESTAMP = "2024-01-01 12:00:00"
META_N_SAMPLES = 900
META_FEATURE_COLUMNS = ("close", "volume", "return_1d")
META_OVERLAP_DF_ROWS = 50
META_OVERLAP_START = "2023-06-01"  # ends inside training window
META_FUTURE_START = "2023-07-02"  # strictly after train_end
META_BOUNDARY_DF_ROWS = 10
META_BOUNDARY_START = META_TRAIN_END  # exact boundary should still be rejected
META_PLAIN_ROWS = 10

# extend() new-window samples
META_EXTEND_NEW_START = "2023-07-02"  # strictly after META_TRAIN_END
META_EXTEND_NEW_END = "2023-12-29"
META_EXTEND_ADDITIONAL_SAMPLES = 120
META_EXTEND_OVERLAP_NEW_START = "2023-06-01"  # inside training window


class TestTemporalTripleSplit:
    def test_valid_split(self) -> None:
        df = make_daily_df(TRIPLE_DEFAULT_DF_ROWS)
        split = TemporalTripleSplit.from_dataframe(
            df, val_pct=DEFAULT_VAL_PCT, holdout_pct=DEFAULT_HOLDOUT_PCT, gap=DEFAULT_GAP
        )
        assert len(split.train) > 0
        assert len(split.validation) > 0
        assert len(split.holdout) > 0

    def test_temporal_ordering(self) -> None:
        df = make_daily_df(TRIPLE_DEFAULT_DF_ROWS)
        split = TemporalTripleSplit.from_dataframe(
            df, val_pct=DEFAULT_VAL_PCT, holdout_pct=DEFAULT_HOLDOUT_PCT, gap=DEFAULT_GAP
        )
        assert split.train.index.max() < split.validation.index.min()
        assert split.validation.index.max() < split.holdout.index.min()

    def test_gaps_respected(self) -> None:
        df = make_daily_df(TRIPLE_DEFAULT_DF_ROWS)
        split = TemporalTripleSplit.from_dataframe(df, gap=DEFAULT_GAP)
        train_end_loc = df.index.get_loc(split.train.index[-1])
        val_start_loc = df.index.get_loc(split.validation.index[0])
        val_end_loc = df.index.get_loc(split.validation.index[-1])
        holdout_start_loc = df.index.get_loc(split.holdout.index[0])
        assert val_start_loc - train_end_loc > DEFAULT_GAP  # type: ignore[operator]
        assert holdout_start_loc - val_end_loc > DEFAULT_GAP  # type: ignore[operator]

    def test_proportions_approximate(self) -> None:
        df = make_daily_df(TRIPLE_LARGE_DF_ROWS)
        split = TemporalTripleSplit.from_dataframe(
            df, val_pct=DEFAULT_VAL_PCT, holdout_pct=DEFAULT_HOLDOUT_PCT, gap=DEFAULT_GAP
        )
        total = len(split.train) + len(split.validation) + len(split.holdout)
        # Proportions are approximate due to gaps
        assert len(split.holdout) == int(TRIPLE_LARGE_DF_ROWS * DEFAULT_HOLDOUT_PCT)
        assert len(split.validation) == int(TRIPLE_LARGE_DF_ROWS * DEFAULT_VAL_PCT)
        assert total < TRIPLE_LARGE_DF_ROWS  # gaps eat some rows

    def test_overlap_train_val_raises(self) -> None:
        df = make_daily_df(TRIPLE_SMALL_DF_ROWS)
        with pytest.raises(LeakageError, match="overlaps"):
            TemporalTripleSplit(
                train=df.iloc[:OVERLAP_TRAIN_TO_VAL_TRAIN_END],
                validation=df.iloc[OVERLAP_TRAIN_TO_VAL_VAL_START:OVERLAP_TRAIN_TO_VAL_VAL_END],
                holdout=df.iloc[OVERLAP_TRAIN_TO_VAL_HOLDOUT_START:],
            )

    def test_overlap_val_holdout_raises(self) -> None:
        df = make_daily_df(TRIPLE_SMALL_DF_ROWS)
        with pytest.raises(LeakageError, match="overlaps"):
            TemporalTripleSplit(
                train=df.iloc[:OVERLAP_VAL_TO_HOLDOUT_TRAIN_END],
                validation=df.iloc[OVERLAP_VAL_TO_HOLDOUT_VAL_START:OVERLAP_VAL_TO_HOLDOUT_VAL_END],
                holdout=df.iloc[OVERLAP_VAL_TO_HOLDOUT_HOLDOUT_START:],
            )

    def test_empty_region_raises(self) -> None:
        df = make_daily_df(TRIPLE_SMALL_DF_ROWS)
        with pytest.raises(ValueError, match="non-empty"):
            TemporalTripleSplit(
                train=df.iloc[:0],
                validation=df.iloc[EMPTY_TRAIN_VAL_START:EMPTY_TRAIN_VAL_END],
                holdout=df.iloc[EMPTY_TRAIN_HOLDOUT_START:],
            )

    def test_non_datetime_index_raises(self) -> None:
        plain = pd.DataFrame({"v": range(NON_DATETIME_PLAIN_ROWS)})
        df = make_daily_df(NON_DATETIME_PLAIN_ROWS)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            TemporalTripleSplit(
                train=plain,
                validation=df.iloc[NON_DATETIME_VAL_START:NON_DATETIME_VAL_END],
                holdout=df.iloc[NON_DATETIME_HOLDOUT_START:],
            )

    def test_from_dataframe_too_short_raises(self) -> None:
        df = make_daily_df(TRIPLE_TINY_DF_ROWS)
        with pytest.raises(ValueError, match="required"):
            TemporalTripleSplit.from_dataframe(
                df, val_pct=DEFAULT_VAL_PCT, holdout_pct=DEFAULT_HOLDOUT_PCT, gap=DEFAULT_GAP
            )

    def test_from_dataframe_gap_zero(self) -> None:
        df = make_daily_df(TRIPLE_DEFAULT_DF_ROWS)
        split = TemporalTripleSplit.from_dataframe(df, gap=0)
        assert len(split.train) > 0
        assert len(split.validation) > 0
        assert len(split.holdout) > 0
        # With gap=0, regions are still non-overlapping (adjacent is ok)
        assert split.train.index.max() < split.validation.index.min()

    def test_from_dataframe_invalid_val_pct(self) -> None:
        df = make_daily_df(TRIPLE_DEFAULT_DF_ROWS)
        with pytest.raises(ValueError, match="val_pct"):
            TemporalTripleSplit.from_dataframe(df, val_pct=0.0)
        with pytest.raises(ValueError, match="val_pct"):
            TemporalTripleSplit.from_dataframe(df, val_pct=1.0)

    def test_from_dataframe_invalid_holdout_pct(self) -> None:
        df = make_daily_df(TRIPLE_DEFAULT_DF_ROWS)
        with pytest.raises(ValueError, match="holdout_pct"):
            TemporalTripleSplit.from_dataframe(df, holdout_pct=-0.1)

    def test_from_dataframe_pct_sum_too_large(self) -> None:
        df = make_daily_df(TRIPLE_DEFAULT_DF_ROWS)
        with pytest.raises(ValueError, match="val_pct \\+ holdout_pct"):
            TemporalTripleSplit.from_dataframe(
                df,
                val_pct=INVALID_PCT_TOO_LARGE_VAL,
                holdout_pct=INVALID_PCT_TOO_LARGE_HOLDOUT,
            )

    def test_frozen(self) -> None:
        df = make_daily_df(TRIPLE_DEFAULT_DF_ROWS)
        split = TemporalTripleSplit.from_dataframe(df)
        with pytest.raises(AttributeError):
            split.train = df  # type: ignore[misc]


class TestTrainingMetadata:
    @pytest.fixture()
    def sample_metadata(self) -> TrainingMetadata:
        return TrainingMetadata(
            train_start=pd.Timestamp(META_TRAIN_START),
            train_end=pd.Timestamp(META_TRAIN_END),
            n_train_samples=META_N_SAMPLES,
            fit_timestamp=pd.Timestamp(META_FIT_TIMESTAMP),
            interval=Interval.DAILY,
            feature_columns=META_FEATURE_COLUMNS,
        )

    def test_validate_no_overlap_raises_on_overlap(self, sample_metadata: TrainingMetadata) -> None:
        overlapping = make_daily_df(META_OVERLAP_DF_ROWS, start=META_OVERLAP_START)
        with pytest.raises(LeakageError, match="data leakage"):
            sample_metadata.validate_no_overlap(overlapping)

    def test_validate_no_overlap_passes_on_future_data(
        self, sample_metadata: TrainingMetadata
    ) -> None:
        future = make_daily_df(META_OVERLAP_DF_ROWS, start=META_FUTURE_START)
        sample_metadata.validate_no_overlap(future)  # should not raise

    def test_validate_no_overlap_boundary(self, sample_metadata: TrainingMetadata) -> None:
        """Eval data starting exactly at train_end should be rejected."""
        boundary = make_daily_df(META_BOUNDARY_DF_ROWS, start=META_BOUNDARY_START)
        with pytest.raises(LeakageError):
            sample_metadata.validate_no_overlap(boundary)

    def test_validate_no_overlap_non_datetime_index(
        self, sample_metadata: TrainingMetadata
    ) -> None:
        plain = pd.DataFrame({"v": range(META_PLAIN_ROWS)})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            sample_metadata.validate_no_overlap(plain)

    def test_frozen(self, sample_metadata: TrainingMetadata) -> None:
        with pytest.raises(AttributeError):
            sample_metadata.n_train_samples = 0  # type: ignore[misc]

    def test_to_dict_from_dict_roundtrip(self, sample_metadata: TrainingMetadata) -> None:
        d = sample_metadata.to_dict()
        restored = TrainingMetadata.from_dict(d)
        assert restored.train_start == sample_metadata.train_start
        assert restored.train_end == sample_metadata.train_end
        assert restored.n_train_samples == sample_metadata.n_train_samples
        assert restored.interval == sample_metadata.interval
        assert restored.feature_columns == sample_metadata.feature_columns

    def test_to_dict_types(self, sample_metadata: TrainingMetadata) -> None:
        d = sample_metadata.to_dict()
        assert isinstance(d["train_start"], str)
        assert isinstance(d["train_end"], str)
        assert isinstance(d["interval"], str)
        assert isinstance(d["feature_columns"], list)

    def test_extend_advances_end_and_sample_count(self, sample_metadata: TrainingMetadata) -> None:
        extended = sample_metadata.extend(
            new_start=pd.Timestamp(META_EXTEND_NEW_START),
            new_end=pd.Timestamp(META_EXTEND_NEW_END),
            additional_samples=META_EXTEND_ADDITIONAL_SAMPLES,
        )
        assert extended.train_start == sample_metadata.train_start
        assert extended.train_end == pd.Timestamp(META_EXTEND_NEW_END)
        assert (
            extended.n_train_samples
            == sample_metadata.n_train_samples + META_EXTEND_ADDITIONAL_SAMPLES
        )
        # fit_timestamp preserved — this is provenance, not "last touched".
        assert extended.fit_timestamp == sample_metadata.fit_timestamp

    def test_extend_rejects_overlapping_new_window(self, sample_metadata: TrainingMetadata) -> None:
        """``new_start`` inside the training window is a leakage vector — the
        caller would double-count rows already seen during fit()."""
        with pytest.raises(LeakageError, match="new_start > train_end"):
            sample_metadata.extend(
                new_start=pd.Timestamp(META_EXTEND_OVERLAP_NEW_START),
                new_end=pd.Timestamp(META_EXTEND_NEW_END),
                additional_samples=META_EXTEND_ADDITIONAL_SAMPLES,
            )

    def test_extend_rejects_zero_samples(self, sample_metadata: TrainingMetadata) -> None:
        with pytest.raises(ValueError, match="additional_samples >= 1"):
            sample_metadata.extend(
                new_start=pd.Timestamp(META_EXTEND_NEW_START),
                new_end=pd.Timestamp(META_EXTEND_NEW_END),
                additional_samples=0,
            )

    def test_extend_rejects_new_end_at_boundary(self, sample_metadata: TrainingMetadata) -> None:
        """``new_end == train_end`` is zero forward progress — a subsequent
        ``validate_no_overlap`` would check against a stale window. Rejected
        as ``LeakageError`` for the same reason overlapping ``new_start`` is."""
        with pytest.raises(LeakageError, match="new_end > train_end"):
            sample_metadata.extend(
                new_start=pd.Timestamp(META_EXTEND_NEW_START),
                new_end=sample_metadata.train_end,
                additional_samples=META_EXTEND_ADDITIONAL_SAMPLES,
            )

    def test_training_metadata_none_before_fit(self) -> None:
        """IPredictor.training_metadata returns None before fit()."""
        from src.models.interface import IPredictor

        class DummyPredictor(IPredictor):
            def fit(self, train_data: pd.DataFrame, target: pd.Series, **kwargs: object) -> None:
                pass

            def predict(self, data: pd.DataFrame) -> pd.Series:
                return pd.Series(dtype=float)

            def predict_single(self, recent_window: pd.DataFrame) -> float:
                return 0.0

        p = DummyPredictor()
        assert p.training_metadata is None
