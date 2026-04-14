"""Tests for TemporalTripleSplit and TrainingMetadata."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.temporal import TemporalTripleSplit, TrainingMetadata
from src.core.types import Interval
from tests.conftest import make_daily_df


class TestTemporalTripleSplit:
    def test_valid_split(self) -> None:
        df = make_daily_df(200)
        split = TemporalTripleSplit.from_dataframe(df, val_pct=0.15, holdout_pct=0.15, gap=5)
        assert len(split.train) > 0
        assert len(split.validation) > 0
        assert len(split.holdout) > 0

    def test_temporal_ordering(self) -> None:
        df = make_daily_df(200)
        split = TemporalTripleSplit.from_dataframe(df, val_pct=0.15, holdout_pct=0.15, gap=5)
        assert split.train.index.max() < split.validation.index.min()
        assert split.validation.index.max() < split.holdout.index.min()

    def test_gaps_respected(self) -> None:
        df = make_daily_df(200)
        gap = 5
        split = TemporalTripleSplit.from_dataframe(df, gap=gap)
        train_end_loc = df.index.get_loc(split.train.index[-1])
        val_start_loc = df.index.get_loc(split.validation.index[0])
        val_end_loc = df.index.get_loc(split.validation.index[-1])
        holdout_start_loc = df.index.get_loc(split.holdout.index[0])
        assert val_start_loc - train_end_loc > gap  # type: ignore[operator]
        assert holdout_start_loc - val_end_loc > gap  # type: ignore[operator]

    def test_proportions_approximate(self) -> None:
        df = make_daily_df(500)
        split = TemporalTripleSplit.from_dataframe(df, val_pct=0.15, holdout_pct=0.15, gap=5)
        total = len(split.train) + len(split.validation) + len(split.holdout)
        # Proportions are approximate due to gaps
        assert len(split.holdout) == int(500 * 0.15)
        assert len(split.validation) == int(500 * 0.15)
        assert total < 500  # gaps eat some rows

    def test_overlap_train_val_raises(self) -> None:
        df = make_daily_df(100)
        with pytest.raises(LeakageError, match="overlaps"):
            TemporalTripleSplit(
                train=df.iloc[:60],
                validation=df.iloc[50:80],
                holdout=df.iloc[85:],
            )

    def test_overlap_val_holdout_raises(self) -> None:
        df = make_daily_df(100)
        with pytest.raises(LeakageError, match="overlaps"):
            TemporalTripleSplit(
                train=df.iloc[:30],
                validation=df.iloc[40:70],
                holdout=df.iloc[65:],
            )

    def test_empty_region_raises(self) -> None:
        df = make_daily_df(100)
        with pytest.raises(ValueError, match="non-empty"):
            TemporalTripleSplit(
                train=df.iloc[:0],
                validation=df.iloc[40:60],
                holdout=df.iloc[70:],
            )

    def test_non_datetime_index_raises(self) -> None:
        plain = pd.DataFrame({"v": range(50)})
        df = make_daily_df(50)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            TemporalTripleSplit(train=plain, validation=df.iloc[20:35], holdout=df.iloc[40:])

    def test_from_dataframe_too_short_raises(self) -> None:
        df = make_daily_df(10)
        with pytest.raises(ValueError, match="required"):
            TemporalTripleSplit.from_dataframe(df, val_pct=0.15, holdout_pct=0.15, gap=5)

    def test_from_dataframe_gap_zero(self) -> None:
        df = make_daily_df(200)
        split = TemporalTripleSplit.from_dataframe(df, gap=0)
        assert len(split.train) > 0
        assert len(split.validation) > 0
        assert len(split.holdout) > 0
        # With gap=0, regions are still non-overlapping (adjacent is ok)
        assert split.train.index.max() < split.validation.index.min()

    def test_from_dataframe_invalid_val_pct(self) -> None:
        df = make_daily_df(200)
        with pytest.raises(ValueError, match="val_pct"):
            TemporalTripleSplit.from_dataframe(df, val_pct=0.0)
        with pytest.raises(ValueError, match="val_pct"):
            TemporalTripleSplit.from_dataframe(df, val_pct=1.0)

    def test_from_dataframe_invalid_holdout_pct(self) -> None:
        df = make_daily_df(200)
        with pytest.raises(ValueError, match="holdout_pct"):
            TemporalTripleSplit.from_dataframe(df, holdout_pct=-0.1)

    def test_from_dataframe_pct_sum_too_large(self) -> None:
        df = make_daily_df(200)
        with pytest.raises(ValueError, match="val_pct \\+ holdout_pct"):
            TemporalTripleSplit.from_dataframe(df, val_pct=0.6, holdout_pct=0.5)

    def test_frozen(self) -> None:
        df = make_daily_df(200)
        split = TemporalTripleSplit.from_dataframe(df)
        with pytest.raises(AttributeError):
            split.train = df  # type: ignore[misc]


class TestTrainingMetadata:
    @pytest.fixture()
    def sample_metadata(self) -> TrainingMetadata:
        return TrainingMetadata(
            train_start=pd.Timestamp("2020-01-02"),
            train_end=pd.Timestamp("2023-06-30"),
            n_train_samples=900,
            fit_timestamp=pd.Timestamp("2024-01-01 12:00:00"),
            interval=Interval.DAILY,
            feature_columns=("close", "volume", "return_1d"),
        )

    def test_validate_no_overlap_raises_on_overlap(self, sample_metadata: TrainingMetadata) -> None:
        overlapping = make_daily_df(50, start="2023-06-01")
        with pytest.raises(LeakageError, match="data leakage"):
            sample_metadata.validate_no_overlap(overlapping)

    def test_validate_no_overlap_passes_on_future_data(
        self, sample_metadata: TrainingMetadata
    ) -> None:
        future = make_daily_df(50, start="2023-07-02")
        sample_metadata.validate_no_overlap(future)  # should not raise

    def test_validate_no_overlap_boundary(self, sample_metadata: TrainingMetadata) -> None:
        """Eval data starting exactly at train_end should be rejected."""
        boundary = make_daily_df(10, start="2023-06-30")
        with pytest.raises(LeakageError):
            sample_metadata.validate_no_overlap(boundary)

    def test_validate_no_overlap_non_datetime_index(
        self, sample_metadata: TrainingMetadata
    ) -> None:
        plain = pd.DataFrame({"v": range(10)})
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
