"""Tests for TemporalDataset."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.models.dataset import TemporalDataset

DF_DEFAULT_ROW_COUNT = 20
DF_START_DATE = "2024-01-01"
FEATURE_B_SCALE = 0.1
TARGET_SCALE = 2.0
DEFAULT_FEATURE_COLUMNS = ["feature_a", "feature_b"]

SMALL_LOOKBACK = 3
DEFAULT_LOOKBACK = 5

SMALL_ROW_COUNT = 10
TOO_SMALL_ROW_COUNT = 5


def _make_df(n: int = DF_DEFAULT_ROW_COUNT) -> pd.DataFrame:
    """Create a simple DataFrame for dataset testing."""

    idx = pd.bdate_range(DF_START_DATE, periods=n, freq="B")
    return pd.DataFrame(
        {
            "feature_a": list(range(n)),
            "feature_b": [x * FEATURE_B_SCALE for x in range(n)],
            "target": [x * TARGET_SCALE for x in range(n)],
        },
        index=idx,
    )


class TestTemporalDataset:
    def test_basic_creation(self) -> None:
        df = _make_df(DF_DEFAULT_ROW_COUNT)
        ds = TemporalDataset(
            df,
            target_column="target",
            lookback_window=DEFAULT_LOOKBACK,
            feature_columns=DEFAULT_FEATURE_COLUMNS,
        )
        assert len(ds) == DF_DEFAULT_ROW_COUNT - DEFAULT_LOOKBACK

    def test_getitem_shapes(self) -> None:
        df = _make_df(DF_DEFAULT_ROW_COUNT)
        ds = TemporalDataset(
            df,
            target_column="target",
            lookback_window=DEFAULT_LOOKBACK,
            feature_columns=["feature_a", "feature_b"],
        )
        features, target = ds[0]
        assert features.shape == (DEFAULT_LOOKBACK, 2)
        assert target.shape == torch.Size([])

    def test_getitem_values(self) -> None:
        """Verify features are from [idx, idx+lookback) and target is at idx+lookback."""

        df = _make_df(DF_DEFAULT_ROW_COUNT)
        ds = TemporalDataset(
            df,
            target_column="target",
            lookback_window=SMALL_LOOKBACK,
            feature_columns=["feature_a"],
        )
        features, target = ds[0]
        assert features.tolist() == [[0.0], [1.0], [2.0]]
        assert target.item() == SMALL_LOOKBACK * TARGET_SCALE

    def test_last_valid_index(self) -> None:
        """Verify the last sample doesn't go out of bounds."""

        df = _make_df(SMALL_ROW_COUNT)
        ds = TemporalDataset(
            df,
            target_column="target",
            lookback_window=DEFAULT_LOOKBACK,
            feature_columns=DEFAULT_FEATURE_COLUMNS,
        )
        assert len(ds) == SMALL_ROW_COUNT - DEFAULT_LOOKBACK
        features, target = ds[SMALL_ROW_COUNT - DEFAULT_LOOKBACK - 1]
        assert features.shape == (DEFAULT_LOOKBACK, 2)
        assert target.item() == (SMALL_ROW_COUNT - 1) * TARGET_SCALE

    def test_no_future_leakage(self) -> None:
        """Each sample's features must precede its target temporally."""

        df = _make_df(DF_DEFAULT_ROW_COUNT)
        ds = TemporalDataset(
            df,
            target_column="target",
            lookback_window=DEFAULT_LOOKBACK,
            feature_columns=["feature_a"],
        )
        for i in range(len(ds)):
            features, target = ds[i]
            last_feature_val = features[-1, 0].item()
            expected_target = (i + DEFAULT_LOOKBACK) * TARGET_SCALE
            assert target.item() == expected_target
            assert last_feature_val < i + DEFAULT_LOOKBACK

    def test_dtypes(self) -> None:
        df = _make_df(DF_DEFAULT_ROW_COUNT)
        ds = TemporalDataset(
            df,
            target_column="target",
            lookback_window=DEFAULT_LOOKBACK,
            feature_columns=DEFAULT_FEATURE_COLUMNS,
        )
        features, target = ds[0]
        assert features.dtype == torch.float32
        assert target.dtype == torch.float32

    def test_rejects_empty_feature_columns(self) -> None:
        df = _make_df(DF_DEFAULT_ROW_COUNT)
        with pytest.raises(ValueError, match="feature_columns"):
            TemporalDataset(
                df,
                target_column="target",
                lookback_window=DEFAULT_LOOKBACK,
                feature_columns=[],
            )

    def test_explicit_feature_columns(self) -> None:
        df = _make_df(DF_DEFAULT_ROW_COUNT)
        ds = TemporalDataset(
            df,
            target_column="target",
            lookback_window=DEFAULT_LOOKBACK,
            feature_columns=["feature_a"],
        )
        features, _ = ds[0]
        assert features.shape[1] == 1

    def test_rejects_non_datetime_index(self) -> None:
        df = pd.DataFrame({"a": range(SMALL_ROW_COUNT), "b": range(SMALL_ROW_COUNT)})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            TemporalDataset(
                df,
                target_column="b",
                lookback_window=SMALL_LOOKBACK,
                feature_columns=["a"],
            )

    def test_rejects_unsorted_index(self) -> None:
        idx = pd.DatetimeIndex(
            pd.date_range(DF_START_DATE, periods=SMALL_ROW_COUNT, freq="D")[::-1]
        )
        df = pd.DataFrame({"a": range(SMALL_ROW_COUNT), "b": range(SMALL_ROW_COUNT)}, index=idx)
        with pytest.raises(ValueError, match="sorted"):
            TemporalDataset(
                df,
                target_column="b",
                lookback_window=SMALL_LOOKBACK,
                feature_columns=["a"],
            )

    def test_rejects_lookback_zero(self) -> None:
        df = _make_df(SMALL_ROW_COUNT)
        with pytest.raises(ValueError, match="lookback_window must be >= 1"):
            TemporalDataset(
                df,
                target_column="target",
                lookback_window=0,
                feature_columns=DEFAULT_FEATURE_COLUMNS,
            )

    def test_rejects_missing_target_column(self) -> None:
        df = _make_df(SMALL_ROW_COUNT)
        with pytest.raises(ValueError, match="target_column"):
            TemporalDataset(
                df,
                target_column="nonexistent",
                lookback_window=SMALL_LOOKBACK,
                feature_columns=DEFAULT_FEATURE_COLUMNS,
            )

    def test_rejects_insufficient_rows(self) -> None:
        df = _make_df(TOO_SMALL_ROW_COUNT)
        with pytest.raises(ValueError, match="needs >"):
            TemporalDataset(
                df,
                target_column="target",
                lookback_window=DEFAULT_LOOKBACK,
                feature_columns=DEFAULT_FEATURE_COLUMNS,
            )
