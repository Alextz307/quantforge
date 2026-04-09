"""Tests for TemporalDataset."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.models.dataset import TemporalDataset


def _make_df(n: int = 20) -> pd.DataFrame:
    """Create a simple DataFrame for dataset testing."""
    idx = pd.bdate_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "feature_a": list(range(n)),
            "feature_b": [x * 0.1 for x in range(n)],
            "target": [x * 2.0 for x in range(n)],
        },
        index=idx,
    )


class TestTemporalDataset:
    def test_basic_creation(self) -> None:
        df = _make_df(20)
        ds = TemporalDataset(df, target_column="target", lookback_window=5)
        assert len(ds) == 15  # 20 - 5

    def test_getitem_shapes(self) -> None:
        df = _make_df(20)
        ds = TemporalDataset(
            df,
            target_column="target",
            lookback_window=5,
            feature_columns=["feature_a", "feature_b"],
        )
        features, target = ds[0]
        assert features.shape == (5, 2)  # (lookback, n_features)
        assert target.shape == torch.Size([])  # scalar

    def test_getitem_values(self) -> None:
        """Verify features are from [idx, idx+lookback) and target is at idx+lookback."""
        df = _make_df(20)
        ds = TemporalDataset(
            df, target_column="target", lookback_window=3, feature_columns=["feature_a"]
        )
        features, target = ds[0]
        # Features should be rows 0, 1, 2 of feature_a
        assert features.tolist() == [[0.0], [1.0], [2.0]]
        # Target should be row 3 of target column (3 * 2.0 = 6.0)
        assert target.item() == 6.0

    def test_last_valid_index(self) -> None:
        """Verify the last sample doesn't go out of bounds."""
        df = _make_df(10)
        ds = TemporalDataset(df, target_column="target", lookback_window=5)
        assert len(ds) == 5
        # Last valid index
        features, target = ds[4]
        assert features.shape == (5, 2)
        # Target should be row 9 (last row): 9 * 2.0 = 18.0
        assert target.item() == 18.0

    def test_no_future_leakage(self) -> None:
        """Each sample's features must precede its target temporally."""
        df = _make_df(20)
        ds = TemporalDataset(
            df, target_column="target", lookback_window=5, feature_columns=["feature_a"]
        )
        for i in range(len(ds)):
            features, target = ds[i]
            # Last feature value (feature_a = row index) must be < target row index
            last_feature_val = features[-1, 0].item()
            # target is at index i + lookback, target value = (i + 5) * 2.0
            expected_target = (i + 5) * 2.0
            assert target.item() == expected_target
            assert last_feature_val < i + 5  # feature row < target row

    def test_dtypes(self) -> None:
        df = _make_df(20)
        ds = TemporalDataset(df, target_column="target", lookback_window=5)
        features, target = ds[0]
        assert features.dtype == torch.float32
        assert target.dtype == torch.float32

    def test_auto_feature_columns(self) -> None:
        """When feature_columns is None, all non-target columns are used."""
        df = _make_df(20)
        ds = TemporalDataset(df, target_column="target", lookback_window=5)
        features, _ = ds[0]
        assert features.shape[1] == 2  # feature_a, feature_b

    def test_explicit_feature_columns(self) -> None:
        df = _make_df(20)
        ds = TemporalDataset(
            df, target_column="target", lookback_window=5, feature_columns=["feature_a"]
        )
        features, _ = ds[0]
        assert features.shape[1] == 1

    def test_rejects_non_datetime_index(self) -> None:
        df = pd.DataFrame({"a": range(10), "b": range(10)})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            TemporalDataset(df, target_column="b", lookback_window=3)

    def test_rejects_unsorted_index(self) -> None:
        idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=10, freq="D")[::-1])
        df = pd.DataFrame({"a": range(10), "b": range(10)}, index=idx)
        with pytest.raises(ValueError, match="sorted"):
            TemporalDataset(df, target_column="b", lookback_window=3)

    def test_rejects_lookback_zero(self) -> None:
        df = _make_df(10)
        with pytest.raises(ValueError, match="lookback_window must be >= 1"):
            TemporalDataset(df, target_column="target", lookback_window=0)

    def test_rejects_missing_target_column(self) -> None:
        df = _make_df(10)
        with pytest.raises(ValueError, match="target_column"):
            TemporalDataset(df, target_column="nonexistent", lookback_window=3)

    def test_rejects_insufficient_rows(self) -> None:
        df = _make_df(5)
        with pytest.raises(ValueError, match="needs >"):
            TemporalDataset(df, target_column="target", lookback_window=5)
