"""Temporal-aware PyTorch dataset for time-series ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TemporalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset that respects temporal boundaries.

    Each sample is a (features, target) pair where features are a window
    of lookback_window rows and target is the value at the next timestep.
    No future data is ever included in a sample's feature window.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        lookback_window: int,
        feature_columns: list[str] | None = None,
    ) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                "DataFrame must have a DatetimeIndex; fix by setting df.index "
                "to a DatetimeIndex (or calling df.set_index('date'))."
            )
        if not df.index.is_monotonic_increasing:
            raise ValueError(
                "DataFrame must be sorted by DatetimeIndex; fix by calling "
                "df.sort_index() before constructing the dataset."
            )
        if lookback_window < 1:
            raise ValueError(
                f"lookback_window must be >= 1, got {lookback_window}; fix by "
                f"passing a window of at least 1 bar."
            )
        if target_column not in df.columns:
            raise ValueError(
                f"target_column '{target_column}' not in DataFrame; fix by "
                f"adding the target column or by passing a name that matches "
                f"one of {list(df.columns)}."
            )
        if len(df) <= lookback_window:
            raise ValueError(
                f"DataFrame has {len(df)} rows but needs > {lookback_window} "
                f"for lookback_window={lookback_window}; fix by widening the "
                f"input window or by shrinking lookback_window."
            )

        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target_column]

        self._features = torch.from_numpy(df[feature_columns].to_numpy(dtype=np.float32).copy())
        self._targets = torch.from_numpy(df[target_column].to_numpy(dtype=np.float32).copy())
        self._lookback = lookback_window

    def __len__(self) -> int:
        return len(self._features) - self._lookback

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (feature_window, target) for sample idx.

        feature_window: shape (lookback_window, n_features)
        target: scalar tensor
        """
        feature_window = self._features[idx : idx + self._lookback]
        target = self._targets[idx + self._lookback]
        return feature_window, target
