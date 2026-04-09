"""Anti-leakage temporal boundaries and walk-forward validation.

Provides TemporalSplit (frozen dataclass with overlap validation),
WalkForwardValidator (expanding/sliding window splits), and
PurgedGroupTimeSeriesSplit (embargo-based purged CV).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from src.core.exceptions import LeakageError


@dataclass(frozen=True)
class TemporalSplit:
    """A single train/test split with temporal integrity validation.

    The frozen dataclass prevents rebinding of train/test DataFrames
    after creation. The __post_init__ validates that train data
    strictly precedes test data (no temporal overlap).
    """

    train: pd.DataFrame
    test: pd.DataFrame
    split_date: pd.Timestamp
    fold_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.train.index, pd.DatetimeIndex):
            raise TypeError("train DataFrame must have a DatetimeIndex")
        if not isinstance(self.test.index, pd.DatetimeIndex):
            raise TypeError("test DataFrame must have a DatetimeIndex")
        if len(self.train) == 0 or len(self.test) == 0:
            raise ValueError("train and test DataFrames must be non-empty")

        train_max = self.train.index.max()
        test_min = self.test.index.min()
        if train_max >= test_min:
            raise LeakageError(
                f"Train end {train_max} overlaps with test start {test_min}. "
                f"Training data must strictly precede test data."
            )


class WalkForwardValidator:
    """Expanding-window walk-forward validation splitter.

    Generates temporal train/test splits where:
    - Training window expands (or slides) forward through time
    - A gap (embargo) separates train from test to prevent leakage
    - Test window has a fixed size
    """

    def __init__(
        self,
        n_splits: int = 4,
        test_size: int = 252,
        gap: int = 5,
        expanding: bool = True,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")

        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding

    def split(self, df: pd.DataFrame) -> Iterator[TemporalSplit]:
        """Generate expanding-window temporal splits.

        The splits are computed from the end of the data backward:
        - The last `test_size` bars form the last test set
        - Working backward, each fold's test set is `test_size` bars
        - Training data extends from the start (expanding) or slides

        Args:
            df: DataFrame with DatetimeIndex, sorted chronologically.

        Yields:
            TemporalSplit for each fold.

        Raises:
            ValueError: If the DataFrame is too short for the requested splits.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex")

        n = len(df)
        min_required = self.n_splits * self.test_size + self.gap + 1
        if n < min_required:
            raise ValueError(
                f"DataFrame has {n} rows but at least {min_required} are "
                f"required for {self.n_splits} splits with test_size="
                f"{self.test_size} and gap={self.gap}"
            )

        for i in range(self.n_splits):
            # Test set position: work backward from the end
            test_end = n - (self.n_splits - 1 - i) * self.test_size
            test_start = test_end - self.test_size

            # Train set ends before the gap
            train_end = test_start - self.gap

            # Train set starts at 0 (expanding) or slides
            if self.expanding:
                train_start = 0
            else:
                # Sliding window: train size equals first fold's train size
                first_train_end = (
                    n - (self.n_splits - 1) * self.test_size - self.test_size - self.gap
                )
                train_start = max(0, train_end - first_train_end)

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]
            split_date = pd.Timestamp(df.index[test_start])

            yield TemporalSplit(
                train=train_df,
                test=test_df,
                split_date=split_date,
                fold_index=i,
            )


class PurgedGroupTimeSeriesSplit:
    """Purged cross-validation with embargo periods.

    Removes a fraction of training data at the boundary to prevent
    information leakage from overlapping rolling features.
    Based on Marcos Lopez de Prado's methodology.

    Note:
        The data is divided into ``n_groups`` chronological groups.
        Group 0 is never used as a test set (no preceding training data),
        so ``split()`` yields ``n_groups - 1`` folds.
    """

    def __init__(
        self,
        n_groups: int = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {n_groups}")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct}")

        self.n_groups = n_groups
        self.embargo_pct = embargo_pct

    @property
    def n_folds(self) -> int:
        """Number of folds yielded by split() (always n_groups - 1)."""
        return self.n_groups - 1

    def split(self, df: pd.DataFrame) -> Iterator[TemporalSplit]:
        """Generate purged time-series splits with embargo.

        Each fold:
        1. Split data into ``n_groups`` groups chronologically
        2. Use one group as test set (starting from group 1)
        3. Remove ``embargo_pct`` of training data at the train/test boundary
        4. Remaining earlier data is the training set

        Args:
            df: DataFrame with DatetimeIndex.

        Yields:
            ``n_groups - 1`` TemporalSplit instances (group 0 has no training data).
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex")

        n = len(df)
        group_size = n // self.n_groups
        embargo_size = max(1, int(n * self.embargo_pct))

        if group_size < 2:
            raise ValueError(f"DataFrame has {n} rows, too few for {self.n_groups} groups")

        for i in range(1, self.n_groups):
            test_start = i * group_size
            test_end = (i + 1) * group_size if i < self.n_groups - 1 else n

            # Purge: remove embargo_size rows before the test set
            train_end = max(0, test_start - embargo_size)

            if train_end < 1:
                warnings.warn(
                    f"PurgedGroupTimeSeriesSplit: fold {i} skipped because "
                    f"embargo ({embargo_size} rows) leaves no training data. "
                    f"Reduce embargo_pct or n_groups.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            train_df = df.iloc[:train_end]
            test_df = df.iloc[test_start:test_end]
            split_date = pd.Timestamp(df.index[test_start])

            yield TemporalSplit(
                train=train_df,
                test=test_df,
                split_date=split_date,
                fold_index=i,
            )
