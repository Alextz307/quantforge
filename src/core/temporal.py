"""Anti-leakage temporal boundaries and walk-forward validation.

Provides TemporalSplit (frozen dataclass with overlap validation),
TemporalTripleSplit (train/validation/holdout with embargo gaps),
TrainingMetadata (immutable record of model training period),
WalkForwardValidator (expanding/sliding window splits), and
PurgedGroupTimeSeriesSplit (embargo-based purged CV).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.core.exceptions import LeakageError

if TYPE_CHECKING:
    from src.core.types import Interval


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


@dataclass(frozen=True)
class TemporalTripleSplit:
    """Three-way temporal split: train -> validation -> holdout.

    Provides anti-leakage guarantees across three temporal regions:
    - train: Model fitting and walk-forward development
    - validation: Hyperparameter tuning, overfitting checks, model comparison
    - holdout: Final evaluation ONLY — never touched during development
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame

    def __post_init__(self) -> None:
        for name, part in [
            ("train", self.train),
            ("validation", self.validation),
            ("holdout", self.holdout),
        ]:
            if not isinstance(part.index, pd.DatetimeIndex):
                raise TypeError(f"{name} DataFrame must have a DatetimeIndex")
            if len(part) == 0:
                raise ValueError(f"{name} DataFrame must be non-empty")

        train_max = self.train.index.max()
        val_min = self.validation.index.min()
        if train_max >= val_min:
            raise LeakageError(
                f"Train end {train_max} overlaps with validation start {val_min}. "
                f"Training data must strictly precede validation data."
            )

        val_max = self.validation.index.max()
        holdout_min = self.holdout.index.min()
        if val_max >= holdout_min:
            raise LeakageError(
                f"Validation end {val_max} overlaps with holdout start {holdout_min}. "
                f"Validation data must strictly precede holdout data."
            )

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        val_pct: float = 0.15,
        holdout_pct: float = 0.15,
        gap: int = 5,
    ) -> TemporalTripleSplit:
        """Split a DataFrame into train/validation/holdout with embargo gaps.

        Args:
            df: Full dataset with DatetimeIndex, sorted chronologically.
            val_pct: Fraction of data for validation (default 15%).
            holdout_pct: Fraction of data for holdout (default 15%).
            gap: Embargo bars between each region (default 5).
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex")
        if not 0.0 < val_pct < 1.0:
            raise ValueError(f"val_pct must be in (0, 1), got {val_pct}")
        if not 0.0 < holdout_pct < 1.0:
            raise ValueError(f"holdout_pct must be in (0, 1), got {holdout_pct}")
        if val_pct + holdout_pct >= 1.0:
            raise ValueError(f"val_pct + holdout_pct must be < 1.0, got {val_pct + holdout_pct}")

        n = len(df)
        holdout_size = max(1, int(n * holdout_pct))
        val_size = max(1, int(n * val_pct))
        min_required = val_size + holdout_size + 2 * gap + 1
        if n < min_required:
            raise ValueError(
                f"DataFrame has {n} rows but at least {min_required} are required "
                f"for val_pct={val_pct}, holdout_pct={holdout_pct}, gap={gap}"
            )

        holdout_start = n - holdout_size
        val_end = holdout_start - gap
        val_start = val_end - val_size
        train_end = val_start - gap

        train = df.iloc[:train_end]
        validation = df.iloc[val_start:val_end]
        holdout = df.iloc[holdout_start:]

        return TemporalTripleSplit(train=train, validation=validation, holdout=holdout)


@dataclass(frozen=True)
class TrainingMetadata:
    """Immutable record of what a model saw during training.

    Stored by every model/strategy after fit()/train(). Used at evaluation
    time to verify eval data doesn't overlap with the training period.
    Serializable via to_dict()/from_dict() for model persistence.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_train_samples: int
    fit_timestamp: pd.Timestamp
    interval: Interval
    feature_columns: tuple[str, ...]

    def validate_no_overlap(self, eval_data: pd.DataFrame) -> None:
        """Raise LeakageError if eval data overlaps training period.

        Assumes eval_data has a chronologically sorted DatetimeIndex
        (enforced throughout the framework).
        """
        if not isinstance(eval_data.index, pd.DatetimeIndex):
            raise TypeError("eval_data must have a DatetimeIndex")
        eval_start: pd.Timestamp = eval_data.index[0]
        if eval_start <= self.train_end:
            raise LeakageError(
                f"Evaluation data starts at {eval_start} but model "
                f"was trained through {self.train_end}. "
                f"This would constitute data leakage."
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize to JSON-friendly dict. Timestamps become ISO strings."""
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "n_train_samples": self.n_train_samples,
            "fit_timestamp": self.fit_timestamp.isoformat(),
            "interval": self.interval.value,
            "feature_columns": list(self.feature_columns),
        }

    @staticmethod
    def from_fit(
        train_data: pd.DataFrame,
        interval: Interval,
        feature_columns: tuple[str, ...],
    ) -> TrainingMetadata:
        """Create metadata from a completed fit() call."""
        return TrainingMetadata(
            train_start=pd.Timestamp(train_data.index[0]),
            train_end=pd.Timestamp(train_data.index[-1]),
            n_train_samples=len(train_data),
            fit_timestamp=pd.Timestamp.now(),
            interval=interval,
            feature_columns=feature_columns,
        )

    def extend(
        self,
        new_start: pd.Timestamp,
        new_end: pd.Timestamp,
        additional_samples: int,
    ) -> TrainingMetadata:
        """Return a copy with ``train_end`` advanced and ``n_train_samples`` incremented.

        ``fit_timestamp`` is preserved — it records when the model was *first*
        fit, not when it was last touched. ``train_start``, ``interval``, and
        ``feature_columns`` are also preserved (update() extends the training
        window forward, it never rewinds or changes schema).

        Raises ``LeakageError`` if ``new_start <= train_end`` OR
        ``new_end <= train_end`` — the new window must start strictly after
        the existing training window AND end strictly after it. The first
        mirrors ``validate_no_overlap``'s invariant at extend-time so a caller
        passing overlapping ``new_data`` double-counts nothing. The second
        catches a non-monotonic ``new_data`` whose last bar precedes the
        training cutoff (zero forward progress — a subsequent
        ``validate_no_overlap`` would still compare against a stale window).

        Most callers should use :meth:`extend_from` instead — it reads the
        ``new_start`` / ``new_end`` / ``additional_samples`` off a DataFrame
        in one call and avoids the ``pd.Timestamp(...)`` coercions sprinkled
        across every ``update()`` override.
        """
        if new_start <= self.train_end:
            raise LeakageError(
                f"extend() requires new_start > train_end; got new_start={new_start}, "
                f"train_end={self.train_end}. Overlapping new_data would double-count "
                f"rows already in the training window."
            )
        if new_end <= self.train_end:
            raise LeakageError(
                f"extend() requires new_end > train_end; got new_end={new_end}, "
                f"train_end={self.train_end}. Zero forward progress leaves "
                f"``validate_no_overlap`` checking against a stale window."
            )
        if additional_samples < 1:
            raise ValueError(f"extend() requires additional_samples >= 1, got {additional_samples}")
        return TrainingMetadata(
            train_start=self.train_start,
            train_end=new_end,
            n_train_samples=self.n_train_samples + additional_samples,
            fit_timestamp=self.fit_timestamp,
            interval=self.interval,
            feature_columns=self.feature_columns,
        )

    def extend_from(self, new_data: pd.DataFrame) -> TrainingMetadata:
        """Convenience wrapper: read ``(new_start, new_end, additional_samples)``
        straight off ``new_data`` and delegate to :meth:`extend`.

        Collapses the 4-line boilerplate every ``update()`` override would
        otherwise repeat into a single call. Requires a non-empty DataFrame
        with a ``DatetimeIndex`` — both invariants are enforced elsewhere
        in the framework, but they're checked here too so a misuse surfaces
        with a pointed error rather than a confusing ``LeakageError``.
        """
        if not isinstance(new_data.index, pd.DatetimeIndex):
            raise TypeError("extend_from() requires a DataFrame with a DatetimeIndex")
        if len(new_data) == 0:
            raise ValueError("extend_from() requires a non-empty DataFrame")
        return self.extend(
            new_start=pd.Timestamp(new_data.index[0]),
            new_end=pd.Timestamp(new_data.index[-1]),
            additional_samples=len(new_data),
        )

    @staticmethod
    def from_dict(d: dict[str, object]) -> TrainingMetadata:
        """Deserialize from dict. Used when loading a saved model."""
        from src.core.types import Interval

        raw_cols = d["feature_columns"]
        if not isinstance(raw_cols, list):
            raise TypeError(f"feature_columns must be a list, got {type(raw_cols).__name__}")
        return TrainingMetadata(
            train_start=pd.Timestamp(str(d["train_start"])),
            train_end=pd.Timestamp(str(d["train_end"])),
            n_train_samples=int(str(d["n_train_samples"])),
            fit_timestamp=pd.Timestamp(str(d["fit_timestamp"])),
            interval=Interval(str(d["interval"])),
            feature_columns=tuple(raw_cols),
        )


def _snap_train_end_backward(dates: pd.DatetimeIndex, train_end: int) -> int:
    """Snap ``train_end`` backward so ``iloc[:train_end]`` ends at a day close.

    ``train_end`` is exclusive. On return, bar ``train_end - 1`` is the last
    bar of its calendar date and bar ``train_end`` (if present) is the first
    bar of a later date. Already-at-boundary indices are returned unchanged.
    """
    if train_end <= 0 or train_end >= len(dates):
        return train_end
    prefix = dates[:train_end].values
    shifted = dates[1 : train_end + 1].values
    boundaries = np.flatnonzero(prefix != shifted)
    if boundaries.size == 0:
        return 0
    return int(boundaries[-1]) + 1


def _first_bar_after_gap_days(dates: pd.DatetimeIndex, train_end: int, gap_days: int) -> int:
    """First bar strictly ``gap_days`` trading days past the last training date.

    ``gap_days=0`` returns the first bar of the next distinct date;
    ``gap_days=k`` skips ``k`` distinct dates of embargo. Trading days are the
    distinct normalized dates observed — holiday gaps are handled naturally.
    """
    if train_end < 1:
        raise ValueError(f"_first_bar_after_gap_days requires train_end >= 1, got {train_end}.")
    prev = dates[train_end - 1 : -1].values
    tail = dates[train_end:].values
    new_date_offsets = np.flatnonzero(tail != prev)
    target = gap_days + 1
    if new_date_offsets.size < target:
        raise ValueError(
            f"snap_to_day: only {new_date_offsets.size} distinct dates remain after "
            f"{dates[train_end - 1]}; gap={gap_days} cannot be honored."
        )
    return train_end + int(new_date_offsets[target - 1])


class WalkForwardValidator:
    """Expanding-window walk-forward validation splitter.

    Generates temporal train/test splits where:
    - Training window expands (or slides) forward through time
    - A gap (embargo) separates train from test to prevent leakage
    - Test window has a fixed size

    When ``snap_to_day=True``, every fold's train ends at a day close and
    test starts at the first bar of a new calendar date. The ``gap`` argument
    is then interpreted as **trading days** of embargo (distinct dates
    observed in ``df``), not bars. This honours the framework's intraday
    day-boundary rule for hourly/minute data.
    """

    def __init__(
        self,
        n_splits: int = 4,
        test_size: int = 252,
        gap: int = 5,
        expanding: bool = True,
        snap_to_day: bool = False,
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
        self.snap_to_day = snap_to_day

    def split(self, df: pd.DataFrame) -> Iterator[TemporalSplit]:
        """Generate expanding-window temporal splits.

        The splits are computed from the end of the data backward:
        - The last `test_size` bars form the last test set
        - Working backward, each fold's test set is `test_size` bars
        - Training data extends from the start (expanding) or slides

        When ``snap_to_day=True``, ``train_end`` is snapped backward to a
        day boundary and ``test_start`` is pushed forward by ``gap`` trading
        days. ``test_size`` remains a bar count.

        Args:
            df: DataFrame with DatetimeIndex, sorted chronologically.

        Yields:
            TemporalSplit for each fold.

        Raises:
            ValueError: If the DataFrame is too short for the requested splits,
                or if ``snap_to_day=True`` and fewer than 2 distinct dates are
                present.
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

        normalized_dates: pd.DatetimeIndex | None = None
        if self.snap_to_day:
            assert isinstance(df.index, pd.DatetimeIndex)
            normalized_dates = df.index.normalize()
            if normalized_dates.nunique() < 2:
                raise ValueError(
                    "snap_to_day=True requires at least 2 distinct dates in the DataFrame."
                )

        first_train_end = n - (self.n_splits - 1) * self.test_size - self.test_size - self.gap

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
                train_start = max(0, train_end - first_train_end)

            if self.snap_to_day:
                assert normalized_dates is not None
                train_end = _snap_train_end_backward(normalized_dates, train_end)
                # Sliding: recompute train_start from snapped train_end so the
                # window size stays invariant across folds.
                if not self.expanding:
                    train_start = max(0, train_end - first_train_end)
                if train_end <= train_start:
                    raise ValueError(
                        f"snap_to_day: fold {i} has empty train window after "
                        f"snapping to day boundary."
                    )
                test_start = _first_bar_after_gap_days(normalized_dates, train_end, self.gap)
                test_end = test_start + self.test_size
                if test_end > n:
                    raise ValueError(
                        f"snap_to_day: fold {i} would need {test_end - n} bars past "
                        f"end-of-frame for a full test window of {self.test_size}; "
                        f"provide more data or reduce test_size / gap."
                    )

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
