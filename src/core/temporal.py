"""Anti-leakage temporal boundaries and walk-forward validation.

Provides TemporalSplit (frozen dataclass with overlap validation),
TemporalTripleSplit (train/validation/holdout with embargo gaps),
TrainingMetadata (immutable record of model training period),
WalkForwardValidator (expanding/sliding window splits),
PurgedGroupTimeSeriesSplit (embargo-based purged CV), and
resolve_holdout_boundary (dev / holdout split timestamp resolver).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass, replace
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
class TrackedMetadata:
    """A single ``TrainingMetadata`` tagged with the component that produced it.

    Composite strategies own wrapped models (GARCH, LSTM, XGBoost, ARMA) — each
    has its own ``_training_metadata``. When a fold's leakage tripwire fires,
    the caller needs to know WHICH component drifted, not just "list index 2 of
    4". The ``origin`` string ("strategy" / "garch" / "lstm" / ...) carries
    that information from the composite's ``get_all_training_metadata()`` down
    to the error message. ``metadata`` is ``None`` when a component never
    completed ``fit()`` — downstream iteration logs a warning and skips.

    ``is_pretrained`` marks leaves loaded frozen from a prior standalone
    training run (injected via ``pretrained_leaves`` ctor kwarg). The deep
    metadata check runs a strict-no-overlap invariant for these: not only
    must ``train_end < fold.test_start`` (always enforced), but
    ``train_end < fold.train_start`` as well — otherwise the strategy fits
    its own state on bars where the leaf is in-sample, producing an
    inflated backtest. Fresh leaves (``is_pretrained=False``, the default)
    have ``train_end == fold.train_end`` by construction, so the stricter
    invariant is suppressed for them.
    """

    origin: str
    metadata: TrainingMetadata | None
    is_pretrained: bool = False


def collect_metadata(
    *pairs: tuple[str, TrainingMetadata | None],
) -> tuple[TrackedMetadata, ...]:
    """Bundle ``(origin, metadata)`` pairs into a ``tuple[TrackedMetadata, ...]``.

    Shared helper so every composite override reads as a flat list of
    ``("<origin>", self._<leaf>.training_metadata)`` pairs rather than a
    tuple-of-constructors. Centralises the ``None`` passthrough and keeps
    callers from reinventing the shape.
    """
    return tuple(TrackedMetadata(origin=o, metadata=m) for o, m in pairs)


def mark_pretrained(tracked: tuple[TrackedMetadata, ...]) -> tuple[TrackedMetadata, ...]:
    """Return a copy of ``tracked`` with ``is_pretrained=True`` on every entry.

    Composite leaves are frozen as whole units — when a strategy's
    ``pretrained_leaves`` map pins a key, every nested metadata entry from
    the leaf's ``get_all_training_metadata()`` inherits the frozen-from-
    disk status. Centralised here so every strategy's override reads
    identically and can't drift.
    """
    return tuple(replace(t, is_pretrained=True) for t in tracked)


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

    @staticmethod
    def from_dict(d: dict[str, object]) -> TrainingMetadata:
        """Deserialize from dict. Used when loading a saved model."""
        from src.core import json_io
        from src.core.types import Interval

        return TrainingMetadata(
            train_start=json_io.get_timestamp(d, "train_start"),
            train_end=json_io.get_timestamp(d, "train_end"),
            n_train_samples=json_io.get_int(d, "n_train_samples"),
            fit_timestamp=json_io.get_timestamp(d, "fit_timestamp"),
            interval=Interval(json_io.get_str(d, "interval")),
            feature_columns=tuple(json_io.get_str_list(d, "feature_columns")),
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


def resolve_holdout_boundary(
    df: pd.DataFrame,
    *,
    holdout_pct: float = 0.0,
    holdout_start: pd.Timestamp | None = None,
) -> pd.Timestamp | None:
    """Resolve the absolute timestamp at which the holdout region begins.

    The holdout contract is described at length on
    :class:`src.core.config.ValidationConfig`; in short, it carves the END of
    ``df`` off so walk-forward / HPO never see it. This helper is the single
    canonical resolver used by the runner, so that every call site derives
    the boundary identically (seed once, persist, re-read from manifest).

    Inputs
    ------
    df:
        The full fetched OHLCV frame. Must have a ``DatetimeIndex`` with at
        least 2 rows.
    holdout_pct:
        Fraction of ``df`` to reserve as holdout, sliced from the end. The
        cutoff index is ``int(len(df) * (1 - holdout_pct))`` and the boundary
        is ``df.index[cutoff]``. Mutually exclusive with ``holdout_start``.
    holdout_start:
        Pinned absolute timestamp at which holdout begins. Must be present in
        ``df.index`` (not merely in range). Mutually exclusive with
        ``holdout_pct``.

    Returns
    -------
    The first timestamp of the holdout region, or ``None`` if neither knob
    requests a reservation. Runners slicing with this boundary should use
    ``df[df.index < boundary]`` for ``dev`` and ``df[df.index >= boundary]``
    for ``holdout`` — the returned timestamp is the first bar OF holdout,
    not the last bar of dev.

    Raises
    ------
    TypeError:
        If ``df`` has no ``DatetimeIndex``.
    ValueError:
        If both knobs are non-default (caller should have validated at the
        config layer; this is defense-in-depth). Also if ``holdout_pct`` on
        ``len(df)`` rows would leave an empty dev or empty holdout.
    LeakageError:
        If ``holdout_start`` is not present in ``df.index``. The pinned
        timestamp was presumably written into a manifest on a prior run; its
        absence means the fetched data drifted (vendor adjustment, missing
        bar, holiday reclassification). Refusing here prevents a silent shift
        of the split boundary across runs — the exact leakage path ValidationConfig
        tripwire #2 defends against.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "resolve_holdout_boundary requires a DataFrame with a DatetimeIndex; "
            f"got {type(df.index).__name__}."
        )
    if holdout_pct > 0.0 and holdout_start is not None:
        raise ValueError(
            "resolve_holdout_boundary: at most one of holdout_pct / holdout_start "
            "may be set (the ValidationConfig model_validator should have caught this)."
        )

    if holdout_start is not None:
        ts = pd.Timestamp(holdout_start)
        if ts not in df.index:
            raise LeakageError(
                f"pinned holdout_start {ts} is not present in the fetched data "
                f"[{df.index[0]} .. {df.index[-1]}]; data may have drifted since "
                f"the boundary was recorded. Refusing to resolve — a silent shift "
                f"would move bars across the dev/holdout line."
            )
        return ts

    if holdout_pct > 0.0:
        n = len(df)
        cutoff = int(n * (1.0 - holdout_pct))
        if cutoff <= 0:
            raise ValueError(
                f"holdout_pct={holdout_pct} on {n} bars yields empty dev region "
                f"(cutoff={cutoff}); shrink holdout_pct or fetch more data."
            )
        if cutoff >= n:
            raise ValueError(
                f"holdout_pct={holdout_pct} on {n} bars yields empty holdout region "
                f"(cutoff={cutoff}); grow holdout_pct above the minimum per-bar fraction."
            )
        return pd.Timestamp(df.index[cutoff])

    return None
