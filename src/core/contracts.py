"""
Validation decorators for anti-leakage contracts.

These decorators enforce temporal integrity on functions that
process time-series data, preventing lookahead bias.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import pandas as pd

from src.core.exceptions import LeakageError


def _find_max_timestamp_in_args(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> pd.Timestamp | None:
    """
    Extract the maximum timestamp from any DataFrame arguments.
    """

    max_ts: pd.Timestamp | None = None
    for arg in (*args, *kwargs.values()):
        if isinstance(arg, pd.DataFrame) and isinstance(arg.index, pd.DatetimeIndex):
            ts = arg.index.max()
            if max_ts is None or ts > max_ts:
                max_ts = ts
    return max_ts


def no_future_data[**P, R](
    func: Callable[P, R],
) -> Callable[P, R]:
    """
    Validate that output timestamps do not exceed input max timestamp.

    Raises LeakageError if the output DataFrame/Series references
    timestamps beyond the input's maximum.

    Scope limitations (by design):
        - Only checks timestamp boundaries, not value-level contamination.
          A function that uses future prices to compute current features
          will NOT be caught if output timestamps stay within input range.
        - Non-pandas return types (numpy arrays, dicts, tuples) bypass
          the check entirely - only DataFrame/Series outputs are validated.
        - When multiple DataFrames are passed as arguments, the bound is
          the maximum timestamp across ALL inputs. This means a function
          receiving both train and test data will use the test max as the
          bound, which cannot detect cross-contamination.

    For full anti-leakage protection, combine with TemporalSplit (which
    enforces strict train/test separation) and WalkForwardValidator.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        input_max = _find_max_timestamp_in_args(args, kwargs)
        result = func(*args, **kwargs)

        if input_max is not None and isinstance(result, (pd.DataFrame, pd.Series)):
            if isinstance(result.index, pd.DatetimeIndex) and len(result) > 0:
                output_max = result.index.max()
                if output_max > input_max:
                    raise LeakageError(
                        f"Output contains future data: output max timestamp "
                        f"{output_max} > input max timestamp {input_max}; fix by "
                        f"removing the lookahead in the wrapped function (no "
                        f".shift(-k), no centered rolling windows, no .bfill())."
                    )
        return result

    return wrapper


def temporally_sorted[**P, R](
    func: Callable[P, R],
) -> Callable[P, R]:
    """
    Ensure input DataFrame has a monotonically increasing DatetimeIndex.

    Raises LeakageError if the input is not temporally sorted.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        for arg in (*args, *kwargs.values()):
            if isinstance(arg, pd.DataFrame) and isinstance(arg.index, pd.DatetimeIndex):
                if not arg.index.is_monotonic_increasing:
                    raise LeakageError(
                        "Input DataFrame is not temporally sorted; "
                        "DatetimeIndex must be monotonically increasing. Fix by "
                        "calling df.sort_index() before passing into the contract."
                    )
        return func(*args, **kwargs)

    return wrapper


def no_nan_in_output[**P, R](
    func: Callable[P, R],
) -> Callable[P, R]:
    """
    Assert that the returned DataFrame/Series has no NaN values.

    Raises ValueError if the output contains any NaN values.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        result = func(*args, **kwargs)

        if isinstance(result, pd.DataFrame):
            col_has_nan = result.isna().any()
            if col_has_nan.any():
                nan_cols = result.columns[col_has_nan].tolist()
                raise ValueError(
                    f"Output DataFrame contains NaN values in columns: {nan_cols}; "
                    f"fix by .dropna() at the wrapped function's boundary or by "
                    f"forward-filling warmup gaps before returning."
                )
        elif isinstance(result, pd.Series):
            if result.isna().any():
                raise ValueError(
                    "Output Series contains NaN values; fix by .dropna() at the "
                    "wrapped function's boundary or by forward-filling warmup gaps."
                )

        return result

    return wrapper
