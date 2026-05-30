"""
Deterministic content hash for OHLCV DataFrames.

``fingerprint_bars(df)`` returns a version-stable SHA-256 digest covering the
exact bytes that affect downstream strategy state: the ordered list of
column names, the int64 timestamps in the index, and the float64 bytes of
the canonical OHLCV columns.

Why not ``df.to_parquet`` + hash, or ``pd.util.hash_pandas_object``?

* ``to_parquet`` mixes in pandas/version metadata that drifts with minor
  releases - a no-op upgrade would invalidate every cached hash.
* ``hash_pandas_object`` deliberately salts its output across pandas
  minor versions (its docstring flags this); the salt makes collisions
  rare but invalidates round-trip use across environments.

Both fail the property we need: ``fingerprint_bars(df)`` from one run must
equal the stored ``data_hash`` from a different run ON THE SAME DATA, so
``experiment holdout-eval`` can refuse to proceed on vendor-drifted data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.core.constants import OHLCV_COLUMNS, PAIRS_LEG_SUFFIXES
from src.core.exceptions import LeakageError


def _fingerprint(df: pd.DataFrame, value_columns: Sequence[str], func_name: str) -> str:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"{func_name} requires a DataFrame with a DatetimeIndex; "
            f"got {type(df.index).__name__}. Fix by setting df.index to a "
            f"DatetimeIndex (or calling df.set_index('date'))."
        )
    missing = [c for c in value_columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{func_name}: DataFrame missing required columns {missing}; "
            f"present columns: {sorted(df.columns)}. Fix by running the "
            f"frame through the appropriate ingestion path first."
        )
    h = hashlib.sha256()
    h.update(",".join(sorted(df.columns)).encode())
    h.update(df.index.values.view("int64").tobytes())
    for col in value_columns:
        h.update(np.asarray(df[col], dtype=np.float64).tobytes())
    return h.hexdigest()


def fingerprint_bars(df: pd.DataFrame) -> str:
    """
    SHA-256 content hash over ``df``'s column names, timestamps, and OHLCV bytes.

    ``df`` MUST have a ``DatetimeIndex`` and contain every column in
    :data:`OHLCV_COLUMNS`. Column ORDER in the input is IGNORED: values
    are always hashed in canonical ``OHLCV_COLUMNS`` order. Column SET is
    part of the hash so a frame missing ``volume`` cannot collide with
    one where volume is all-zeros.
    """

    return _fingerprint(df, OHLCV_COLUMNS, "fingerprint_bars")


def fingerprint_pair_bars(df: pd.DataFrame) -> str:
    """
    Wide-format pairs analogue of :func:`fingerprint_bars`.

    Expects ``open_a / high_a / ... / volume_b`` columns produced by the
    multi-ticker fetch path.
    """

    suffix_a, suffix_b = PAIRS_LEG_SUFFIXES
    leg_cols = [f"{c}{suffix_a}" for c in OHLCV_COLUMNS] + [f"{c}{suffix_b}" for c in OHLCV_COLUMNS]
    return _fingerprint(df, leg_cols, "fingerprint_pair_bars")


def fingerprint_multi_bars(df: pd.DataFrame, tickers: Sequence[str]) -> str:
    """
    Wide-format multi-feature analogue of :func:`fingerprint_bars`.

    Expects ``<ohlcv>_<TICKER>`` columns produced by the multi-feature
    fetch path (e.g. ``close_SPY``, ``open_QQQ``). Tickers are sorted
    before hashing so the digest is invariant to input ticker ORDER -
    the same set of tickers always produces the same hash regardless of
    whether the caller passed ``[SPY, QQQ]`` or ``[QQQ, SPY]``.
    """

    sorted_tickers = sorted(tickers)
    leg_cols = [f"{c}_{t}" for t in sorted_tickers for c in OHLCV_COLUMNS]
    return _fingerprint(df, leg_cols, "fingerprint_multi_bars")


def assert_data_hash_matches(
    actual: str,
    expected: str,
    *,
    context: str,
    fix_hint: str | None = None,
) -> None:
    """
    Refuse on ``data_hash`` drift between a refetch and the manifest.

    Drifted bars silently slide every temporal boundary downstream
    (dev/holdout, walk-forward fold edges). That's an anti-leakage
    vector regardless of which consumer reads it next, so we centralise
    the check + raise here and use :class:`LeakageError` uniformly.
    ``context`` names the consumer (e.g. ``"holdout boundary anchor"``)
    so the raised message identifies which caller fired the tripwire
    without the caller needing to spell out the consequence each time.

    ``fix_hint`` overrides the default "use the same data source /
    cache" instruction. Callers whose recovery path is different pass a
    more specific hint here.

    Callers that need to know the actual hash for a downstream payload
    compute it once and pass it in; the helper does NOT recompute (the
    fingerprint is O(N bars) and a comparison loop should pay it once).
    """

    if actual != expected:
        hint = fix_hint or (
            "Fix by using the same data source / cache as the original "
            "run, or re-run the source so its manifest reflects the new bars."
        )
        raise LeakageError(
            f"data_hash drift detected ({context}): manifest recorded "
            f"{expected[:12]}..., re-fetched {actual[:12]}...; using "
            f"drifted bars would silently shift temporal boundaries. "
            f"{hint}"
        )
