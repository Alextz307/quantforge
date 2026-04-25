"""Deterministic content hash for OHLCV DataFrames.

``fingerprint_bars(df)`` returns a version-stable SHA-256 digest covering the
exact bytes that affect downstream strategy state: the ordered list of
column names, the int64 timestamps in the index, and the float64 bytes of
the canonical OHLCV columns.

Why not ``df.to_parquet`` + hash, or ``pd.util.hash_pandas_object``?

* ``to_parquet`` mixes in pandas/version metadata that drifts with minor
  releases — a no-op upgrade would invalidate every cached hash.
* ``hash_pandas_object`` deliberately salts its output across pandas
  minor versions (its docstring flags this); the salt makes collisions
  rare but invalidates round-trip use across environments.

Both fail the property we need: ``fingerprint_bars(df)`` from one run must
equal the stored ``data_hash`` from a different run ON THE SAME DATA, so
``experiment holdout-eval`` can refuse to proceed on vendor-drifted data.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.core.constants import OHLCV_COLUMNS


def fingerprint_bars(df: pd.DataFrame) -> str:
    """SHA-256 content hash over ``df``'s column names, timestamps, and OHLCV bytes.

    ``df`` MUST have a ``DatetimeIndex`` and contain every column in
    :data:`OHLCV_COLUMNS` — ingestion-time ``validate_bars`` enforces this
    upstream. A missing column is treated as a caller contract violation
    and surfaces as a pointed ``KeyError``, not silently hashes to the
    same value as one where the column is all-zeros.

    Column ORDER in the input is IGNORED: we always hash column values in
    the canonical ``OHLCV_COLUMNS`` order, so two frames with identical
    contents but different column reorderings produce the same digest.
    Column SET is part of the hash via the joined column-name string —
    a frame missing ``volume`` cannot collide with one where volume is
    all-zeros.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "fingerprint_bars requires a DataFrame with a DatetimeIndex; "
            f"got {type(df.index).__name__}."
        )

    h = hashlib.sha256()
    # Mix the ordered column-name SET first — distinguishes a frame with
    # ``volume`` absent from one where ``volume == 0.0``. Sorted so the
    # hash is invariant to input column order.
    h.update(",".join(sorted(df.columns)).encode())
    h.update(df.index.values.view("int64").tobytes())
    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            raise KeyError(
                f"fingerprint_bars: DataFrame missing required column {col!r}; "
                f"present columns: {sorted(df.columns)}."
            )
        h.update(np.asarray(df[col], dtype=np.float64).tobytes())
    return h.hexdigest()
