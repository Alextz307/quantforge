"""
Semantic quality checks for a post-normalized OHLCV DataFrame.

Runs once per fetch (see ``IDataSource.fetch``), between ``DataNormalizer`` and
the cache write. Catches bad data at ingestion so it doesn't reach cached
storage or surface late from an indicator's OHLC guard deep inside a backtest.

Explicitly does NOT cover:
- Column presence / DatetimeIndex contracts - owned by ``DataNormalizer``.
- Structural alignment of bars/signals - owned by ``CppBacktestEngine._validate_inputs``.
- Temporal overlap of train/test windows - owned by ``TrainingMetadata.validate_no_overlap``.
- Gap detection - weekends/holidays on daily bars and session boundaries on
  intraday bars produce so many legitimate gaps that a general gap check is
  noisy; callers with interval-aware calendars should add their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.constants import OHLCV_COLUMNS
from src.core.exceptions import DataQualityError

_MAX_EXAMPLES = 3
_PRICE_COLUMNS = OHLCV_COLUMNS[:4]


def validate_bars(df: pd.DataFrame) -> None:
    """
    Raise ``DataQualityError`` if a normalized OHLCV frame fails sanity checks.

    Precondition: ``df`` has already been through ``DataNormalizer`` (lowercase
    OHLCV columns present, ``DatetimeIndex``, sorted by time). Defensive checks
    for those preconditions run first so a caller bypassing the normalizer gets
    a ``DataQualityError`` rather than an obscure KeyError downstream.
    """

    if df.empty:
        raise DataQualityError("bars: empty DataFrame")

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(f"bars: missing required columns {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataQualityError("bars: index must be DatetimeIndex")

    nat_mask = pd.Series(df.index.isna())
    if bool(nat_mask.any()):
        nat_positions = np.flatnonzero(nat_mask.to_numpy())[:_MAX_EXAMPLES].tolist()
        raise DataQualityError(f"bars: NaT in index at positions {nat_positions}")

    ohlcv = df[list(OHLCV_COLUMNS)]
    not_finite = ~np.isfinite(ohlcv.to_numpy(dtype=float, copy=False))
    bad_rows = not_finite.any(axis=1)
    if bool(bad_rows.any()):
        bad_cols = [c for i, c in enumerate(OHLCV_COLUMNS) if bool(not_finite[:, i].any())]
        first_idx = df.index[bad_rows][:_MAX_EXAMPLES].tolist()
        raise DataQualityError(
            f"bars: non-finite (NaN or inf) in columns {bad_cols} at {first_idx}"
        )

    for col in _PRICE_COLUMNS:
        non_pos = df[col] <= 0
        if bool(non_pos.any()):
            raise DataQualityError(
                f"bars: {col} must be > 0, "
                f"non-positive at {df.index[non_pos][:_MAX_EXAMPLES].tolist()}"
            )

    neg_vol = df["volume"] < 0
    if bool(neg_vol.any()):
        raise DataQualityError(
            f"bars: volume must be >= 0, negative at {df.index[neg_vol][:_MAX_EXAMPLES].tolist()}"
        )

    max_oc = df[["open", "close"]].max(axis=1)
    min_oc = df[["open", "close"]].min(axis=1)

    bad_high = df["high"] < max_oc
    if bool(bad_high.any()):
        raise DataQualityError(
            f"bars: high < max(open, close) at {df.index[bad_high][:_MAX_EXAMPLES].tolist()}"
        )

    bad_low = df["low"] > min_oc
    if bool(bad_low.any()):
        raise DataQualityError(
            f"bars: low > min(open, close) at {df.index[bad_low][:_MAX_EXAMPLES].tolist()}"
        )

    bad_hl = df["high"] < df["low"]
    if bool(bad_hl.any()):
        raise DataQualityError(f"bars: high < low at {df.index[bad_hl][:_MAX_EXAMPLES].tolist()}")

    if df.index.has_duplicates:
        dup_mask = df.index.duplicated(keep="first")
        dups = df.index[dup_mask][:_MAX_EXAMPLES].tolist()
        raise DataQualityError(f"bars: duplicate timestamps at {dups}")
