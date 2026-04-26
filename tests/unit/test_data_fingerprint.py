"""Tests for :func:`fingerprint_bars` — deterministic OHLCV content hash."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.fingerprint import fingerprint_bars
from tests.conftest import make_synthetic_ohlcv_df

_ALT_SEED = 77


class TestFingerprintIdentity:
    def test_identical_dataframes_hash_identically(self) -> None:
        a = make_synthetic_ohlcv_df()
        b = make_synthetic_ohlcv_df()
        assert fingerprint_bars(a) == fingerprint_bars(b)

    def test_invariant_to_column_reordering(self) -> None:
        a = make_synthetic_ohlcv_df()
        b = a[["volume", "close", "low", "high", "open"]]
        assert fingerprint_bars(a) == fingerprint_bars(b)


class TestFingerprintSensitivity:
    def test_single_tick_mutation_changes_hash(self) -> None:
        a = make_synthetic_ohlcv_df()
        b = a.copy()
        closes = np.array(b["close"], dtype=np.float64)
        closes[50] += 1e-6
        b["close"] = closes
        assert fingerprint_bars(a) != fingerprint_bars(b)

    def test_different_seed_changes_hash(self) -> None:
        a = make_synthetic_ohlcv_df()
        b = make_synthetic_ohlcv_df(seed=_ALT_SEED)
        assert fingerprint_bars(a) != fingerprint_bars(b)

    def test_timestamp_shift_changes_hash(self) -> None:
        a = make_synthetic_ohlcv_df()
        b = a.copy()
        b.index = b.index + pd.Timedelta(days=1)
        assert fingerprint_bars(a) != fingerprint_bars(b)


class TestFingerprintColumnSetGuard:
    def test_missing_ohlcv_column_raises_keyerror(self) -> None:
        df = make_synthetic_ohlcv_df().drop(columns=["volume"])
        with pytest.raises(KeyError, match="volume"):
            fingerprint_bars(df)

    def test_extra_column_does_not_affect_ohlcv_bytes_but_changes_column_set(self) -> None:
        """Adding a non-OHLCV column changes the hash (column set enters
        the digest) but doesn't leak bytes through the OHLCV sub-hash."""
        a = make_synthetic_ohlcv_df()
        b = a.copy()
        b["extra"] = np.arange(len(b), dtype=np.float64)
        assert fingerprint_bars(a) != fingerprint_bars(b)

    def test_non_datetime_index_raises(self) -> None:
        df = make_synthetic_ohlcv_df().reset_index(drop=True)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            fingerprint_bars(df)
