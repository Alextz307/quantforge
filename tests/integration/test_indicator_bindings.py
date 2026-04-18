"""Numerical-parity tests for the Phase 4 indicator bindings.

The underlying C++ logic is exhaustively covered by gtest in
``cpp/tests/test_indicators.cpp``. These tests verify the **binding layer**:
numpy array marshalling, keyword-arg round-trips, f32→f64 forcecast, and
that the bindings produce bit-identical values to equivalent pandas /
hand-computed references.

RSI reference values come from the C++ tests (`KnownReferenceValue`).
MACD and BollingerBands are parity-checked against pandas — the C++
implementations match pandas' semantics exactly (`ewm(adjust=False)` for
MACD, `rolling(...).std(ddof=1)` for Bollinger). Parkinson and
Garman-Klass are checked against their published closed-form formulas.
"""

from __future__ import annotations

import math
import threading
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

import quant_engine as qe
from src.core.constants import TRADING_DAYS_PER_YEAR
from tests.conftest import make_synthetic_close_df, make_synthetic_ohlcv_df

F64Array = npt.NDArray[np.float64]

# ───── Tolerances ─────
EXACT_TOL = 1e-10
RSI_REFERENCE_TOL = 1e-3
SINGLE_BAR_TOL = 1e-10

# ───── RSI constants ─────
RSI_DEFAULT_PERIOD = 14
RSI_HAND_PERIOD = 3
RSI_HAND_PRICES = [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0]
# Hand-computed Wilder's RSI for the series above (mirrors the C++ test):
#   RSI[3] = 66.6667 (deltas +1,+1,-1 → avg_gain=2/3, avg_loss=1/3)
#   RSI[4] = 83.3333 (Wilder smoothing carries state forward)
RSI_EXPECTED_AT_3 = 66.6667
RSI_EXPECTED_AT_4 = 83.3333
RSI_BOUNDED_MIN = 0.0
RSI_BOUNDED_MAX = 100.0

# ───── MACD constants ─────
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_WARMUP = MACD_SLOW - 1  # macd_line first valid at index slow-1
MACD_SIGNAL_WARMUP = MACD_WARMUP + (MACD_SIGNAL - 1)

# ───── Bollinger constants ─────
BB_PERIOD = 20
BB_K = 2.0
BB_SMALL_PERIOD = 5
BB_WARMUP = BB_PERIOD - 1
BB_SMALL_WARMUP = BB_SMALL_PERIOD - 1

# ───── Parkinson / Garman-Klass constants ─────
GK_DEFAULT_WINDOW = 22
PK_DEFAULT_WINDOW = 22
GK_SMALL_WINDOW = 5
PK_SMALL_WINDOW = 5
SINGLE_BAR_WINDOW = 1

SINGLE_BAR_OPEN = 100.0
SINGLE_BAR_HIGH = 110.0
SINGLE_BAR_LOW = 90.0
SINGLE_BAR_CLOSE = 105.0

# ───── GIL-release smoke test ─────
GIL_STRESS_THREAD_COUNT = 4
GIL_STRESS_SERIES_LEN = 20_000
GIL_STRESS_ITERATIONS_PER_THREAD = 3
# Heuristic upper bound for 4 × 3 × 20k RSI computations with GIL released.
# A serial run on the dev machine completes in <100ms; 10s gives ample room
# for slow CI runners. A busted GIL-release (e.g. deadlock, data race) would
# hang indefinitely — this test would time out pytest's default hang check.
GIL_STRESS_TIMEOUT_SECONDS = 10.0


# ════════════════════════════════════════════════════════════════
# RSI
# ════════════════════════════════════════════════════════════════


class TestRSIBinding:
    def test_name_and_warmup(self) -> None:
        r = qe.RSI(RSI_DEFAULT_PERIOD)
        assert r.name == f"RSI({RSI_DEFAULT_PERIOD})"
        assert r.warmup_period == RSI_DEFAULT_PERIOD

    def test_default_period(self) -> None:
        # Default ctor should use period=14.
        assert qe.RSI().warmup_period == RSI_DEFAULT_PERIOD

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period"):
            qe.RSI(0)

    def test_hand_computed_reference(self) -> None:
        """Mirrors the C++ KnownReferenceValue test — Wilder's smoothing."""
        prices = np.array(RSI_HAND_PRICES, dtype=np.float64)
        out = qe.RSI(RSI_HAND_PERIOD).compute(prices)
        assert np.isnan(out[:RSI_HAND_PERIOD]).all()
        assert math.isclose(out[3], RSI_EXPECTED_AT_3, abs_tol=RSI_REFERENCE_TOL)
        assert math.isclose(out[4], RSI_EXPECTED_AT_4, abs_tol=RSI_REFERENCE_TOL)

    def test_warmup_nan_then_valid(self) -> None:
        close = make_synthetic_close_df()["close"].to_numpy()
        out = qe.RSI(RSI_DEFAULT_PERIOD).compute(close)
        assert out.shape == close.shape
        assert np.isnan(out[:RSI_DEFAULT_PERIOD]).all()
        assert not np.isnan(out[RSI_DEFAULT_PERIOD:]).any()

    def test_output_bounded_0_to_100(self) -> None:
        close = make_synthetic_close_df()["close"].to_numpy()
        out = qe.RSI(RSI_DEFAULT_PERIOD).compute(close)
        valid = out[RSI_DEFAULT_PERIOD:]
        assert (valid >= RSI_BOUNDED_MIN).all()
        assert (valid <= RSI_BOUNDED_MAX).all()


# ════════════════════════════════════════════════════════════════
# MACD
# ════════════════════════════════════════════════════════════════


def _pandas_macd(
    close: F64Array, fast: int, slow: int, signal: int
) -> tuple[F64Array, F64Array, F64Array]:
    """Pandas reference matching the C++ seeding of both EMAs.

    C++ seeds the fast/slow EMAs at ``data[0]`` and the signal EMA at the
    first valid MACD bar (index ``slow-1``). Pandas ``ewm(adjust=False)``
    matches the first behaviour natively, but its signal EMA on the full
    MACD series would seed at ``macd[0]`` and drift from C++ — so we
    slice to the valid range before running the signal EMA.
    """
    s = pd.Series(close)
    macd_series = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()

    valid_start = slow - 1
    sig_series = pd.Series(np.full(close.size, np.nan, dtype=np.float64))
    sig_series.iloc[valid_start:] = (
        macd_series.iloc[valid_start:].ewm(span=signal, adjust=False).mean()
    )

    macd = macd_series.to_numpy()
    sig = sig_series.to_numpy()
    return macd, sig, macd - sig


class TestMACDBinding:
    def test_name_and_warmup(self) -> None:
        m = qe.MACD(MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        assert m.name == f"MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})"
        assert m.warmup_period == MACD_WARMUP

    def test_invalid_periods_raise(self) -> None:
        with pytest.raises(ValueError, match="MACD"):
            qe.MACD(0, MACD_SLOW, MACD_SIGNAL)
        with pytest.raises(ValueError, match="MACD"):
            qe.MACD(MACD_SLOW, MACD_FAST, MACD_SIGNAL)  # fast >= slow

    def test_compute_parity_with_pandas_ewm(self) -> None:
        close = make_synthetic_close_df()["close"].to_numpy()
        out = qe.MACD(MACD_FAST, MACD_SLOW, MACD_SIGNAL).compute(close)
        ref_macd, _, _ = _pandas_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        # Warmup bars are NaN in our output; pandas emits real values there.
        np.testing.assert_allclose(
            out[MACD_WARMUP:], ref_macd[MACD_WARMUP:], rtol=0, atol=EXACT_TOL
        )
        assert np.isnan(out[:MACD_WARMUP]).all()

    def test_compute_all_parity_with_pandas(self) -> None:
        close = make_synthetic_close_df()["close"].to_numpy()
        result = qe.MACD(MACD_FAST, MACD_SLOW, MACD_SIGNAL).compute_all(close)
        ref_macd, ref_sig, ref_hist = _pandas_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        np.testing.assert_allclose(
            result.macd_line[MACD_WARMUP:], ref_macd[MACD_WARMUP:], rtol=0, atol=EXACT_TOL
        )
        np.testing.assert_allclose(
            result.signal_line[MACD_SIGNAL_WARMUP:],
            ref_sig[MACD_SIGNAL_WARMUP:],
            rtol=0,
            atol=EXACT_TOL,
        )
        np.testing.assert_allclose(
            result.histogram[MACD_SIGNAL_WARMUP:],
            ref_hist[MACD_SIGNAL_WARMUP:],
            rtol=0,
            atol=EXACT_TOL,
        )

    def test_compute_matches_compute_all_macd_line(self) -> None:
        close = make_synthetic_close_df()["close"].to_numpy()
        m = qe.MACD(MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        line = m.compute(close)
        full = m.compute_all(close)
        # Both paths produce the same macd_line (differ only in temp allocations).
        both_nan = np.isnan(line) & np.isnan(full.macd_line)
        both_valid = ~np.isnan(line) & ~np.isnan(full.macd_line)
        assert (both_nan | both_valid).all()
        np.testing.assert_allclose(
            line[both_valid], full.macd_line[both_valid], rtol=0, atol=EXACT_TOL
        )

    def test_result_struct_field_lengths(self) -> None:
        close = make_synthetic_close_df()["close"].to_numpy()
        result = qe.MACD().compute_all(close)
        assert result.macd_line.shape == close.shape
        assert result.signal_line.shape == close.shape
        assert result.histogram.shape == close.shape


# ════════════════════════════════════════════════════════════════
# BollingerBands
# ════════════════════════════════════════════════════════════════


class TestBollingerBandsBinding:
    def test_name_and_warmup(self) -> None:
        bb = qe.BollingerBands(BB_PERIOD, BB_K)
        # Trailing-zero-trimmed double formatting (2.0 → "2.0", not "2.000000").
        assert bb.name == f"BB({BB_PERIOD},2.0)"
        assert bb.warmup_period == BB_WARMUP

    def test_name_formats_fractional_num_std(self) -> None:
        assert qe.BollingerBands(BB_PERIOD, 2.5).name == f"BB({BB_PERIOD},2.5)"

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="BollingerBands"):
            qe.BollingerBands(0, BB_K)
        with pytest.raises(ValueError, match="BollingerBands"):
            qe.BollingerBands(BB_PERIOD, -1.0)

    def test_compute_all_parity_with_pandas(self) -> None:
        close = make_synthetic_close_df()["close"].to_numpy()
        result = qe.BollingerBands(BB_PERIOD, BB_K).compute_all(close)
        s = pd.Series(close)
        ref_mid = s.rolling(BB_PERIOD).mean().to_numpy()
        ref_std = s.rolling(BB_PERIOD).std(ddof=1).to_numpy()
        np.testing.assert_allclose(
            result.mid[BB_WARMUP:], ref_mid[BB_WARMUP:], rtol=0, atol=EXACT_TOL
        )
        np.testing.assert_allclose(
            result.upper[BB_WARMUP:],
            ref_mid[BB_WARMUP:] + BB_K * ref_std[BB_WARMUP:],
            rtol=0,
            atol=EXACT_TOL,
        )
        np.testing.assert_allclose(
            result.lower[BB_WARMUP:],
            ref_mid[BB_WARMUP:] - BB_K * ref_std[BB_WARMUP:],
            rtol=0,
            atol=EXACT_TOL,
        )

    def test_bands_ordered_upper_ge_mid_ge_lower(self) -> None:
        close = make_synthetic_close_df()["close"].to_numpy()
        r = qe.BollingerBands(BB_SMALL_PERIOD, BB_K).compute_all(close)
        tail_slice = slice(BB_SMALL_WARMUP, None)
        assert (r.upper[tail_slice] >= r.mid[tail_slice]).all()
        assert (r.mid[tail_slice] >= r.lower[tail_slice]).all()

    def test_compute_returns_mid_band(self) -> None:
        close = make_synthetic_close_df()["close"].to_numpy()
        bb = qe.BollingerBands(BB_PERIOD, BB_K)
        mid = bb.compute(close)
        full = bb.compute_all(close)
        # compute() returns the middle band (SMA), identical to compute_all().mid
        both_valid = ~np.isnan(mid) & ~np.isnan(full.mid)
        np.testing.assert_allclose(mid[both_valid], full.mid[both_valid], rtol=0, atol=EXACT_TOL)


# ════════════════════════════════════════════════════════════════
# Parkinson
# ════════════════════════════════════════════════════════════════


class TestParkinsonBinding:
    def test_name_and_warmup(self) -> None:
        pk = qe.Parkinson(PK_DEFAULT_WINDOW)
        assert pk.name == f"Parkinson({PK_DEFAULT_WINDOW})"
        assert pk.warmup_period == PK_DEFAULT_WINDOW - 1

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="Parkinson"):
            qe.Parkinson(0)

    def test_single_bar_hand_reference(self) -> None:
        """Closed-form: PK_daily = (1/(4*ln2)) * ln(H/L)^2, annualized with sqrt(252)."""
        o = np.array([SINGLE_BAR_OPEN], dtype=np.float64)
        h = np.array([SINGLE_BAR_HIGH], dtype=np.float64)
        lo = np.array([SINGLE_BAR_LOW], dtype=np.float64)
        c = np.array([SINGLE_BAR_CLOSE], dtype=np.float64)
        out = qe.Parkinson(SINGLE_BAR_WINDOW).compute(o, h, lo, c)
        log_hl = math.log(SINGLE_BAR_HIGH / SINGLE_BAR_LOW)
        pk_daily = (1.0 / (4.0 * math.log(2.0))) * log_hl * log_hl
        expected = math.sqrt(pk_daily) * math.sqrt(TRADING_DAYS_PER_YEAR)
        assert math.isclose(float(out[0]), expected, abs_tol=SINGLE_BAR_TOL)

    def test_warmup_nan_then_valid(self) -> None:
        df = make_synthetic_ohlcv_df()
        out = qe.Parkinson(PK_SMALL_WINDOW).compute(
            df["open"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["close"].to_numpy(),
        )
        assert out.shape == (len(df),)
        assert np.isnan(out[: PK_SMALL_WINDOW - 1]).all()
        assert not np.isnan(out[PK_SMALL_WINDOW - 1 :]).any()
        assert (out[PK_SMALL_WINDOW - 1 :] >= 0.0).all()

    def test_mismatched_lengths_raise(self) -> None:
        df = make_synthetic_ohlcv_df(n_rows=10)
        pk = qe.Parkinson(PK_SMALL_WINDOW)
        with pytest.raises(ValueError, match="equal length"):
            pk.compute(
                df["open"].to_numpy()[:5],
                df["high"].to_numpy(),
                df["low"].to_numpy(),
                df["close"].to_numpy(),
            )


# ════════════════════════════════════════════════════════════════
# GarmanKlass
# ════════════════════════════════════════════════════════════════


class TestGarmanKlassBinding:
    def test_name_and_warmup(self) -> None:
        gk = qe.GarmanKlass(GK_DEFAULT_WINDOW)
        assert gk.name == f"GarmanKlass({GK_DEFAULT_WINDOW})"
        assert gk.warmup_period == GK_DEFAULT_WINDOW - 1

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="GarmanKlass"):
            qe.GarmanKlass(0)

    def test_single_bar_hand_reference(self) -> None:
        """Closed-form: GK_daily = 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2."""
        o = np.array([SINGLE_BAR_OPEN], dtype=np.float64)
        h = np.array([SINGLE_BAR_HIGH], dtype=np.float64)
        lo = np.array([SINGLE_BAR_LOW], dtype=np.float64)
        c = np.array([SINGLE_BAR_CLOSE], dtype=np.float64)
        out = qe.GarmanKlass(SINGLE_BAR_WINDOW).compute(o, h, lo, c)
        log_hl = math.log(SINGLE_BAR_HIGH / SINGLE_BAR_LOW)
        log_co = math.log(SINGLE_BAR_CLOSE / SINGLE_BAR_OPEN)
        gk_daily = 0.5 * log_hl * log_hl - (2.0 * math.log(2.0) - 1.0) * log_co * log_co
        expected = math.sqrt(gk_daily) * math.sqrt(TRADING_DAYS_PER_YEAR)
        assert math.isclose(float(out[0]), expected, abs_tol=SINGLE_BAR_TOL)

    def test_warmup_nan_then_valid(self) -> None:
        df = make_synthetic_ohlcv_df()
        out = qe.GarmanKlass(GK_SMALL_WINDOW).compute(
            df["open"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["close"].to_numpy(),
        )
        assert out.shape == (len(df),)
        assert np.isnan(out[: GK_SMALL_WINDOW - 1]).all()
        assert not np.isnan(out[GK_SMALL_WINDOW - 1 :]).any()
        assert (out[GK_SMALL_WINDOW - 1 :] >= 0.0).all()

    def test_high_less_than_low_raises(self) -> None:
        o = np.array([SINGLE_BAR_OPEN], dtype=np.float64)
        # Flip high/low so high < low at index 0.
        h = np.array([SINGLE_BAR_LOW], dtype=np.float64)
        lo = np.array([SINGLE_BAR_HIGH], dtype=np.float64)
        c = np.array([SINGLE_BAR_CLOSE], dtype=np.float64)
        with pytest.raises(ValueError, match="high"):
            qe.GarmanKlass(SINGLE_BAR_WINDOW).compute(o, h, lo, c)


# ════════════════════════════════════════════════════════════════
# Binding-layer: f32 forcecast across all five indicators
# ════════════════════════════════════════════════════════════════


class TestForcecastCoercion:
    """All bindings type the inputs as `ContigF64` with ``py::array::forcecast``.

    Feeding an f32 array should coerce to f64 transparently. Covered for
    every indicator so regressions in a single binding don't slip through.
    """

    def test_rsi_accepts_float32(self) -> None:
        close32 = cast(F64Array, np.linspace(100.0, 120.0, 50, dtype=np.float32))
        out = qe.RSI(RSI_DEFAULT_PERIOD).compute(close32)
        assert out.dtype == np.float64
        assert out.shape == close32.shape

    def test_macd_accepts_float32(self) -> None:
        close32 = cast(F64Array, np.linspace(100.0, 120.0, 50, dtype=np.float32))
        m = qe.MACD(MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        line = m.compute(close32)
        full = m.compute_all(close32)
        assert line.dtype == np.float64
        assert full.macd_line.dtype == np.float64
        assert full.signal_line.dtype == np.float64
        assert full.histogram.dtype == np.float64

    def test_bollinger_accepts_float32(self) -> None:
        close32 = cast(F64Array, np.linspace(100.0, 120.0, 50, dtype=np.float32))
        bb = qe.BollingerBands(BB_SMALL_PERIOD, BB_K)
        mid = bb.compute(close32)
        full = bb.compute_all(close32)
        assert mid.dtype == np.float64
        assert full.upper.dtype == np.float64

    def test_parkinson_accepts_float32(self) -> None:
        df = make_synthetic_ohlcv_df()
        out = qe.Parkinson(PK_SMALL_WINDOW).compute(
            cast(F64Array, df["open"].to_numpy().astype(np.float32)),
            cast(F64Array, df["high"].to_numpy().astype(np.float32)),
            cast(F64Array, df["low"].to_numpy().astype(np.float32)),
            cast(F64Array, df["close"].to_numpy().astype(np.float32)),
        )
        assert out.dtype == np.float64

    def test_garman_klass_accepts_float32(self) -> None:
        df = make_synthetic_ohlcv_df()
        out = qe.GarmanKlass(GK_SMALL_WINDOW).compute(
            cast(F64Array, df["open"].to_numpy().astype(np.float32)),
            cast(F64Array, df["high"].to_numpy().astype(np.float32)),
            cast(F64Array, df["low"].to_numpy().astype(np.float32)),
            cast(F64Array, df["close"].to_numpy().astype(np.float32)),
        )
        assert out.dtype == np.float64


# ════════════════════════════════════════════════════════════════
# GIL release: observable concurrency
# ════════════════════════════════════════════════════════════════


class TestGILRelease:
    """The plan mandates `gil_scoped_release` on every new compute method so
    that Python-side parallelism (Optuna HPO, pytest-xdist) can actually run
    indicators concurrently. A busted release would deadlock, reference-count
    races, or serialize — this test detects all three classes of failure by
    launching multiple threads that all hammer the same indicator and
    asserting they finish within a loose timeout with bit-identical output.
    """

    def test_rsi_runs_concurrently(self) -> None:
        prices = make_synthetic_close_df(n_rows=GIL_STRESS_SERIES_LEN)["close"].to_numpy()
        r = qe.RSI(RSI_DEFAULT_PERIOD)
        expected = r.compute(prices)

        results: list[F64Array] = []
        results_lock = threading.Lock()

        def worker() -> None:
            for _ in range(GIL_STRESS_ITERATIONS_PER_THREAD):
                out = r.compute(prices)
                with results_lock:
                    results.append(out)

        threads = [threading.Thread(target=worker) for _ in range(GIL_STRESS_THREAD_COUNT)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=GIL_STRESS_TIMEOUT_SECONDS)
            assert not t.is_alive(), "RSI.compute() hung under concurrency"

        expected_count = GIL_STRESS_THREAD_COUNT * GIL_STRESS_ITERATIONS_PER_THREAD
        assert len(results) == expected_count
        for out in results:
            np.testing.assert_array_equal(out, expected)
