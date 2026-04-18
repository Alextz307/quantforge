"""Opt-in perf guard: the C++ paths stay faster than their Python baselines.

Enable with ``PERF_GUARD=1``. CI does not gate on these (timing-flaky).
Each test measures best-of-N wall time for the C++ path vs a Python baseline
and asserts a minimum speedup ratio. GARCH and the state machines compare
against hand-rolled Python reference loops. RSI and MACD compare against
``pandas.ewm`` (the fastest pandas-native equivalent); the 2× threshold here
is a parity guard against drift, not a claim that C++ beats a pure-Python
implementation of the same indicator.
"""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

import quant_engine

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

pytestmark = pytest.mark.skipif(
    os.environ.get("PERF_GUARD") != "1",
    reason="set PERF_GUARD=1 to run perf-guard suite",
)

N_BARS = 10_000
WARMUP_REPS = 2
MEASURE_REPS = 5
SEED = 42

GARCH_SPEEDUP_MIN = 10.0
STATE_MACHINE_SPEEDUP_MIN = 20.0
RSI_SPEEDUP_MIN = 2.0
MACD_SPEEDUP_MIN = 2.0

GARCH_OMEGA = 0.05
GARCH_ALPHA = [0.10]
GARCH_BETA = [0.85]
GARCH_MU = 0.0
GARCH_BACKCAST = 1.0
VARIANCE_FLOOR = 1e-12

START_PRICE = 100.0
RETURN_STD = 0.01
BAND_HALF_WIDTH = 2.0
TREND_SCALE = 0.99
ZSCORE_STD = 1.5
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_LOSS_Z = 3.0

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def _best_of(fn: Callable[[], object]) -> float:
    for _ in range(WARMUP_REPS):
        fn()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        best = float("inf")
        for _ in range(MEASURE_REPS):
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            if dt < best:
                best = dt
        return best
    finally:
        if gc_was_enabled:
            gc.enable()


def _python_garch_filter(
    r: FloatArray,
    omega: float,
    alpha: list[float],
    beta: list[float],
    mu: float,
    backcast: float,
) -> FloatArray:
    n = len(r)
    sigma2 = np.empty(n)
    p, q = len(alpha), len(beta)
    for t in range(n):
        var_t = omega
        for i in range(p):
            idx = t - i - 1
            e2 = (r[idx] - mu) ** 2 if idx >= 0 else backcast
            var_t += alpha[i] * e2
        for j in range(q):
            idx = t - j - 1
            past = sigma2[idx] if idx >= 0 else backcast
            var_t += beta[j] * past
        sigma2[t] = max(var_t, VARIANCE_FLOOR)
    return sigma2


def _python_mean_reversion_sm(
    close: FloatArray,
    mid: FloatArray,
    upper: FloatArray,
    lower: FloatArray,
    trend_ma: FloatArray,
) -> FloatArray:
    n = len(close)
    out = np.full(n, np.nan)
    pos = 0.0
    for t in range(n):
        if np.isnan(mid[t]) or np.isnan(upper[t]) or np.isnan(lower[t]) or np.isnan(trend_ma[t]):
            continue
        is_bull = close[t] > trend_ma[t]
        if pos == 0.0:
            if is_bull and close[t] < lower[t]:
                pos = 1.0
            elif not is_bull and close[t] > upper[t]:
                pos = -1.0
        elif pos == 1.0:
            if close[t] >= mid[t]:
                pos = 0.0
        else:
            if close[t] <= mid[t]:
                pos = 0.0
        out[t] = pos
    return out


def _python_pairs_sm(z: FloatArray, entry: float, exit_z: float, stop: float) -> FloatArray:
    n = len(z)
    out = np.full(n, np.nan)
    pos = 0.0
    for t in range(n):
        if np.isnan(z[t]):
            continue
        abs_z = abs(z[t])
        if abs_z >= stop:
            pos = 0.0
        elif pos == 0.0:
            if z[t] >= entry:
                pos = -1.0
            elif z[t] <= -entry:
                pos = 1.0
        elif abs_z <= exit_z:
            pos = 0.0
        out[t] = pos
    return out


def _pandas_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _pandas_macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def test_garch_filter_speedup() -> None:
    rng = np.random.default_rng(SEED)
    r = rng.standard_normal(N_BARS)
    params = quant_engine.GarchParams(
        omega=GARCH_OMEGA,
        alpha=GARCH_ALPHA,
        beta=GARCH_BETA,
        mu=GARCH_MU,
        backcast=GARCH_BACKCAST,
    )
    py = _best_of(
        lambda: _python_garch_filter(
            r, GARCH_OMEGA, GARCH_ALPHA, GARCH_BETA, GARCH_MU, GARCH_BACKCAST
        )
    )
    cpp = _best_of(lambda: quant_engine.garch_filter(r, params))
    speedup = py / cpp
    assert speedup >= GARCH_SPEEDUP_MIN, (
        f"GARCH filter speedup {speedup:.1f}x < {GARCH_SPEEDUP_MIN}x"
    )


def test_mean_reversion_state_machine_speedup() -> None:
    rng = np.random.default_rng(SEED)
    close = START_PRICE + rng.standard_normal(N_BARS).cumsum()
    mid = close.copy()
    upper = close + BAND_HALF_WIDTH
    lower = close - BAND_HALF_WIDTH
    trend_ma = close * TREND_SCALE
    py = _best_of(lambda: _python_mean_reversion_sm(close, mid, upper, lower, trend_ma))
    cpp = _best_of(
        lambda: quant_engine.run_mean_reversion_state_machine(close, mid, upper, lower, trend_ma)
    )
    speedup = py / cpp
    assert speedup >= STATE_MACHINE_SPEEDUP_MIN, (
        f"mean-reversion SM speedup {speedup:.1f}x < {STATE_MACHINE_SPEEDUP_MIN}x"
    )


def test_pairs_state_machine_speedup() -> None:
    rng = np.random.default_rng(SEED)
    z = rng.normal(0.0, ZSCORE_STD, N_BARS)
    py = _best_of(lambda: _python_pairs_sm(z, ENTRY_Z, EXIT_Z, STOP_LOSS_Z))
    cpp = _best_of(lambda: quant_engine.run_pairs_state_machine(z, ENTRY_Z, EXIT_Z, STOP_LOSS_Z))
    speedup = py / cpp
    assert speedup >= STATE_MACHINE_SPEEDUP_MIN, (
        f"pairs SM speedup {speedup:.1f}x < {STATE_MACHINE_SPEEDUP_MIN}x"
    )


def test_rsi_speedup() -> None:
    rng = np.random.default_rng(SEED)
    close_arr = START_PRICE * np.exp(rng.standard_normal(N_BARS) * RETURN_STD).cumprod()
    close = pd.Series(close_arr)
    rsi = quant_engine.RSI(RSI_PERIOD)
    py = _best_of(lambda: _pandas_rsi(close, RSI_PERIOD))
    cpp = _best_of(lambda: rsi.compute(close_arr))
    speedup = py / cpp
    assert speedup >= RSI_SPEEDUP_MIN, f"RSI speedup {speedup:.1f}x < {RSI_SPEEDUP_MIN}x"


def test_macd_speedup() -> None:
    rng = np.random.default_rng(SEED)
    close_arr = START_PRICE * np.exp(rng.standard_normal(N_BARS) * RETURN_STD).cumprod()
    close = pd.Series(close_arr)
    macd = quant_engine.MACD(MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    py = _best_of(lambda: _pandas_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL))
    cpp = _best_of(lambda: macd.compute_all(close_arr))
    speedup = py / cpp
    assert speedup >= MACD_SPEEDUP_MIN, f"MACD speedup {speedup:.1f}x < {MACD_SPEEDUP_MIN}x"
