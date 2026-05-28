"""
Numerical-parity tests for the C++ state-machine bindings.

The recursive position-carry logic is exhaustively covered by gtest in
``cpp/tests/test_state_machines.cpp``. These tests verify the **binding layer**:
numpy array marshalling, keyword-argument surface, NaN passthrough, and that
``AdaptiveBollingerStrategy`` / ``PairsTradingStrategy`` still produce
bit-identical positions when their recursion is delegated to C++.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

import quant_engine as qe

F64Array = npt.NDArray[np.float64]

ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_LOSS_Z = 3.0


def _python_mean_reversion(
    close: F64Array,
    mid: F64Array,
    upper: F64Array,
    lower: F64Array,
    trend_ma: F64Array,
) -> F64Array:
    """
    Pure-Python reference mirroring the original Python state machine.

    Kept inlined so the test remains valid if the Python implementation is
    ever re-deleted or refactored.
    """

    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    position = 0.0
    for t in range(n):
        if np.isnan(mid[t]) or np.isnan(upper[t]) or np.isnan(lower[t]) or np.isnan(trend_ma[t]):
            continue
        is_bull = close[t] > trend_ma[t]
        if position == 0.0:
            if is_bull and close[t] < lower[t]:
                position = 1.0
            elif (not is_bull) and close[t] > upper[t]:
                position = -1.0
        elif position == 1.0:
            if close[t] >= mid[t]:
                position = 0.0
        elif position == -1.0:
            if close[t] <= mid[t]:
                position = 0.0
        out[t] = position
    return out


def _python_pairs(
    zscore: F64Array,
    entry_z: float,
    exit_z: float,
    stop_loss_z: float,
) -> F64Array:
    n = len(zscore)
    out = np.full(n, np.nan, dtype=np.float64)
    position = 0.0
    for t in range(n):
        z = zscore[t]
        if np.isnan(z):
            continue
        if abs(z) >= stop_loss_z:
            position = 0.0
        elif position == 0.0:
            if z >= entry_z:
                position = -1.0
            elif z <= -entry_z:
                position = 1.0
        elif abs(z) <= exit_z:
            position = 0.0
        out[t] = position
    return out


class TestMeanReversionBinding:
    @pytest.mark.parametrize(
        ("close", "mid", "upper", "lower", "trend_ma", "expected"),
        [
            pytest.param(
                np.array([90.0, 95.0, 100.0]),
                np.array([100.0, 100.0, 100.0]),
                np.array([110.0, 110.0, 110.0]),
                np.array([92.0, 92.0, 92.0]),
                np.array([80.0, 80.0, 80.0]),
                np.array([1.0, 1.0, 0.0]),
                id="bull_long_entry_exit",
            ),
            pytest.param(
                np.array([115.0, 105.0, 100.0]),
                np.array([100.0, 100.0, 100.0]),
                np.array([108.0, 108.0, 108.0]),
                np.array([90.0, 90.0, 90.0]),
                np.array([120.0, 120.0, 120.0]),
                np.array([-1.0, -1.0, 0.0]),
                id="bear_short_entry_exit",
            ),
            pytest.param(
                np.array([90.0, 95.0, 96.0]),
                np.array([100.0, 100.0, 100.0]),
                np.array([110.0, np.nan, 110.0]),
                np.array([92.0, 92.0, 92.0]),
                np.array([80.0, 80.0, 80.0]),
                np.array([1.0, np.nan, 1.0]),
                id="nan_band_holds_position",
            ),
        ],
    )
    def test_hand_built_parity_against_python_reference(
        self,
        close: F64Array,
        mid: F64Array,
        upper: F64Array,
        lower: F64Array,
        trend_ma: F64Array,
        expected: F64Array,
    ) -> None:
        got = qe.run_mean_reversion_state_machine(
            close=close, mid=mid, upper=upper, lower=lower, trend_ma=trend_ma
        )
        ref = _python_mean_reversion(close, mid, upper, lower, trend_ma)
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(got, ref)

    def test_empty_input_returns_empty(self) -> None:
        empty = np.array([], dtype=np.float64)
        got = qe.run_mean_reversion_state_machine(
            close=empty, mid=empty, upper=empty, lower=empty, trend_ma=empty
        )
        assert got.shape == (0,)

    def test_length_mismatch_raises(self) -> None:
        a = np.ones(5, dtype=np.float64)
        b = np.ones(4, dtype=np.float64)
        with pytest.raises(ValueError, match="same length"):
            qe.run_mean_reversion_state_machine(close=a, mid=b, upper=a, lower=a, trend_ma=a)


class TestPairsBinding:
    @pytest.mark.parametrize(
        ("z_values", "expected"),
        [
            pytest.param(
                [0.0, 2.5, 0.3, 0.1],
                [0.0, -1.0, 0.0, 0.0],
                id="entry_short_and_exit",
            ),
            pytest.param(
                [0.0, -2.5, -0.3, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                id="entry_long_and_exit",
            ),
            pytest.param(
                [2.5, 3.5, 1.0],
                [-1.0, 0.0, 0.0],
                id="stop_loss_forces_flat",
            ),
            pytest.param(
                [2.5, np.nan, 0.3],
                [-1.0, np.nan, 0.0],
                id="nan_holds_position",
            ),
        ],
    )
    def test_hand_built_parity_against_python_reference(
        self, z_values: list[float], expected: list[float]
    ) -> None:
        z = np.array(z_values, dtype=np.float64)
        exp = np.array(expected, dtype=np.float64)
        got = qe.run_pairs_state_machine(
            zscore=z, entry_zscore=ENTRY_Z, exit_zscore=EXIT_Z, stop_loss_zscore=STOP_LOSS_Z
        )
        ref = _python_pairs(z, ENTRY_Z, EXIT_Z, STOP_LOSS_Z)
        np.testing.assert_array_equal(got, exp)
        np.testing.assert_array_equal(got, ref)

    def test_empty_input_returns_empty(self) -> None:
        got = qe.run_pairs_state_machine(
            zscore=np.array([], dtype=np.float64),
            entry_zscore=ENTRY_Z,
            exit_zscore=EXIT_Z,
            stop_loss_zscore=STOP_LOSS_Z,
        )
        assert got.shape == (0,)


class TestFloat32Forcecast:
    """
    pybind11's ``py::array::forcecast`` should accept f32 and up-cast.
    """

    def test_mean_reversion_accepts_float32(self) -> None:
        close = np.array([90.0, 95.0, 100.0], dtype=np.float32)
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        upper = np.array([110.0, 110.0, 110.0], dtype=np.float32)
        lower = np.array([92.0, 92.0, 92.0], dtype=np.float32)
        trend_ma = np.array([80.0, 80.0, 80.0], dtype=np.float32)

        # Stub declares f64; pybind11's forcecast accepts f32 and up-casts.
        got = qe.run_mean_reversion_state_machine(
            close=cast(F64Array, close),
            mid=cast(F64Array, mid),
            upper=cast(F64Array, upper),
            lower=cast(F64Array, lower),
            trend_ma=cast(F64Array, trend_ma),
        )
        np.testing.assert_array_equal(got, np.array([1.0, 1.0, 0.0]))

    def test_pairs_accepts_float32(self) -> None:
        z = np.array([0.0, 2.5, 0.3], dtype=np.float32)
        got = qe.run_pairs_state_machine(
            zscore=cast(F64Array, z),
            entry_zscore=ENTRY_Z,
            exit_zscore=EXIT_Z,
            stop_loss_zscore=STOP_LOSS_Z,
        )
        np.testing.assert_array_equal(got, np.array([0.0, -1.0, 0.0]))


class TestLongSeriesParity:
    """
    Smoke-check C++ / Python parity on a longer synthetic series.
    """

    def test_mean_reversion_matches_python_reference(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        close_series = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 0.5, n)))
        mid_series = close_series.rolling(20).mean()
        band_half = 1.0
        trend_series = close_series.rolling(50).mean()

        close = np.asarray(close_series, dtype=np.float64)
        mid = np.asarray(mid_series, dtype=np.float64)
        upper = np.asarray(mid_series + band_half, dtype=np.float64)
        lower = np.asarray(mid_series - band_half, dtype=np.float64)
        trend_ma = np.asarray(trend_series, dtype=np.float64)

        got = qe.run_mean_reversion_state_machine(
            close=close, mid=mid, upper=upper, lower=lower, trend_ma=trend_ma
        )
        ref = _python_mean_reversion(close, mid, upper, lower, trend_ma)
        np.testing.assert_array_equal(got, ref)

    def test_pairs_matches_python_reference(self) -> None:
        rng = np.random.default_rng(123)
        z = rng.normal(0.0, 1.5, 500)
        z[100] = 5.0
        z[200] = np.nan
        got = qe.run_pairs_state_machine(
            zscore=z,
            entry_zscore=ENTRY_Z,
            exit_zscore=EXIT_Z,
            stop_loss_zscore=STOP_LOSS_Z,
        )
        ref = _python_pairs(z, ENTRY_Z, EXIT_Z, STOP_LOSS_Z)
        np.testing.assert_array_equal(got, ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
