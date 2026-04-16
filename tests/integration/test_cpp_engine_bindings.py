"""Integration tests for the pybind11 bindings of BacktestEngine + MetricsCalculator.

The underlying C++ logic is exhaustively tested in
``cpp/tests/test_backtest_engine.cpp`` and ``cpp/tests/test_metrics.cpp``.
These tests verify the **binding layer**: numpy array marshalling,
argument validation, enum/struct round-trips, and that ``run_scenarios``
preserves order and matches per-scenario ``run()`` output.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

import quant_engine as qe
from src.core.types import Interval
from tests.conftest import (
    BAR_LADDER_BASE_PRICE,
    DAILY_FIXED_VOLUME,
    GLOBAL_NUMPY_SEED,
    SPY_RETURN_STD,
)

F64Array = npt.NDArray[np.float64]
I64Array = npt.NDArray[np.int64]

# ───── Engine constants ─────
INITIAL_CAPITAL = 10_000.0
TRANSACTION_FEE_RATE = 0.001  # 10 bps round-trip
FLAT_PRICE = BAR_LADDER_BASE_PRICE
FIXED_VOLUME = float(DAILY_FIXED_VOLUME)

# ───── Test harness constants ─────
N_BARS_MIN = 2  # minimum to trigger one fill (bar 0 primes, bar 1 executes)
N_BARS_SHORT = 3
N_BARS_LONG = 100
RETURN_VOL = SPY_RETURN_STD
SIGNAL_FLIP_PERIOD = 5
SIGNAL_FLIP_PERIOD_TIGHT = 3

# ───── Slippage scenarios (bps) ─────
BASE_BPS_LIGHT = 1.0
BASE_BPS_MEDIUM = 5.0
BASE_BPS_HEAVY = 20.0
BASE_BPS_EXTREME = 50.0
VOLUME_IMPACT_COEFF_HEAVY = 10.0

# ───── Metrics constants (mirror cpp/tests/test_metrics.cpp) ─────
ANNUALIZATION_DAILY = Interval.DAILY.annualization_factor()
ANNUALIZATION_TWO = 2
ANNUALIZATION_FOUR = 4
RF_HALF_PCT = 0.005
ABS_TOL = 1e-9
EXACT_TOL = 1e-12

DRAWDOWN_PEAK = 110.0
DRAWDOWN_TROUGH = 80.0

WIN_RATE_MIXED_EXPECTED = 0.5  # 2 positives out of 4 non-zero returns
SORTINO_MEAN_EXCESS = 0.052  # hand-derived in cpp SortinoHandComputed
SORTINO_DOWNSIDE_VAR = 0.00058  # ditto
# Equity doubles over one period with ann=2 → growth^ann - 1 = 2^2 - 1.
DOUBLED_ANN_RETURN_EXPECTED = (2.0**ANNUALIZATION_TWO) - 1.0

# ───── Synthetic-trade constants for hand calculations ─────
SINGLE_TRADE_EXPECTED_FINAL_EQUITY = INITIAL_CAPITAL * (1.0 - TRANSACTION_FEE_RATE)
SINGLE_TRADE_EXPECTED_RETURN = -TRANSACTION_FEE_RATE


def _flat_bars(
    n: int, price: float = FLAT_PRICE
) -> tuple[I64Array, F64Array, F64Array, F64Array, F64Array, F64Array]:
    ts = np.arange(n, dtype=np.int64)
    flat = np.full(n, price, dtype=np.float64)
    vol = np.full(n, FIXED_VOLUME, dtype=np.float64)
    return ts, flat.copy(), flat.copy(), flat.copy(), flat.copy(), vol


# ════════════════════════════════════════════════════════════════
# SlippageConfig / enum round-trip
# ════════════════════════════════════════════════════════════════


class TestSlippageConfig:
    def test_defaults(self) -> None:
        cfg = qe.SlippageConfig()
        assert cfg.model == qe.SlippageModel.Fixed
        assert cfg.base_bps == 1.0
        assert cfg.volume_impact_coeff == 0.0

    def test_kwargs_ctor(self) -> None:
        cfg = qe.SlippageConfig(
            model=qe.SlippageModel.VolumeScaled,
            base_bps=3.0,
            volume_impact_coeff=0.5,
        )
        assert cfg.model == qe.SlippageModel.VolumeScaled
        assert cfg.base_bps == 3.0
        assert cfg.volume_impact_coeff == 0.5

    def test_attrs_mutable(self) -> None:
        cfg = qe.SlippageConfig()
        cfg.base_bps = 7.5
        cfg.model = qe.SlippageModel.NoSlippage
        assert cfg.base_bps == 7.5
        assert cfg.model == qe.SlippageModel.NoSlippage

    def test_enum_members(self) -> None:
        assert {
            qe.SlippageModel.NoSlippage,
            qe.SlippageModel.Fixed,
            qe.SlippageModel.VolumeScaled,
        } == set(qe.SlippageModel.__members__.values())


# ════════════════════════════════════════════════════════════════
# BacktestEngine.run — numpy marshalling
# ════════════════════════════════════════════════════════════════


class TestBacktestEngineRun:
    def test_flat_signal_no_trades(self) -> None:
        """All-NaN signals → no fill ever → equity stays at initial_capital."""
        ts, o, h, lo, c, v = _flat_bars(N_BARS_LONG)
        sig = np.full(N_BARS_LONG, np.nan, dtype=np.float64)
        eng = qe.BacktestEngine(initial_capital=INITIAL_CAPITAL)
        result = eng.run(
            timestamps=ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=v,
            signals=sig,
            slippage=qe.SlippageConfig(model=qe.SlippageModel.NoSlippage),
        )
        assert result.trade_count == 0
        assert result.total_return == 0.0
        assert result.equity_curve.shape == (N_BARS_LONG,)
        assert np.all(result.equity_curve == INITIAL_CAPITAL)

    def test_equity_curve_is_numpy_float64(self) -> None:
        """equity_curve marshals back as a float64 1-D numpy array."""
        ts, o, h, lo, c, v = _flat_bars(N_BARS_SHORT)
        sig = np.full(N_BARS_SHORT, np.nan, dtype=np.float64)
        eng = qe.BacktestEngine()
        result = eng.run(
            timestamps=ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=v,
            signals=sig,
            slippage=qe.SlippageConfig(),
        )
        assert isinstance(result.equity_curve, np.ndarray)
        assert result.equity_curve.dtype == np.float64
        assert result.equity_curve.ndim == 1

    def test_hand_calculated_single_trade(self) -> None:
        """Zero-slip one-way trade on flat prices → equity = initial * (1 - fee)."""
        ts, o, h, lo, c, v = _flat_bars(N_BARS_MIN)
        sig = np.array([1.0, 1.0], dtype=np.float64)
        eng = qe.BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            transaction_fee_rate=TRANSACTION_FEE_RATE,
        )
        result = eng.run(
            timestamps=ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=v,
            signals=sig,
            slippage=qe.SlippageConfig(model=qe.SlippageModel.NoSlippage),
        )
        assert result.trade_count == 1
        assert math.isclose(
            float(result.equity_curve[-1]),
            SINGLE_TRADE_EXPECTED_FINAL_EQUITY,
            abs_tol=ABS_TOL,
        )
        assert math.isclose(result.total_return, SINGLE_TRADE_EXPECTED_RETURN, abs_tol=ABS_TOL)

    def test_accepts_float32_arrays_via_forcecast(self) -> None:
        """`c_style | forcecast` on the binding coerces f32 inputs to f64."""
        n = N_BARS_SHORT
        ts = np.arange(n, dtype=np.int64)
        flat32 = np.full(n, FLAT_PRICE, dtype=np.float32)
        vol32 = np.full(n, FIXED_VOLUME, dtype=np.float32)
        sig32 = np.full(n, np.nan, dtype=np.float32)
        eng = qe.BacktestEngine()
        result = eng.run(
            timestamps=ts,
            open=flat32,
            high=flat32,
            low=flat32,
            close=flat32,
            volume=vol32,
            signals=sig32,
            slippage=qe.SlippageConfig(),
        )
        assert result.equity_curve.shape == (n,)

    def test_mismatched_ohlcv_length_raises(self) -> None:
        ts = np.arange(N_BARS_SHORT, dtype=np.int64)
        short = np.full(N_BARS_SHORT - 1, FLAT_PRICE, dtype=np.float64)
        good = np.full(N_BARS_SHORT, FLAT_PRICE, dtype=np.float64)
        vol = np.full(N_BARS_SHORT, FIXED_VOLUME, dtype=np.float64)
        sig = np.full(N_BARS_SHORT, np.nan, dtype=np.float64)
        eng = qe.BacktestEngine()
        with pytest.raises(ValueError, match="same length"):
            eng.run(
                timestamps=ts,
                open=short,
                high=good,
                low=good,
                close=good,
                volume=vol,
                signals=sig,
                slippage=qe.SlippageConfig(),
            )

    def test_mismatched_signals_length_raises(self) -> None:
        ts, o, h, lo, c, v = _flat_bars(N_BARS_SHORT)
        sig = np.full(N_BARS_SHORT - 1, np.nan, dtype=np.float64)
        eng = qe.BacktestEngine()
        with pytest.raises(ValueError, match="signals"):
            eng.run(
                timestamps=ts,
                open=o,
                high=h,
                low=lo,
                close=c,
                volume=v,
                signals=sig,
                slippage=qe.SlippageConfig(),
            )

    def test_allow_short_false_clips_negative_signals(self) -> None:
        """allow_short=False clips -1 to 0 (no trade); allow_short=True opens a short."""
        ts, o, h, lo, c, v = _flat_bars(N_BARS_MIN)
        short_sig = np.array([-1.0, -1.0], dtype=np.float64)
        no_slip = qe.SlippageConfig(model=qe.SlippageModel.NoSlippage)
        eng_long_only = qe.BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            transaction_fee_rate=TRANSACTION_FEE_RATE,
            allow_short=False,
        )
        long_only = eng_long_only.run(
            timestamps=ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=v,
            signals=short_sig,
            slippage=no_slip,
        )
        assert long_only.trade_count == 0
        assert long_only.total_return == 0.0

        eng_short_ok = qe.BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            transaction_fee_rate=TRANSACTION_FEE_RATE,
            allow_short=True,
        )
        short_ok = eng_short_ok.run(
            timestamps=ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=v,
            signals=short_sig,
            slippage=no_slip,
        )
        assert short_ok.trade_count == 1

    def test_volume_scaled_slippage_propagates(self) -> None:
        """VolumeScaled with impact > 0 produces strictly worse fills than plain Fixed."""
        rng = np.random.default_rng(GLOBAL_NUMPY_SEED)
        n = N_BARS_LONG
        ts = np.arange(n, dtype=np.int64)
        close = np.cumprod(1.0 + rng.normal(0, RETURN_VOL, n)) * FLAT_PRICE
        # Tight volume so |qty|/volume matters — otherwise impact → 0.
        vol = np.full(n, INITIAL_CAPITAL / FLAT_PRICE, dtype=np.float64)
        sig = np.where((np.arange(n) // SIGNAL_FLIP_PERIOD_TIGHT) % 2 == 0, 1.0, -1.0).astype(
            np.float64
        )
        eng = qe.BacktestEngine()
        fixed_only, volume_scaled = eng.run_scenarios(
            timestamps=ts,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=vol,
            signals=sig,
            scenarios=[
                qe.SlippageConfig(model=qe.SlippageModel.Fixed, base_bps=BASE_BPS_LIGHT),
                qe.SlippageConfig(
                    model=qe.SlippageModel.VolumeScaled,
                    base_bps=BASE_BPS_LIGHT,
                    volume_impact_coeff=VOLUME_IMPACT_COEFF_HEAVY,
                ),
            ],
        )
        assert volume_scaled.total_return < fixed_only.total_return


# ════════════════════════════════════════════════════════════════
# BacktestEngine.run_scenarios
# ════════════════════════════════════════════════════════════════


class TestRunScenarios:
    @staticmethod
    def _alternating_sample() -> tuple[
        I64Array, F64Array, F64Array, F64Array, F64Array, F64Array, F64Array
    ]:
        rng = np.random.default_rng(GLOBAL_NUMPY_SEED)
        n = N_BARS_LONG
        ts = np.arange(n, dtype=np.int64)
        close = np.cumprod(1.0 + rng.normal(0, RETURN_VOL, n)) * FLAT_PRICE
        vol = np.full(n, FIXED_VOLUME, dtype=np.float64)
        # Flip signal every 5 bars so scenarios see plenty of trades
        sig = np.where((np.arange(n) // SIGNAL_FLIP_PERIOD) % 2 == 0, 1.0, -1.0).astype(np.float64)
        return ts, close, close, close, close, vol, sig

    def test_order_preserved_and_matches_individual_runs(self) -> None:
        """bulk[i] must equal a standalone run(..., slippage=scenarios[i])."""
        ts, o, h, lo, c, v, sig = self._alternating_sample()
        scenarios = [
            qe.SlippageConfig(model=qe.SlippageModel.NoSlippage),
            qe.SlippageConfig(model=qe.SlippageModel.Fixed, base_bps=BASE_BPS_MEDIUM),
            qe.SlippageConfig(model=qe.SlippageModel.Fixed, base_bps=BASE_BPS_HEAVY),
        ]
        eng = qe.BacktestEngine()
        bulk = eng.run_scenarios(
            timestamps=ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=v,
            signals=sig,
            scenarios=scenarios,
        )
        assert len(bulk) == len(scenarios)
        for i, s in enumerate(scenarios):
            single = eng.run(
                timestamps=ts,
                open=o,
                high=h,
                low=lo,
                close=c,
                volume=v,
                signals=sig,
                slippage=s,
            )
            np.testing.assert_allclose(bulk[i].equity_curve, single.equity_curve, rtol=0, atol=0)
            assert bulk[i].total_return == single.total_return
            assert bulk[i].trade_count == single.trade_count

    def test_higher_slippage_monotonically_lowers_return(self) -> None:
        """Heavier slippage on an actively-trading strategy can only hurt PnL."""
        ts, o, h, lo, c, v, sig = self._alternating_sample()
        eng = qe.BacktestEngine()
        results = eng.run_scenarios(
            timestamps=ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=v,
            signals=sig,
            scenarios=[
                qe.SlippageConfig(model=qe.SlippageModel.NoSlippage),
                qe.SlippageConfig(model=qe.SlippageModel.Fixed, base_bps=BASE_BPS_LIGHT),
                qe.SlippageConfig(model=qe.SlippageModel.Fixed, base_bps=BASE_BPS_EXTREME),
            ],
        )
        zero, normal, extreme = results
        assert extreme.total_return <= normal.total_return <= zero.total_return

    def test_empty_scenarios_list_returns_empty(self) -> None:
        ts, o, h, lo, c, v, sig = self._alternating_sample()
        eng = qe.BacktestEngine()
        results = eng.run_scenarios(
            timestamps=ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=v,
            signals=sig,
            scenarios=[],
        )
        assert results == []


# ════════════════════════════════════════════════════════════════
# MetricsCalculator — bit-for-bit alignment with C++ test values
# ════════════════════════════════════════════════════════════════


class TestMetricsCalculatorBindings:
    def test_max_drawdown_matches_cpp_reference(self) -> None:
        # Mirrors MaxDrawdownPeakToTrough in test_metrics.cpp.
        equity = np.array(
            [100.0, DRAWDOWN_PEAK, 90.0, 95.0, DRAWDOWN_TROUGH, 85.0],
            dtype=np.float64,
        )
        expected = (DRAWDOWN_TROUGH - DRAWDOWN_PEAK) / DRAWDOWN_PEAK
        assert math.isclose(qe.MetricsCalculator.max_drawdown(equity), expected, abs_tol=EXACT_TOL)

    def test_annualized_return_geometric(self) -> None:
        # Mirrors AnnualizedReturnGeometric: doubles over 1 period, ann=2.
        equity = np.array([100.0, 200.0], dtype=np.float64)
        assert math.isclose(
            qe.MetricsCalculator.annualized_return(equity, ANNUALIZATION_TWO),
            DOUBLED_ANN_RETURN_EXPECTED,
            abs_tol=EXACT_TOL,
        )

    def test_win_rate_mixed(self) -> None:
        # Mirrors WinRateMixed: 2 positives out of 4 non-zero returns.
        returns = np.array([0.1, -0.1, 0.0, 0.2, -0.05], dtype=np.float64)
        assert qe.MetricsCalculator.win_rate(returns) == WIN_RATE_MIXED_EXPECTED

    def test_sharpe_hand_computed(self) -> None:
        # Mirrors SharpeHandComputed.
        returns = np.array([0.01, 0.02, 0.03], dtype=np.float64)
        expected = (0.02 / 0.01) * math.sqrt(ANNUALIZATION_DAILY)
        assert math.isclose(
            qe.MetricsCalculator.sharpe_ratio(returns, ANNUALIZATION_DAILY),
            expected,
            abs_tol=ABS_TOL,
        )

    def test_sortino_hand_computed(self) -> None:
        # Mirrors SortinoHandComputed; mean/downside_var are hand-derived in the
        # C++ test and mirrored here via named constants, not repeated arithmetic.
        returns = np.array([0.1, -0.05, 0.2, -0.02, 0.03], dtype=np.float64)
        downside_std = math.sqrt(SORTINO_DOWNSIDE_VAR)
        expected = (SORTINO_MEAN_EXCESS / downside_std) * math.sqrt(ANNUALIZATION_DAILY)
        assert math.isclose(
            qe.MetricsCalculator.sortino_ratio(returns, ANNUALIZATION_DAILY),
            expected,
            abs_tol=ABS_TOL,
        )

    def test_annualized_volatility_scaling(self) -> None:
        # Mirrors AnnualizedVolatilityScalesBySqrtFactor (sample_std=0.01).
        returns = np.array([0.01, 0.02, 0.03], dtype=np.float64)
        expected = 0.01 * math.sqrt(ANNUALIZATION_FOUR)
        assert math.isclose(
            qe.MetricsCalculator.annualized_volatility(returns, ANNUALIZATION_FOUR),
            expected,
            abs_tol=ABS_TOL,
        )

    def test_compute_populates_all_fields_finite(self) -> None:
        # Same curve as ComputeMatchesIndividualMethods in test_metrics.cpp.
        equity = np.array(
            [100.0, 110.0, 90.0, 95.0, 80.0, 85.0, 120.0, 115.0],
            dtype=np.float64,
        )
        m = qe.MetricsCalculator.compute(equity, ANNUALIZATION_DAILY)
        for field in (
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
        ):
            value = getattr(m, field)
            assert isinstance(value, float)
            assert math.isfinite(value)

    def test_compute_risk_free_lowers_sharpe(self) -> None:
        # Mirrors ComputeRiskFreeLowersSharpeAndSortino.
        equity = np.array([100.0, 101.0, 102.5, 101.5, 103.0, 104.0], dtype=np.float64)
        zero = qe.MetricsCalculator.compute(equity, ANNUALIZATION_DAILY, 0.0)
        with_rf = qe.MetricsCalculator.compute(equity, ANNUALIZATION_DAILY, RF_HALF_PCT)
        assert zero.sharpe_ratio > 0.0
        assert with_rf.sharpe_ratio < zero.sharpe_ratio
        assert with_rf.sortino_ratio < zero.sortino_ratio
        # rf only affects risk-adjusted numerators, not cash-flow metrics.
        assert with_rf.max_drawdown == zero.max_drawdown
        assert with_rf.annualized_return == zero.annualized_return


# ════════════════════════════════════════════════════════════════
# Cross-binding: engine.equity_curve → MetricsCalculator.compute
# ════════════════════════════════════════════════════════════════


class TestEngineMetricsHandoff:
    def test_compute_accepts_engine_equity_curve(self) -> None:
        """engine.run().equity_curve feeds directly into MetricsCalculator.compute."""
        rng = np.random.default_rng(GLOBAL_NUMPY_SEED)
        n = N_BARS_LONG
        ts = np.arange(n, dtype=np.int64)
        close = np.cumprod(1.0 + rng.normal(0, RETURN_VOL, n)) * FLAT_PRICE
        vol = np.full(n, FIXED_VOLUME, dtype=np.float64)
        sig = np.where((np.arange(n) // SIGNAL_FLIP_PERIOD) % 2 == 0, 1.0, -1.0).astype(np.float64)
        eng = qe.BacktestEngine()
        result = eng.run(
            timestamps=ts,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=vol,
            signals=sig,
            slippage=qe.SlippageConfig(model=qe.SlippageModel.NoSlippage),
        )
        m = qe.MetricsCalculator.compute(result.equity_curve, ANNUALIZATION_DAILY)
        assert math.isfinite(m.sharpe_ratio)
        assert math.isfinite(m.max_drawdown)
        assert m.max_drawdown <= 0.0
