"""
Behavioral tests for :mod:`src.analysis.baselines`.

Baseline computation routes through the C++ backtest engine; tests use
the same synthetic-OHLCV fixture the engine integration tests use so
the per-bar invariants (HLOC ordering, positive volume) hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.baselines import BaselineResult, compute_buy_and_hold
from src.core.types import Interval
from src.engine.cpp_engine import CppBacktestEngine
from src.engine.scenarios import SLIPPAGE_SCENARIOS, SlippageScenario
from tests.conftest import make_synthetic_ohlcv_df

_BAH_ROW_COUNT = 250
_BAH_SEED = 7
_BAH_BASE_PRICE = 100.0


class TestComputeBuyAndHold:
    def test_returns_baseline_result_with_full_metric_surface(self) -> None:
        bars = make_synthetic_ohlcv_df(
            n_rows=_BAH_ROW_COUNT, seed=_BAH_SEED, base_price=_BAH_BASE_PRICE
        )
        result = compute_buy_and_hold(
            bars,
            slippage=SLIPPAGE_SCENARIOS[SlippageScenario.ZERO],
            interval=Interval.DAILY,
            engine=CppBacktestEngine(),
        )
        assert isinstance(result, BaselineResult)
        assert len(result.equity_curve) == _BAH_ROW_COUNT
        assert np.isfinite(result.sharpe_ratio)
        assert np.isfinite(result.annualized_return)

    def test_zero_slippage_baseline_tracks_close_to_close_appreciation(self) -> None:
        """
        Under ZERO slippage, a held long position's total return
        should approximate ``close[-1] / close[0] - 1`` (with a small
        deviation from the engine's per-bar PnL accumulation vs. naive
        ratio).
        """

        bars = make_synthetic_ohlcv_df(
            n_rows=_BAH_ROW_COUNT, seed=_BAH_SEED, base_price=_BAH_BASE_PRICE
        )
        result = compute_buy_and_hold(
            bars,
            slippage=SLIPPAGE_SCENARIOS[SlippageScenario.ZERO],
            interval=Interval.DAILY,
            engine=CppBacktestEngine(),
        )
        naive_total = bars["close"].iloc[-1] / bars["close"].iloc[0] - 1.0
        assert result.total_return == pytest.approx(naive_total, abs=0.05)

    def test_higher_slippage_reduces_total_return(self) -> None:
        bars = make_synthetic_ohlcv_df(
            n_rows=_BAH_ROW_COUNT, seed=_BAH_SEED, base_price=_BAH_BASE_PRICE
        )
        zero = compute_buy_and_hold(
            bars,
            slippage=SLIPPAGE_SCENARIOS[SlippageScenario.ZERO],
            interval=Interval.DAILY,
            engine=CppBacktestEngine(),
        )
        high = compute_buy_and_hold(
            bars,
            slippage=SLIPPAGE_SCENARIOS[SlippageScenario.HIGH],
            interval=Interval.DAILY,
            engine=CppBacktestEngine(),
        )
        assert high.total_return <= zero.total_return

    def test_short_window_rejected(self) -> None:
        bars = make_synthetic_ohlcv_df(n_rows=1, seed=_BAH_SEED)
        with pytest.raises(ValueError, match="at least 2 bars"):
            compute_buy_and_hold(
                bars,
                slippage=SLIPPAGE_SCENARIOS[SlippageScenario.ZERO],
                interval=Interval.DAILY,
                engine=CppBacktestEngine(),
            )

    def test_round_trip_through_dict(self) -> None:
        bars = make_synthetic_ohlcv_df(
            n_rows=_BAH_ROW_COUNT, seed=_BAH_SEED, base_price=_BAH_BASE_PRICE
        )
        result = compute_buy_and_hold(
            bars,
            slippage=SLIPPAGE_SCENARIOS[SlippageScenario.ZERO],
            interval=Interval.DAILY,
            engine=CppBacktestEngine(),
        )
        restored = BaselineResult.from_dict(result.to_dict())
        assert restored == result
