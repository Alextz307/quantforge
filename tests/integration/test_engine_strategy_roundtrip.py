"""
End-to-end roundtrip: AdaptiveBollinger train -> signals -> C++ engine -> metrics.

Verifies the CppBacktestEngine adapter plays nicely with an existing strategy.
Strategy correctness is owned by ``tests/integration/test_adaptive_bollinger.py``;
engine correctness by ``cpp/tests/test_backtest_engine.cpp``. This test only
asserts the seam holds: shapes line up, metrics are finite, no NaN propagation.

The trained strategy and its generated signals are cached at
module scope - fitting a GARCH grid (p_max=q_max=5) on 500 bars is the
slow step, and both tests in this file consume the same fit.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_engine import BacktestResult, MetricsCalculator
from src.core.types import Interval
from src.engine import SLIPPAGE_SCENARIOS, CppBacktestEngine, SlippageScenario
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from tests.conftest import make_synthetic_ohlcv_df

ROUNDTRIP_N_ROWS = 500
BOLLINGER_WINDOW = 20
BOLLINGER_K = 2.0
TREND_WINDOW = 100
ANNUALIZATION_DAILY = Interval.DAILY.annualization_factor()
# Large finite bound - Sharpe on synthetic random walks rarely exceeds +/-5.
SHARPE_PLAUSIBLE_BOUND = 10.0
EQUITY_TIE_TOL = 1e-9


@pytest.fixture(scope="module")
def roundtrip_inputs() -> tuple[pd.DataFrame, pd.Series]:
    """
    One-shot bars + signals (GARCH fit happens here, once per module).
    """

    bars = make_synthetic_ohlcv_df(n_rows=ROUNDTRIP_N_ROWS)
    strategy = AdaptiveBollingerStrategy(
        window=BOLLINGER_WINDOW,
        k=BOLLINGER_K,
        trend_window=TREND_WINDOW,
    )
    strategy.train(bars)
    signals = strategy.generate_signals(bars)
    return bars, signals


def test_full_pipeline_produces_finite_metrics(
    roundtrip_inputs: tuple[pd.DataFrame, pd.Series],
) -> None:
    bars, signals = roundtrip_inputs
    engine = CppBacktestEngine()
    result = engine.run(bars, signals, SLIPPAGE_SCENARIOS[SlippageScenario.NORMAL])

    assert len(result.equity_curve) == ROUNDTRIP_N_ROWS
    assert np.isfinite(result.equity_curve).all()
    assert math.isfinite(result.total_return)
    assert result.trade_count >= 0

    metrics = MetricsCalculator.compute(result.equity_curve, ANNUALIZATION_DAILY)
    assert math.isfinite(metrics.sharpe_ratio)
    assert math.isfinite(metrics.sortino_ratio)
    assert math.isfinite(metrics.max_drawdown)
    assert abs(metrics.sharpe_ratio) < SHARPE_PLAUSIBLE_BOUND


def test_run_scenarios_orders_results(
    roundtrip_inputs: tuple[pd.DataFrame, pd.Series],
) -> None:
    bars, signals = roundtrip_inputs
    engine = CppBacktestEngine()
    scenarios: list[BacktestResult] = engine.run_scenarios(
        bars,
        signals,
        [SLIPPAGE_SCENARIOS[s] for s in SlippageScenario],
    )

    assert len(scenarios) == len(SlippageScenario)
    # Higher slippage -> strictly non-better terminal equity (lower or equal,
    # equal only when no trades fire across the higher-slippage delta).
    for prev, curr in zip(scenarios[:-1], scenarios[1:], strict=True):
        assert curr.equity_curve[-1] <= prev.equity_curve[-1] + EQUITY_TIE_TOL
