"""
Walk-forward orchestrator: fold count, ordering, and determinism.

Uses a deterministic ``_FlatLongStrategy`` (no internal randomness) so
the only stochastic input is the synthetic OHLCV; two runs over the
same panel must produce bit-identical per-fold metrics.

Most tests share a single ``orchestrator_results`` module-scoped
fixture — running the full walk-forward once and asserting against the
shared list of folds is ~5× cheaper than re-running per assertion.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quant_engine import PerformanceMetrics
from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.engine import (
    SLIPPAGE_SCENARIOS,
    CppBacktestEngine,
    FoldResult,
    SlippageScenario,
    evaluate_walk_forward,
)
from src.strategies.interface import IStrategy
from tests.conftest import make_synthetic_ohlcv_df, make_walk_forward_validator

WF_N_ROWS = 600
WF_N_SPLITS = 3
WF_TEST_SIZE = 100
# Per-bar rate large enough to dominate the synthetic random walk's drift,
# guaranteeing rf-adjusted Sharpe is strictly less than rf=0 Sharpe.
DAILY_RISK_FREE_RATE = 0.001


class _FlatLongStrategy(IStrategy):
    """
    Always-long strategy: signal = 1.0 every bar after warmup.

    Trivial enough to be fully deterministic and to make hand-checking
    trivial. ``train()`` only records ``training_metadata`` so the
    orchestrator's tripwire passes.
    """

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:
        self._set_fitted_with_metadata(
            TrainingMetadata.from_fit(train_data, Interval.DAILY, ("close",))
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=data.index, name="flat_long_signal")

    @property
    def name(self) -> str:
        return "FlatLong"

    @property
    def required_warmup_bars(self) -> int:
        return 0

    @staticmethod
    def suggest_params(trial: object) -> dict[str, object]:
        return {}


@pytest.fixture(scope="module")
def wf_bars() -> pd.DataFrame:
    return make_synthetic_ohlcv_df(n_rows=WF_N_ROWS)


def _run_orchestrator(bars: pd.DataFrame, scenario: SlippageScenario) -> list[FoldResult]:
    return evaluate_walk_forward(
        _FlatLongStrategy(),
        bars,
        make_walk_forward_validator(WF_N_SPLITS, WF_TEST_SIZE),
        CppBacktestEngine(),
        SLIPPAGE_SCENARIOS[scenario],
        Interval.DAILY,
    )


@pytest.fixture(scope="module")
def orchestrator_results(wf_bars: pd.DataFrame) -> list[FoldResult]:
    return _run_orchestrator(wf_bars, SlippageScenario.NORMAL)


def test_fold_count_matches_validator(orchestrator_results: list[FoldResult]) -> None:
    assert len(orchestrator_results) == WF_N_SPLITS


def test_fold_indices_are_sequential(orchestrator_results: list[FoldResult]) -> None:
    assert [r.fold_index for r in orchestrator_results] == list(range(WF_N_SPLITS))


def test_folds_are_temporally_ordered(orchestrator_results: list[FoldResult]) -> None:
    """
    Each fold's test window starts strictly after the previous one's.
    """

    for prev, curr in zip(
        orchestrator_results[:-1],
        orchestrator_results[1:],
        strict=True,
    ):
        assert prev.test_end < curr.test_start


def test_train_strictly_precedes_test(orchestrator_results: list[FoldResult]) -> None:
    """
    The orchestrator's tripwire would fire otherwise — sanity check.
    """

    for r in orchestrator_results:
        assert r.train_end < r.test_start


def test_fold_result_carries_metrics(orchestrator_results: list[FoldResult]) -> None:
    for r in orchestrator_results:
        assert isinstance(r.metrics, PerformanceMetrics)
        assert r.backtest.trade_count >= 1


def test_two_runs_are_bit_identical(wf_bars: pd.DataFrame) -> None:
    """
    Determinism: same panel + same strategy → identical metrics.

    Intentionally NOT using ``orchestrator_results`` — needs two
    independent runs to verify reproducibility.
    """

    a = _run_orchestrator(wf_bars, SlippageScenario.NORMAL)
    b = _run_orchestrator(wf_bars, SlippageScenario.NORMAL)
    for ra, rb in zip(a, b, strict=True):
        assert ra.backtest.total_return == rb.backtest.total_return
        assert ra.backtest.trade_count == rb.backtest.trade_count
        assert ra.metrics.sharpe_ratio == rb.metrics.sharpe_ratio
        assert ra.metrics.max_drawdown == rb.metrics.max_drawdown


@pytest.mark.parametrize("scenario", list(SlippageScenario))
def test_orchestrator_runs_for_every_scenario(
    wf_bars: pd.DataFrame,
    scenario: SlippageScenario,
) -> None:
    """
    Each predefined scenario can drive a full walk-forward without errors.
    """

    results = _run_orchestrator(wf_bars, scenario)
    assert len(results) == WF_N_SPLITS


def test_risk_free_rate_lowers_sharpe(wf_bars: pd.DataFrame) -> None:
    """
    Non-zero rfr threads through MetricsCalculator and dampens Sharpe per fold.
    """

    baseline = evaluate_walk_forward(
        _FlatLongStrategy(),
        wf_bars,
        make_walk_forward_validator(WF_N_SPLITS, WF_TEST_SIZE),
        CppBacktestEngine(),
        SLIPPAGE_SCENARIOS[SlippageScenario.NORMAL],
        Interval.DAILY,
    )
    with_rfr = evaluate_walk_forward(
        _FlatLongStrategy(),
        wf_bars,
        make_walk_forward_validator(WF_N_SPLITS, WF_TEST_SIZE),
        CppBacktestEngine(),
        SLIPPAGE_SCENARIOS[SlippageScenario.NORMAL],
        Interval.DAILY,
        risk_free_rate=DAILY_RISK_FREE_RATE,
    )
    for base, rfr in zip(baseline, with_rfr, strict=True):
        assert rfr.metrics.sharpe_ratio < base.metrics.sharpe_ratio
        assert rfr.metrics.max_drawdown == base.metrics.max_drawdown
