"""Round-trip tests for benchmarking dataclass ``to_dict`` / ``from_dict``."""

from __future__ import annotations

from src.benchmarking.types import (
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkStats,
    ComparisonReport,
    HardwareInfo,
    RegressionReport,
    ScalingAnalysis,
)
from tests.conftest import make_benchmark_hardware, make_benchmark_result

SAMPLE_NAME = "BM_RSI/10000"
SAMPLE_ITEMS_PER_S = 2.0e8
RUN_ID = "2026-04-20T12-00-00Z_abc1234"
TIMESTAMP = "2026-04-20T12:00:00Z"


def test_hardware_info_round_trip() -> None:
    hw = make_benchmark_hardware(git_sha="abc1234def5678")
    assert HardwareInfo.from_dict(hw.to_dict()) == hw


def test_benchmark_result_round_trip_preserves_counters() -> None:
    r = make_benchmark_result(
        SAMPLE_NAME,
        items_per_second=SAMPLE_ITEMS_PER_S,
        custom_counters={"Cycles": 180000.0, "IPC": 2.0},
        tags=("indicator", "rsi"),
    )
    assert BenchmarkResult.from_dict(r.to_dict()) == r


def test_benchmark_run_round_trip() -> None:
    run = BenchmarkRun(
        run_id=RUN_ID,
        timestamp=TIMESTAMP,
        tags=("pre-optimization", "smoke"),
        results=(make_benchmark_result(SAMPLE_NAME),),
        hardware=make_benchmark_hardware(git_sha="abc1234"),
    )
    assert BenchmarkRun.from_dict(run.to_dict()) == run


def test_benchmark_stats_round_trip() -> None:
    s = BenchmarkStats(
        name=SAMPLE_NAME,
        mean_ns=50_000.0,
        std_ns=1_000.0,
        p5_ns=48_000.0,
        p95_ns=52_000.0,
        ci95_low=49_000.0,
        ci95_high=51_000.0,
        n_samples=10,
    )
    assert BenchmarkStats.from_dict(s.to_dict()) == s


def test_regression_report_round_trip() -> None:
    r = RegressionReport(
        name=SAMPLE_NAME,
        baseline_mean_ns=50_000.0,
        current_mean_ns=60_000.0,
        pct_delta=20.0,
        z_score=3.5,
        is_regression=True,
        is_improvement=False,
    )
    assert RegressionReport.from_dict(r.to_dict()) == r


def test_scaling_analysis_round_trip() -> None:
    s = ScalingAnalysis(
        family="BM_RSI",
        sizes=(10000, 100000, 1000000),
        times_ns=(5.0e4, 5.0e5, 5.0e6),
        slope=1.0,
        intercept=-1.6,
        r_squared=0.999,
        classification="O(n)",
    )
    assert ScalingAnalysis.from_dict(s.to_dict()) == s


def test_comparison_report_round_trip_and_partitions() -> None:
    reports = (
        RegressionReport(
            name="a",
            baseline_mean_ns=100.0,
            current_mean_ns=120.0,
            pct_delta=20.0,
            z_score=3.0,
            is_regression=True,
            is_improvement=False,
        ),
        RegressionReport(
            name="b",
            baseline_mean_ns=100.0,
            current_mean_ns=80.0,
            pct_delta=-20.0,
            z_score=-3.0,
            is_regression=False,
            is_improvement=True,
        ),
    )
    cmp = ComparisonReport(baseline_run_id="baseline", current_run_id="current", reports=reports)
    assert ComparisonReport.from_dict(cmp.to_dict()) == cmp
    assert cmp.regressions == (reports[0],)
    assert cmp.improvements == (reports[1],)
