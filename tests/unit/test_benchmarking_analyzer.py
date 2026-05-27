"""Analyzer tests: summary stats, regression z-gate, and scaling classification."""

from __future__ import annotations

from src.benchmarking.analyzer import (
    DEFAULT_PCT_THRESHOLD,
    DEFAULT_Z_THRESHOLD,
    BenchmarkAnalyzer,
)
from src.benchmarking.types import (
    SCALING_O_N,
    SCALING_O_N_SQUARED,
    SCALING_UNCLEAR,
)
from tests.conftest import make_benchmark_result, make_benchmark_run

TOLERANCE = 1e-9


def test_summarize_single_sample() -> None:
    stats = BenchmarkAnalyzer().summarize([make_benchmark_result("BM_RSI/10000", ns=50_000.0)])
    assert len(stats) == 1
    s = stats[0]
    assert s.mean_ns == 50_000.0
    assert s.std_ns == 0.0
    assert s.n_samples == 1
    assert s.ci95_low == s.ci95_high == 50_000.0


def test_summarize_groups_by_name() -> None:
    samples = [
        make_benchmark_result("BM_RSI/10000", ns=100.0),
        make_benchmark_result("BM_RSI/10000", ns=110.0),
        make_benchmark_result("BM_RSI/10000", ns=120.0),
    ]
    stats = BenchmarkAnalyzer().summarize(samples)
    assert len(stats) == 1
    assert stats[0].n_samples == 3
    assert abs(stats[0].mean_ns - 110.0) < TOLERANCE


def test_detect_regressions_flags_thirty_percent_slowdown() -> None:
    baseline = make_benchmark_run((make_benchmark_result("BM_RSI/10000", ns=100.0),))
    current = make_benchmark_run((make_benchmark_result("BM_RSI/10000", ns=130.0),))
    reports = BenchmarkAnalyzer().detect_regressions(current, baseline)
    assert len(reports) == 1
    r = reports[0]
    assert r.is_regression
    assert not r.is_improvement
    assert abs(r.pct_delta - 30.0) < TOLERANCE


def test_detect_regressions_flags_thirty_percent_speedup() -> None:
    baseline = make_benchmark_run((make_benchmark_result("BM_RSI/10000", ns=100.0),))
    current = make_benchmark_run((make_benchmark_result("BM_RSI/10000", ns=70.0),))
    reports = BenchmarkAnalyzer().detect_regressions(current, baseline)
    assert reports[0].is_improvement
    assert not reports[0].is_regression


def test_detect_regressions_skips_benchmarks_only_in_one_run() -> None:
    baseline = make_benchmark_run(
        (
            make_benchmark_result("BM_RSI/10000", ns=100.0),
            make_benchmark_result("BM_BASELINE_ONLY/10000", ns=200.0),
        )
    )
    current = make_benchmark_run(
        (
            make_benchmark_result("BM_RSI/10000", ns=110.0),
            make_benchmark_result("BM_CURRENT_ONLY/10000", ns=300.0),
        )
    )
    reports = BenchmarkAnalyzer().detect_regressions(current, baseline)
    assert [r.name for r in reports] == ["BM_RSI/10000"]


def test_detect_regressions_tiny_change_is_neutral() -> None:
    baseline = make_benchmark_run((make_benchmark_result("BM_RSI/10000", ns=100.0),))
    current = make_benchmark_run((make_benchmark_result("BM_RSI/10000", ns=102.0),))
    reports = BenchmarkAnalyzer().detect_regressions(
        current, baseline, pct_threshold=DEFAULT_PCT_THRESHOLD, z_threshold=DEFAULT_Z_THRESHOLD
    )
    r = reports[0]
    assert not r.is_regression
    assert not r.is_improvement


def test_analyze_scaling_identifies_linear() -> None:
    run = make_benchmark_run(
        (
            make_benchmark_result("BM_RSI/10000", n=10_000, ns=1.0e5),
            make_benchmark_result("BM_RSI/100000", n=100_000, ns=1.0e6),
            make_benchmark_result("BM_RSI/1000000", n=1_000_000, ns=1.0e7),
        )
    )
    scaling = BenchmarkAnalyzer().analyze_scaling(run, "BM_RSI")
    assert scaling.classification == SCALING_O_N
    assert scaling.r_squared > 0.999
    assert abs(scaling.slope - 1.0) < 0.05


def test_analyze_scaling_identifies_quadratic() -> None:
    run = make_benchmark_run(
        (
            make_benchmark_result("BM_Q/10", n=10, ns=100.0),
            make_benchmark_result("BM_Q/100", n=100, ns=10000.0),
            make_benchmark_result("BM_Q/1000", n=1000, ns=1_000_000.0),
        )
    )
    scaling = BenchmarkAnalyzer().analyze_scaling(run, "BM_Q")
    assert scaling.classification == SCALING_O_N_SQUARED
    assert abs(scaling.slope - 2.0) < 0.1


def test_analyze_scaling_unclear_on_noisy_data() -> None:
    run = make_benchmark_run(
        (
            make_benchmark_result("BM_X/10", n=10, ns=100.0),
            make_benchmark_result("BM_X/100", n=100, ns=5000.0),
            make_benchmark_result("BM_X/1000", n=1000, ns=200.0),
        )
    )
    scaling = BenchmarkAnalyzer().analyze_scaling(run, "BM_X")
    assert scaling.classification == SCALING_UNCLEAR
