"""Comparator tests: pairwise compare + multi-label configuration sweep."""

from __future__ import annotations

import pytest

from src.benchmarking.comparator import BenchmarkComparator
from src.benchmarking.types import BenchmarkRun
from tests.conftest import make_benchmark_result, make_benchmark_run


def _run(run_id: str, ns: float) -> BenchmarkRun:
    return make_benchmark_run((make_benchmark_result("BM_RSI/10000", ns=ns),), run_id=run_id)


def test_compare_returns_run_ids_and_reports() -> None:
    baseline = _run("baseline", 100.0)
    current = _run("current", 130.0)
    report = BenchmarkComparator().compare(baseline, current)
    assert report.baseline_run_id == "baseline"
    assert report.current_run_id == "current"
    assert len(report.reports) == 1
    assert report.reports[0].is_regression


def test_compare_configurations_skips_reference_label() -> None:
    runs = {
        "ref": _run("ref", 100.0),
        "fast": _run("fast", 70.0),
        "slow": _run("slow", 130.0),
    }
    results = BenchmarkComparator().compare_configurations("ref", runs)
    assert set(results) == {"fast", "slow"}
    assert results["fast"].reports[0].is_improvement
    assert results["slow"].reports[0].is_regression


def test_compare_configurations_missing_reference_raises() -> None:
    runs = {"a": _run("a", 100.0)}
    with pytest.raises(KeyError, match="reference_label"):
        BenchmarkComparator().compare_configurations("does-not-exist", runs)
