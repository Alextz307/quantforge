"""Benchmarking orchestrator for the quant engine.

See :mod:`src.benchmarking.runner` for the CLI entrypoint driver
(:mod:`scripts.benchmark`). The package splits into several namespaces:

* :mod:`.types`      — frozen dataclasses + JSON schema.
* :mod:`.store`      — JSONL persistence under ``benchmark_results/``.
* :mod:`.runner`     — Google Benchmark subprocess + Python hybrid driver.
* :mod:`.analyzer`   — summary statistics, regression z-tests, scaling fits.
* :mod:`.reporter`   — matplotlib plots + LaTeX tables.
* :mod:`.comparator` — presentation glue over ``analyzer.detect_regressions``.
"""

from __future__ import annotations

from src.benchmarking.analyzer import BenchmarkAnalyzer
from src.benchmarking.comparator import BenchmarkComparator
from src.benchmarking.reporter import BenchmarkReporter
from src.benchmarking.runner import BenchmarkRunner
from src.benchmarking.store import BenchmarkStore
from src.benchmarking.types import (
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkStats,
    ComparisonReport,
    HardwareInfo,
    RegressionReport,
    ScalingAnalysis,
)

__all__ = [
    "BenchmarkAnalyzer",
    "BenchmarkComparator",
    "BenchmarkReporter",
    "BenchmarkResult",
    "BenchmarkRun",
    "BenchmarkRunner",
    "BenchmarkStats",
    "BenchmarkStore",
    "ComparisonReport",
    "HardwareInfo",
    "RegressionReport",
    "ScalingAnalysis",
]
