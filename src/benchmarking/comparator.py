"""Presentation glue over :mod:`src.benchmarking.analyzer`.

Translates the raw ``RegressionReport`` list from the analyzer into a
``ComparisonReport`` carrying both run IDs and an easy-access
``regressions`` / ``improvements`` partition.
"""

from __future__ import annotations

from src.benchmarking.analyzer import (
    DEFAULT_PCT_THRESHOLD,
    DEFAULT_Z_THRESHOLD,
    BenchmarkAnalyzer,
)
from src.benchmarking.types import BenchmarkRun, ComparisonReport


class BenchmarkComparator:
    def __init__(self, analyzer: BenchmarkAnalyzer | None = None) -> None:
        self._analyzer = analyzer if analyzer is not None else BenchmarkAnalyzer()

    def compare(
        self,
        baseline: BenchmarkRun,
        current: BenchmarkRun,
        *,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        pct_threshold: float = DEFAULT_PCT_THRESHOLD,
    ) -> ComparisonReport:
        reports = self._analyzer.detect_regressions(
            current,
            baseline,
            z_threshold=z_threshold,
            pct_threshold=pct_threshold,
        )
        return ComparisonReport(
            baseline_run_id=baseline.run_id,
            current_run_id=current.run_id,
            reports=tuple(reports),
        )

    def compare_configurations(
        self,
        reference_label: str,
        runs: dict[str, BenchmarkRun],
        *,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        pct_threshold: float = DEFAULT_PCT_THRESHOLD,
    ) -> dict[str, ComparisonReport]:
        """Compare every run in ``runs`` against ``runs[reference_label]``."""
        if reference_label not in runs:
            raise KeyError(
                f"reference_label {reference_label!r} not in runs; available: "
                f"{sorted(runs)}. Fix by passing one of the available labels "
                f"or by adding a run under the requested label first."
            )
        baseline = runs[reference_label]
        return {
            label: self.compare(baseline, run, z_threshold=z_threshold, pct_threshold=pct_threshold)
            for label, run in runs.items()
            if label != reference_label
        }
