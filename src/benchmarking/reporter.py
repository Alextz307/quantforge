"""Plot + LaTeX-table generation for benchmark reports.

Uses the matplotlib ``Agg`` backend unconditionally so thesis-quality PNGs
render identically on macOS / Linux / CI. Tables go through
``pandas.to_latex`` with booktabs styling — no custom LaTeX assembly.

``generate_full_report`` materialises the complete artifact bundle under
``benchmark_results/reports/<timestamp>/``:

    scaling_<family>.png
    scaling_<family>.svg
    regression_vs_<baseline>.png
    components.png
    summary.tex
    regression.tex
    scaling.tex
"""

# ruff: noqa: I001, E402
# Import order is semantically load-bearing: ``src.visualization.plots`` pins
# matplotlib's Agg backend at import time, and that MUST run before
# ``matplotlib.pyplot`` is first imported in this process. Sorting these
# alphabetically would let pyplot initialise with the default (GUI) backend.
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from src.visualization.plots import FIGURE_DPI, FIGURE_HEIGHT_IN, FIGURE_WIDTH_IN

import matplotlib.pyplot as plt
import pandas as pd

from src.benchmarking.analyzer import BenchmarkAnalyzer
from src.benchmarking.comparator import BenchmarkComparator
from src.benchmarking.types import (
    BenchmarkResult,
    BenchmarkRun,
    ComparisonReport,
    ScalingAnalysis,
)
from src.core.fs import ensure_parent_dir
from src.visualization.latex import LATEX_FLOAT_FORMAT
from src.visualization.plots import save_png_and_svg


def _ns_per_item(result: BenchmarkResult) -> float:
    return result.real_time_ns / max(result.params.get("n", 1), 1)


class BenchmarkReporter:
    """Materialises plots + LaTeX tables from :class:`BenchmarkAnalyzer` output.

    Each public method emits one artifact (PNG + SVG for plots, ``.tex``
    for tables); :meth:`generate_full_report` chains them into a single
    timestamped bundle suitable for thesis appendix inclusion. Defaults
    to a fresh analyzer when none is injected, so callers can drive
    reports off ad-hoc :class:`BenchmarkRun` inputs without wiring.
    """

    def __init__(self, analyzer: BenchmarkAnalyzer | None = None) -> None:
        self._analyzer = analyzer if analyzer is not None else BenchmarkAnalyzer()

    def plot_scaling(self, scaling: ScalingAnalysis, out_path: Path) -> Path:
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)
        ax.loglog(scaling.sizes, scaling.times_ns, "o-", label="measured")
        ax.set_xlabel("input size (n)")
        ax.set_ylabel("time (ns)")
        ax.set_title(
            f"{scaling.family}: slope={scaling.slope:.2f}, "
            f"R²={scaling.r_squared:.3f} [{scaling.classification}]"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save_png_and_svg(fig, out_path)
        plt.close(fig)
        return out_path

    def plot_regression_comparison(self, report: ComparisonReport, out_path: Path) -> Path | None:
        if not report.reports:
            return None
        names = [r.name for r in report.reports]
        deltas = [r.pct_delta for r in report.reports]
        colors = [
            "tab:red" if r.is_regression else "tab:green" if r.is_improvement else "tab:gray"
            for r in report.reports
        ]
        fig, ax = plt.subplots(
            figsize=(FIGURE_WIDTH_IN, max(FIGURE_HEIGHT_IN, 0.25 * len(names))),
            dpi=FIGURE_DPI,
        )
        ax.barh(names, deltas, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("pct delta (current vs baseline)")
        ax.set_title(f"{report.current_run_id} vs {report.baseline_run_id}")
        fig.tight_layout()
        ensure_parent_dir(out_path)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_component_breakdown(self, run: BenchmarkRun, out_path: Path) -> Path | None:
        if not run.results:
            return None
        df = pd.DataFrame(
            {
                "family": [r.family for r in run.results],
                "ns_per_item": [_ns_per_item(r) for r in run.results],
            }
        )
        aggregated = df.groupby("family")["ns_per_item"].median().sort_values()
        families = list(aggregated.index)
        values = [float(v) for v in aggregated.values]
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)
        ax.barh(families, values)
        ax.set_xlabel("ns per item (median across sizes)")
        ax.set_title(f"component cost breakdown — run {run.run_id}")
        fig.tight_layout()
        ensure_parent_dir(out_path)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def generate_summary_table(self, run: BenchmarkRun, out_path: Path) -> Path:
        rows = [
            {
                "benchmark": r.name,
                "n": r.params.get("n", 0),
                "time_ns": r.real_time_ns,
                "ns_per_item": _ns_per_item(r),
                "items_per_s": r.items_per_second,
            }
            for r in run.results
        ]
        df = pd.DataFrame(rows)
        latex = df.to_latex(
            index=False,
            float_format=LATEX_FLOAT_FORMAT,
            caption=f"Benchmark summary — run {run.run_id}",
            label=f"tab:bench_summary_{run.run_id}",
        )
        ensure_parent_dir(out_path)
        out_path.write_text(latex, encoding="utf-8")
        return out_path

    def generate_regression_table(self, report: ComparisonReport, out_path: Path) -> Path:
        rows = [
            {
                "benchmark": r.name,
                "baseline_ns": r.baseline_mean_ns,
                "current_ns": r.current_mean_ns,
                "pct_delta": r.pct_delta,
                "z": r.z_score if math.isfinite(r.z_score) else float("nan"),
                "status": (
                    "REGRESSION"
                    if r.is_regression
                    else "IMPROVEMENT"
                    if r.is_improvement
                    else "neutral"
                ),
            }
            for r in report.reports
        ]
        df = pd.DataFrame(rows)

        latex = df.to_latex(
            index=False,
            float_format=LATEX_FLOAT_FORMAT,
            caption=(f"Regression report: {report.current_run_id} vs {report.baseline_run_id}"),
            label=f"tab:bench_regression_{report.current_run_id}",
        )

        ensure_parent_dir(out_path)
        out_path.write_text(latex, encoding="utf-8")
        return out_path

    def generate_full_report(
        self,
        run: BenchmarkRun,
        out_dir: Path,
        *,
        baseline: BenchmarkRun | None = None,
    ) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.plot_component_breakdown(run, out_dir / "components.png")
        self.generate_summary_table(run, out_dir / "summary.tex")

        by_family: dict[str, list[BenchmarkResult]] = defaultdict(list)
        for r in run.results:
            by_family[r.family].append(r)
        for family in sorted(by_family):
            members = by_family[family]
            if len({r.params.get("n", 0) for r in members}) < 2:
                continue
            scaling = self._analyzer.analyze_scaling(run, family)
            safe = family.replace("/", "_")
            self.plot_scaling(scaling, out_dir / f"scaling_{safe}.png")

        if baseline is not None:
            cmp = BenchmarkComparator(self._analyzer).compare(baseline, run)
            self.plot_regression_comparison(cmp, out_dir / "regression.png")
            self.generate_regression_table(cmp, out_dir / "regression.tex")
        return out_dir
