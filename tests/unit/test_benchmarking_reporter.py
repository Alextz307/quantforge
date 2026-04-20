"""Reporter tests: plots, LaTeX tables, and the full-report bundle."""

from __future__ import annotations

from pathlib import Path

from src.benchmarking.analyzer import BenchmarkAnalyzer
from src.benchmarking.reporter import BenchmarkReporter
from src.benchmarking.types import BenchmarkRun
from tests.conftest import make_benchmark_result, make_benchmark_run

SAMPLE_SIZES = (10_000, 100_000, 1_000_000)
LINEAR_TIMES_NS = (1.0e5, 1.0e6, 1.0e7)


def _linear_run(run_id: str = "rid", *, multiplier: float = 1.0) -> BenchmarkRun:
    return make_benchmark_run(
        tuple(
            make_benchmark_result(f"BM_RSI/{n}", n=n, ns=t * multiplier)
            for n, t in zip(SAMPLE_SIZES, LINEAR_TIMES_NS, strict=True)
        ),
        run_id=run_id,
    )


def test_plot_scaling_writes_png_and_svg(tmp_path: Path) -> None:
    run = _linear_run()
    scaling = BenchmarkAnalyzer().analyze_scaling(run, "BM_RSI")
    out_path = tmp_path / "scaling.png"
    result = BenchmarkReporter().plot_scaling(scaling, out_path)
    assert result == out_path
    assert out_path.exists()
    assert out_path.with_suffix(".svg").exists()
    assert out_path.stat().st_size > 0


def test_generate_summary_table_emits_valid_braces(tmp_path: Path) -> None:
    out_path = tmp_path / "summary.tex"
    BenchmarkReporter().generate_summary_table(_linear_run(), out_path)
    text = out_path.read_text(encoding="utf-8")
    assert text.count("{") == text.count("}")
    assert "BM_RSI/10000" in text


def test_generate_full_report_without_baseline(tmp_path: Path) -> None:
    BenchmarkReporter().generate_full_report(_linear_run(), tmp_path / "report")
    out = tmp_path / "report"
    assert (out / "components.png").exists()
    assert (out / "summary.tex").exists()
    assert (out / "scaling_BM_RSI.png").exists()
    assert not (out / "regression.png").exists()  # no baseline -> no regression plot


def test_generate_full_report_with_baseline_emits_regression_artifacts(tmp_path: Path) -> None:
    baseline = _linear_run(run_id="baseline")
    current = _linear_run(run_id="current", multiplier=1.3)
    BenchmarkReporter().generate_full_report(current, tmp_path / "report", baseline=baseline)
    out = tmp_path / "report"
    assert (out / "regression.png").exists()
    reg_text = (out / "regression.tex").read_text(encoding="utf-8")
    assert reg_text.count("{") == reg_text.count("}")
    assert "REGRESSION" in reg_text


def test_plot_regression_comparison_skips_when_no_shared_benchmarks(tmp_path: Path) -> None:
    baseline = _linear_run(run_id="baseline")
    disjoint = make_benchmark_run(
        (make_benchmark_result("BM_OTHER/1000", n=1000, ns=5.0e4),),
        run_id="disjoint",
    )
    BenchmarkReporter().generate_full_report(disjoint, tmp_path / "report", baseline=baseline)
    out = tmp_path / "report"
    # Disjoint universe -> no regression artefacts (empty-stub PNGs would be invalid).
    assert not (out / "regression.png").exists()
