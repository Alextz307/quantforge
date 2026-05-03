"""End-to-end smoke test for ``scripts/benchmark.py``.

Builds the quant_bench binary is a prerequisite; this test is gated by
``RUN_BENCH_SMOKE=1`` so CI does not have to build ``quant_bench`` before
running the full Python test suite. Local developers run it manually when
they want to verify the CLI works end-to-end.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from scripts.benchmark import cli
from tests.conftest import REPO_ROOT

RUN_GATE_ENV = "RUN_BENCH_SMOKE"
QUANT_BENCH = REPO_ROOT / "cpp" / "build" / "benchmarks" / "quant_bench"
SMOKE_TAG = "smoke"
SMOKE_FILTER = "BM_RSI/10000$"

pytestmark = pytest.mark.skipif(
    os.environ.get(RUN_GATE_ENV) != "1",
    reason=f"set {RUN_GATE_ENV}=1 to enable the benchmark CLI smoke test",
)


def test_run_command_produces_a_stored_run(tmp_path: Path) -> None:
    if not QUANT_BENCH.exists():
        pytest.skip(f"quant_bench not built at {QUANT_BENCH}")

    runner = CliRunner()
    store_root = tmp_path / "bench_out"
    result = runner.invoke(
        cli,
        [
            "run",
            "--tag",
            SMOKE_TAG,
            "--filter",
            SMOKE_FILTER,
            "--min-time",
            "0.01",
            "--no-report",
            "--store-root",
            str(store_root),
        ],
    )
    assert result.exit_code == 0, result.output

    runs_dir = store_root / "runs"
    jsonl_files = list(runs_dir.glob("*.jsonl"))
    assert jsonl_files, "expected at least one JSONL run file"


def test_latex_command_emits_valid_braces(tmp_path: Path) -> None:
    if not QUANT_BENCH.exists():
        pytest.skip(f"quant_bench not built at {QUANT_BENCH}")

    store_root = tmp_path / "bench_out"
    reports_root = tmp_path / "reports"

    runner = CliRunner()
    run_result = runner.invoke(
        cli,
        [
            "run",
            "--tag",
            SMOKE_TAG,
            "--filter",
            SMOKE_FILTER,
            "--min-time",
            "0.01",
            "--no-report",
            "--store-root",
            str(store_root),
        ],
    )
    assert run_result.exit_code == 0, run_result.output
    run_files = list((store_root / "runs").glob("*.jsonl"))
    assert run_files

    latex_result = runner.invoke(
        cli,
        ["latex", str(run_files[0]), "--out-dir", str(reports_root)],
    )
    assert latex_result.exit_code == 0, latex_result.output

    summary = (reports_root / "summary.tex").read_text(encoding="utf-8")
    assert summary.count("{") == summary.count("}")
