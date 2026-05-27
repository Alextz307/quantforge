"""End-to-end CLI smoke test for ``python -m scripts.experiment holdout-eval``.

Drives the full CLI stack against a tiny synthetic CSV fixture: first
runs ``experiment run`` to produce a source dev-run with a non-null
``manifest.holdout_start``, then runs ``experiment holdout-eval --run-dir
<source>`` and asserts the bundle layout, the source-of-truth
cross-check, and the boundary pinning.

Opt-in via ``RUN_EXP_SMOKE=1`` (matches ``test_experiment_run_smoke.py``)
because the GARCH AIC grid takes a few seconds even on a 300-bar fixture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from scripts.experiment import cli
from src.core.persistence import (
    EXPERIMENT_MANIFEST_JSON,
    HOLDOUT_EVAL_JSON,
    HOLDOUT_EVALS_SUBDIR,
    RUNS_SUBDIR,
)
from tests.conftest import make_mini_experiment_fixture

_TICKER = "MINI"
# Holdout-eval needs more bars than the run-smoke fixture's 15% baseline.
_HOLDOUT_PCT = 0.30

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EXP_SMOKE") != "1",
    reason="set RUN_EXP_SMOKE=1 to run the experiment CLI smoke test",
)


def _write_fixture(tmp_path: Path) -> Path:
    """Wrap the shared fixture factory with the holdout-eval-specific holdout_pct."""
    return make_mini_experiment_fixture(
        tmp_path, name="holdout_eval_smoke", holdout_pct=_HOLDOUT_PCT
    )


def _invoke_run(config_path: Path, store: Path) -> Path:
    """Run ``experiment run`` and return the produced run directory."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--config", str(config_path), "--store-root", str(store), "--no-report"],
    )
    assert result.exit_code == 0, result.output
    runs_root = store / RUNS_SUBDIR
    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1, f"expected exactly one run dir, got {run_dirs}"
    return run_dirs[0]


def test_holdout_eval_produces_full_artifact_tree(tmp_path: Path) -> None:
    """End-to-end happy path: dev run → holdout-eval → bundle on disk."""
    config_path = _write_fixture(tmp_path)
    store = tmp_path / "experiment_results"

    run_dir = _invoke_run(config_path, store)

    runner = CliRunner()
    he = runner.invoke(
        cli,
        [
            "holdout-eval",
            "--run-dir",
            str(run_dir),
            "--store-root",
            str(store),
        ],
    )
    assert he.exit_code == 0, he.output
    assert "holdout_start:" in he.output
    assert "sharpe:" in he.output

    bundle_dir = store / HOLDOUT_EVALS_SUBDIR / run_dir.name
    assert bundle_dir.is_dir()
    payload_path = bundle_dir / HOLDOUT_EVAL_JSON
    assert payload_path.is_file()
    payload = json.loads(payload_path.read_text())

    assert payload["is_holdout_eval"] is True
    assert payload["source_kind"] == "run"
    assert payload["source_id"] == run_dir.name

    source_manifest = json.loads((run_dir / EXPERIMENT_MANIFEST_JSON).read_text())
    assert payload["holdout_start"] == source_manifest["holdout_start"]
    assert payload["data_hash"] == source_manifest["data_hash"]

    metrics = payload["metrics"]
    for k in ("sharpe_ratio", "sortino_ratio", "max_drawdown", "total_return", "trade_count"):
        assert k in metrics
    assert isinstance(payload["equity_curve"], list)
    assert len(payload["equity_curve"]) == payload["n_holdout_bars"]

    assert (bundle_dir / "tables" / "holdout_metrics.tex").is_file()
    assert (bundle_dir / "plots" / "holdout_equity.png").is_file()
    assert (bundle_dir / "plots" / "holdout_equity.svg").is_file()


def test_holdout_eval_refuses_when_data_hash_drifts(tmp_path: Path) -> None:
    """Mutating the cached CSV between run and holdout-eval trips the data_hash check."""
    config_path = _write_fixture(tmp_path)
    store = tmp_path / "experiment_results"

    run_dir = _invoke_run(config_path, store)

    # Flip one close-price bar so the fingerprint shifts but the row count
    # + DatetimeIndex still load cleanly.
    csv_path = tmp_path / "csv_data" / f"{_TICKER}.csv"
    contents = csv_path.read_text().splitlines()
    # Row 50 columns: 0=date, 1=open, 2=high, 3=low, 4=close.
    parts = contents[50].split(",")
    parts[4] = str(float(parts[4]) + 1.5)
    contents[50] = ",".join(parts)
    csv_path.write_text("\n".join(contents) + "\n")

    runner = CliRunner()
    he = runner.invoke(
        cli,
        [
            "holdout-eval",
            "--run-dir",
            str(run_dir),
            "--store-root",
            str(store),
        ],
    )
    assert he.exit_code != 0
    assert "data_hash drift" in he.output or "leakage" in he.output.lower()


def test_holdout_eval_refuses_when_source_has_no_holdout(tmp_path: Path) -> None:
    """A dev run with holdout_pct=0 has manifest.holdout_start=None — the eval must refuse."""
    config_path = _write_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text())
    payload["validation"]["holdout_pct"] = 0.0
    config_path.write_text(yaml.safe_dump(payload))
    store = tmp_path / "experiment_results"

    run_dir = _invoke_run(config_path, store)

    runner = CliRunner()
    he = runner.invoke(
        cli,
        [
            "holdout-eval",
            "--run-dir",
            str(run_dir),
            "--store-root",
            str(store),
        ],
    )
    assert he.exit_code != 0
    assert "holdout_start=None" in he.output


def test_holdout_eval_requires_exactly_one_source(tmp_path: Path) -> None:
    """The CLI guard must reject zero sources or both sources at once."""
    runner = CliRunner()
    result_zero = runner.invoke(cli, ["holdout-eval", "--store-root", str(tmp_path / "out")])
    assert result_zero.exit_code != 0
    assert "exactly one" in result_zero.output.lower()
