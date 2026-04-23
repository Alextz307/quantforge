"""End-to-end CLI smoke test for ``python -m scripts.experiment run``.

Runs the full CLI stack against a tiny synthetic CSV fixture — exercises
the same code path the user invokes at the shell, modulo ``sys.argv``
plumbing. Opt-in (``RUN_EXP_SMOKE=1``) because even at 100 bars the GARCH
AIC grid search takes a few seconds.

The gated convention matches ``tests/integration/test_benchmark_cli_smoke``.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from scripts.experiment import cli
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_MANIFEST_JSON,
    EXPERIMENT_METRICS_JSON,
    EXPERIMENT_STRATEGY_SUBDIR,
    FOLD_RESULTS_JSONL,
)
from tests.conftest import make_synthetic_ohlcv_df

_TICKER = "MINI"
_N_ROWS = 300
_N_SPLITS = 2
_TEST_SIZE = 60
_GAP = 1
_HOLDOUT_PCT = 0.15

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EXP_SMOKE") != "1",
    reason="set RUN_EXP_SMOKE=1 to run the experiment CLI smoke test",
)


@pytest.fixture
def mini_experiment_fixture(tmp_path: Path) -> Path:
    """Write the synthetic CSV + the YAML config into ``tmp_path`` and return the YAML path."""
    csv_dir = tmp_path / "csv_data"
    csv_dir.mkdir()
    df = make_synthetic_ohlcv_df(n_rows=_N_ROWS, start="2020-01-02")
    df.index.name = "date"
    df.to_csv(csv_dir / f"{_TICKER}.csv")

    cfg_payload = {
        "name": "mini_smoke",
        "seed": 42,
        "data": {
            "source": {"name": "csv", "params": {"data_dir": str(csv_dir)}},
            "tickers": [_TICKER],
            "start": datetime(2020, 1, 2).isoformat(),
            "end": datetime(2022, 1, 1).isoformat(),
            "interval": "daily",
        },
        "strategy": {
            "name": "AdaptiveBollinger",
            "params": {
                "window": 20,
                "trend_window": 50,
                "garch_p_max": 1,
                "garch_q_max": 1,
            },
        },
        "validation": {
            "n_splits": _N_SPLITS,
            "test_size": _TEST_SIZE,
            "gap": _GAP,
            "holdout_pct": _HOLDOUT_PCT,
        },
        "slippage": {"scenario": "normal"},
    }
    yaml_path = tmp_path / "mini_experiment.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(cfg_payload, f)
    return yaml_path


def test_cli_run_produces_full_artifact_tree(
    mini_experiment_fixture: Path,
    tmp_path: Path,
) -> None:
    store = tmp_path / "experiment_results"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--config",
            str(mini_experiment_fixture),
            "--store-root",
            str(store),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "experiment_id:" in result.output

    runs_root = store / "runs"
    assert runs_root.is_dir()
    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for child in (
        EXPERIMENT_CONFIG_YAML,
        EXPERIMENT_MANIFEST_JSON,
        FOLD_RESULTS_JSONL,
        EXPERIMENT_METRICS_JSON,
        EXPERIMENT_STRATEGY_SUBDIR,
    ):
        assert (run_dir / child).exists(), f"missing artifact: {child}"
    # Reporter artifacts (default --report).
    assert (run_dir / "plots" / "equity_curves.png").is_file()
    assert (run_dir / "plots" / "fold_stability.png").is_file()
    assert (run_dir / "tables" / "metrics_summary.tex").is_file()


def test_cli_run_surfaces_invalid_config_error(
    tmp_path: Path,
) -> None:
    bad_yaml = tmp_path / "broken.yaml"
    bad_yaml.write_text("name: only_a_name\n")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--config", str(bad_yaml), "--store-root", str(tmp_path / "out")],
    )
    assert result.exit_code != 0
    assert "failed to load config" in result.output
