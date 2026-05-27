"""End-to-end CLI smoke test for ``python -m scripts.experiment run`` on a pair.

Exercises the multi-ticker pairs path through the orchestrator: two-leg
fetch, wide-format inner-join, ``fingerprint_pair_bars`` dispatch,
``CppBacktestEngine.run_pairs``, cointegration weights round-trip via
``strategy.save()``. The standalone ``test_experiment_run_smoke.py`` only
covers the single-asset path; this file fills the gap so a regression in
the pairs branch fails CI loudly.

Opt-in via ``RUN_EXP_SMOKE=1`` matches the run + holdout-eval smoke tests —
even on 300 synthetic bars, the cointegration ADF + walk-forward fold loop
takes a few seconds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from scripts.experiment import cli
from src.core.config import load_experiment_config
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_STRATEGY_SUBDIR,
    FOLD_RESULTS_JSONL,
    RUNS_SUBDIR,
    WEIGHTS_JSON,
    read_experiment_manifest,
)
from src.core.registry import data_source_registry
from src.data.fingerprint import fingerprint_pair_bars
from src.orchestration.builder import build_experiment
from src.orchestration.experiment import fetch_bars
from tests.conftest import make_pair_mini_experiment_fixture

_TICKER_A = "PAIR_A"
_TICKER_B = "PAIR_B"
_N_SPLITS = 2

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EXP_SMOKE") != "1",
    reason="set RUN_EXP_SMOKE=1 to run the experiment CLI smoke test",
)


def test_pairs_run_produces_full_artifact_tree(tmp_path: Path) -> None:
    """Happy path: pairs CLI run → wide-format hash + cointegration weights on disk."""

    config_path = make_pair_mini_experiment_fixture(tmp_path, n_splits=_N_SPLITS)
    store = tmp_path / "experiment_results"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--config", str(config_path), "--store-root", str(store), "--no-report"],
    )
    assert result.exit_code == 0, result.output

    runs_root = store / RUNS_SUBDIR
    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1, f"expected exactly one run dir, got {run_dirs}"
    run_dir = run_dirs[0]

    cfg = load_experiment_config(run_dir / EXPERIMENT_CONFIG_YAML)
    assert cfg.data.tickers == [_TICKER_A, _TICKER_B]

    # Catches a regression where the orchestrator picks the single-leg
    # ``fingerprint_bars`` for a pair frame — the resulting hash would no
    # longer be sensitive to leg B.
    manifest = read_experiment_manifest(run_dir)
    source = data_source_registry.create_from_config(cfg.data.source)
    refetched = fetch_bars(source, cfg, build_experiment(cfg).strategy)
    assert manifest.data_hash == fingerprint_pair_bars(refetched)

    # A zero hedge_ratio would mean the strategy never fit.
    weights_path = run_dir / EXPERIMENT_STRATEGY_SUBDIR / WEIGHTS_JSON
    assert weights_path.is_file()
    weights = json.loads(weights_path.read_text())
    assert isinstance(weights["hedge_ratio"], float)
    assert weights["hedge_ratio"] != 0.0
    assert weights["is_cointegrated"] is True

    fold_lines = (run_dir / FOLD_RESULTS_JSONL).read_text().splitlines()
    assert len(fold_lines) == _N_SPLITS
