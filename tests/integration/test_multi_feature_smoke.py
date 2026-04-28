"""End-to-end smoke for the multi-feature single-asset dispatch path.

Runs in-process via ``Experiment.run()`` rather than the CLI subprocess
used by the pairs / single-asset smokes — there is no production
multi-feature strategy yet, so the test relies on
``MultiFeatureTestStub`` registered by ``tests/_strategy_stubs.py``.

Opt-in via ``RUN_EXP_SMOKE=1`` matches the run / pairs / holdout-eval
smokes — the full ``Experiment.run`` writes a half-megabyte artifact tree
even on tiny synthetic input.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from src.core.config import load_experiment_config
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    FOLD_RESULTS_JSONL,
    RUNS_SUBDIR,
    read_experiment_manifest,
)
from src.core.registry import data_source_registry
from src.data.fingerprint import fingerprint_multi_bars
from src.orchestration.builder import build_experiment
from src.orchestration.experiment import RunOptions, fetch_bars
from tests.conftest import make_multi_feature_mini_experiment_fixture

_PRIMARY = "MFA"
_FEATURE_TICKERS = ("MFB", "MFC", "MFD")
_N_SPLITS = 2
_STRATEGY_NAME = "_MultiFeatureTestStub"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EXP_SMOKE") != "1",
    reason="set RUN_EXP_SMOKE=1 to run the multi-feature experiment smoke test",
)


def test_multi_feature_run_produces_full_artifact_tree(tmp_path: Path) -> None:
    """Happy path: multi-feature run → wide-format hash + sliced engine PnL on disk."""
    config_path = make_multi_feature_mini_experiment_fixture(
        tmp_path,
        strategy_name=_STRATEGY_NAME,
        primary_ticker=_PRIMARY,
        feature_tickers=_FEATURE_TICKERS,
        n_splits=_N_SPLITS,
    )
    store = tmp_path / "experiment_results"

    cfg = load_experiment_config(config_path)
    experiment = build_experiment(cfg)
    experiment.run(options=RunOptions(store_root=store, write_report=False))

    runs_root = store / RUNS_SUBDIR
    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1, f"expected exactly one run dir, got {run_dirs}"
    run_dir = run_dirs[0]

    persisted_cfg = load_experiment_config(run_dir / EXPERIMENT_CONFIG_YAML)
    assert persisted_cfg.data.tickers == [_PRIMARY, *_FEATURE_TICKERS]

    # The manifest's data_hash must match what ``fingerprint_multi_bars``
    # produces on a fresh refetch — guards against silent regression where
    # the orchestrator picks the single-leg or pairs fingerprint helper for
    # an N-ticker frame.
    manifest = read_experiment_manifest(run_dir)
    source = data_source_registry.create_from_config(persisted_cfg.data.source)
    refetched = fetch_bars(source, persisted_cfg, experiment.strategy)
    assert manifest.data_hash == fingerprint_multi_bars(refetched, persisted_cfg.data.tickers)

    fold_lines = (run_dir / FOLD_RESULTS_JSONL).read_text().splitlines()
    assert len(fold_lines) == _N_SPLITS


def test_multi_feature_validator_rejects_primary_outside_tickers(tmp_path: Path) -> None:
    """A config with primary_ticker not in data.tickers must fail at
    build_experiment time, before any fetch."""
    config_path = make_multi_feature_mini_experiment_fixture(
        tmp_path,
        strategy_name=_STRATEGY_NAME,
        primary_ticker=_PRIMARY,
        feature_tickers=_FEATURE_TICKERS,
        n_splits=_N_SPLITS,
    )
    payload = yaml.safe_load(config_path.read_text())
    payload["strategy"]["params"]["primary_ticker"] = "NOT_LISTED"
    with config_path.open("w") as f:
        yaml.safe_dump(payload, f)

    cfg = load_experiment_config(config_path)
    with pytest.raises(ValueError, match="primary_ticker"):
        build_experiment(cfg)
