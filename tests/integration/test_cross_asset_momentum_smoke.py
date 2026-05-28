"""
End-to-end smoke for CrossAssetMomentumStrategy via the multi-feature path.

Runs in-process via ``Experiment.run()`` (mirrors
``test_multi_feature_smoke.py``'s pattern) — the strategy is the first
production exemplar of the multi-feature dispatch path. Asserts the
manifest's ``data_hash`` matches ``fingerprint_multi_bars`` (catches a
silent regression where the orchestrator routes to the wrong fingerprint
helper for an N-ticker frame), the classifier artifact landed under the
strategy's save tree, and the walk-forward fold count matches the spec.

Opt-in via ``RUN_EXP_SMOKE=1`` matches the run / pairs / multi-feature
smokes — even on tiny synthetic data the full ``Experiment.run`` writes a
half-megabyte artifact tree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.core.config import load_experiment_config
from src.core.persistence import (
    CLASSIFIER_SUBDIR,
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_STRATEGY_SUBDIR,
    FOLD_RESULTS_JSONL,
    MODEL_UBJ,
    RUNS_SUBDIR,
    read_experiment_manifest,
)
from src.core.registry import data_source_registry
from src.data.fingerprint import fingerprint_multi_bars
from src.orchestration.builder import build_experiment
from src.orchestration.experiment import RunOptions, fetch_bars
from tests.conftest import make_multi_feature_mini_experiment_fixture

_PRIMARY = "CAM_A"
_FEATURE_TICKERS: tuple[str, ...] = ("CAM_B", "CAM_C", "CAM_D")
_LAGS: list[int] = [1, 3, 7]
_DIRECTION_THRESHOLD = 0.55
_N_SPLITS = 2
_STRATEGY_NAME = "CrossAssetMomentum"

# Compact params: a fitted booster is enough; no need for a well-trained one.
_COMPACT_N_ESTIMATORS = 10
_COMPACT_MAX_DEPTH = 2

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EXP_SMOKE") != "1",
    reason="set RUN_EXP_SMOKE=1 to run the cross-asset momentum smoke test",
)


def test_cross_asset_momentum_run_produces_full_artifact_tree(tmp_path: Path) -> None:
    """
    Happy path: production multi-feature run → wide-format hash + classifier on disk.
    """

    config_path = make_multi_feature_mini_experiment_fixture(
        tmp_path,
        strategy_name=_STRATEGY_NAME,
        primary_ticker=_PRIMARY,
        feature_tickers=_FEATURE_TICKERS,
        n_splits=_N_SPLITS,
        extra_strategy_params={
            "lags": _LAGS,
            "direction_threshold": _DIRECTION_THRESHOLD,
            "n_estimators": _COMPACT_N_ESTIMATORS,
            "max_depth": _COMPACT_MAX_DEPTH,
        },
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
    assert persisted_cfg.strategy.name == _STRATEGY_NAME

    # Guard against the orchestrator picking single-leg or pairs fingerprint
    # for an N-ticker frame.
    manifest = read_experiment_manifest(run_dir)
    source = data_source_registry.create_from_config(persisted_cfg.data.source)
    refetched = fetch_bars(source, persisted_cfg, experiment.strategy)
    assert manifest.data_hash == fingerprint_multi_bars(refetched, persisted_cfg.data.tickers)

    classifier_artifact = run_dir / EXPERIMENT_STRATEGY_SUBDIR / CLASSIFIER_SUBDIR / MODEL_UBJ
    assert classifier_artifact.is_file()

    fold_lines = (run_dir / FOLD_RESULTS_JSONL).read_text().splitlines()
    assert len(fold_lines) == _N_SPLITS
