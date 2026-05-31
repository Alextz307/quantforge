"""
End-to-end CLI smoke test for ``python -m scripts.experiment importance``.

Builds a real MomentumGatekeeper run (the lightest feature-consuming strategy),
then drives the ``importance`` subcommand against it. The reproduction decision
is forced per test (its own logic is unit-tested in
``tests/unit/test_experiment_importance.py``) so the branch *mechanics* - where
the artifact lands, what stays untouched - are verified without depending on
XGBoost being bit-identical across two fits. Opt-in (``RUN_EXP_SMOKE=1``)
because a feature-consuming walk-forward takes a few seconds.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import scripts.experiment as experiment_cli
from scripts.experiment import cli
from src.core import json_io
from src.core.config import ExperimentConfig
from src.core.persistence import (
    EXPERIMENT_METRICS_JSON,
    FEATURE_IMPORTANCE_DIVERGED_JSON,
    FEATURE_IMPORTANCE_JSON,
    RUNS_SUBDIR,
)
from src.orchestration.builder import build_experiment
from src.orchestration.experiment import RunOptions
from src.orchestration.run_loader import load_experiment_config_from_run
from tests.conftest import make_synthetic_ohlcv_df

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EXP_SMOKE") != "1",
    reason="set RUN_EXP_SMOKE=1 to run the importance CLI smoke test",
)

# Single-thread XGBoost: avoids the macOS libomp hang and keeps the fit cheap.
os.environ.setdefault("OMP_NUM_THREADS", "1")

_TICKER = "MINI"
_N_ROWS = 800
_TEST_SIZE = 200  # > the ~126-bar roc_126 warmup so OOS folds have scorable rows
_DIVERGED_JOB_NAME = "job_diverged"


def _mg_config(csv_dir: Path) -> ExperimentConfig:
    payload: dict[str, Any] = {
        "name": "mg_importance",
        "seed": 42,
        "data": {
            "source": {"name": "csv", "params": {"data_dir": str(csv_dir)}},
            "tickers": [_TICKER],
            "start": datetime(2020, 1, 2),
            "end": datetime(2024, 1, 1),
            "interval": "daily",
        },
        "strategy": {
            "name": "MomentumGatekeeper",
            "params": {
                "ma_window": 20,
                "prob_threshold": 0.5,
                "n_estimators": 15,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "val_split_ratio": 0.2,
            },
        },
        "validation": {
            "n_splits": 2,
            "test_size": _TEST_SIZE,
            "gap": 5,
            "expanding": True,
            "holdout_pct": 0.1,
        },
        "slippage": {"scenario": "normal"},
    }
    return ExperimentConfig.model_validate(payload)


def _build_run(tmp_path: Path) -> tuple[Path, Path, str]:
    """
    Run a MomentumGatekeeper experiment (no importance) and return store/run/id.
    """

    csv_dir = tmp_path / "csv_data"
    csv_dir.mkdir()
    df = make_synthetic_ohlcv_df(n_rows=_N_ROWS, start="2020-01-02")
    df.index.name = "date"
    df.to_csv(csv_dir / f"{_TICKER}.csv")

    store = tmp_path / "experiment_results"
    result = build_experiment(_mg_config(csv_dir)).run(
        RunOptions(store_root=store, write_report=False)
    )
    run_dir = store / RUNS_SUBDIR / result.experiment_id
    assert not (run_dir / FEATURE_IMPORTANCE_JSON).exists()
    return store, run_dir, result.experiment_id


def test_importance_reproduced_backfills_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_dir, original_id = _build_run(tmp_path)
    metrics_before = json_io.read_dict(run_dir / EXPERIMENT_METRICS_JSON)

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(experiment_cli, "_metrics_reproduced", lambda *a, **k: True)
    monkeypatch.setattr(experiment_cli, "attribute_via_username", lambda **k: calls.append(k))

    result = CliRunner().invoke(
        cli, ["importance", "--run-dir", str(run_dir), "--store-root", str(store)]
    )

    assert result.exit_code == 0, result.output
    payload = json_io.read_dict(run_dir / FEATURE_IMPORTANCE_JSON)
    assert payload["recomputed"] is True
    assert payload["reproduced"] is True
    assert "source_run" not in payload
    # Metrics are byte-for-byte preserved and no second run was created.
    assert json_io.read_dict(run_dir / EXPERIMENT_METRICS_JSON) == metrics_before
    assert [d.name for d in (store / RUNS_SUBDIR).iterdir()] == [original_id]
    # A backfill leaves no divergence pointer.
    assert not (run_dir / FEATURE_IMPORTANCE_DIVERGED_JSON).exists()
    # A reproduced backfill writes into the original run, so it must not
    # re-attribute (and thereby silently claim) that run.
    assert calls == []


def test_importance_diverged_saves_new_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, run_dir, original_id = _build_run(tmp_path)
    metrics_before = json_io.read_dict(run_dir / EXPERIMENT_METRICS_JSON)

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(experiment_cli, "_metrics_reproduced", lambda *a, **k: False)
    monkeypatch.setattr(experiment_cli, "attribute_via_username", lambda **k: calls.append(k))

    result = CliRunner().invoke(
        cli,
        [
            "importance",
            "--run-dir",
            str(run_dir),
            "--store-root",
            str(store),
            "--name",
            _DIVERGED_JOB_NAME,
        ],
    )

    assert result.exit_code == 0, result.output
    # Original run is untouched: no importance attached, metrics unchanged.
    assert not (run_dir / FEATURE_IMPORTANCE_JSON).exists()
    assert json_io.read_dict(run_dir / EXPERIMENT_METRICS_JSON) == metrics_before

    new_dirs = [d for d in (store / RUNS_SUBDIR).iterdir() if d.name != original_id]
    assert len(new_dirs) == 1
    new_run = new_dirs[0]
    payload = json_io.read_dict(new_run / FEATURE_IMPORTANCE_JSON)
    assert payload["recomputed"] is True
    assert payload["reproduced"] is False
    assert payload["source_run"] == original_id
    # The re-run's name is the passed job id, so the webapp's manifest scan
    # resolves the job back to this new run.
    assert load_experiment_config_from_run(new_run).name == _DIVERGED_JOB_NAME
    # The original run records where its importance landed, so its detail page
    # can link to the new run persistently.
    pointer = json_io.read_dict(run_dir / FEATURE_IMPORTANCE_DIVERGED_JSON)
    assert pointer["diverged_run_id"] == new_run.name
    # The diverged re-run is a brand-new artifact, so it (and only it) is the
    # one attributed - never the original run.
    assert [c["experiment_id"] for c in calls] == [new_run.name]
