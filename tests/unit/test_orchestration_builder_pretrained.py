"""Builder wires a pretrained leaf through to the strategy + manifest record.

Tests the ``build_experiment`` pretrained-leaf glue:

* A config with ``pretrained_leaves`` triggers ``load_model_artifact`` and
  passes the loaded model to the strategy ctor.
* The strategy ends up with ``is_pretrained=True`` entries in
  ``get_all_training_metadata()`` — verified post-train on the built
  experiment.
* The builder produces ``PretrainedLeafRecord`` instances on the
  Experiment object so ``Experiment.run()`` can stamp provenance into
  the manifest.

Full walk-forward runs are gated — this file stays focused on the
builder's glue.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.config import ExperimentConfig, StandaloneModelConfig
from src.orchestration.builder import build_experiment
from src.orchestration.model_artifact import save_model_artifact
from src.orchestration.standalone_training import train_model_standalone
from tests.conftest import (
    make_synthetic_ohlcv_df,
    seed_globally,
)

_TICKER = "SYNTH"
_FEATURES: list[str] = ["feat_a", "feat_b"]
_N_ROWS = 200
_CSV_SEED = 7
_FEATURE_NOISE_SEED = 13
_SAMPLE_ROWS = 80
_SAMPLE_OHLCV_SEED = 11
_SAMPLE_FEATURE_SEED = 19
_LEAF_GAP_DAYS = 365
_CONFIG_SEED = 42
_POSITION_SCALE = 15.0
_COMPACT_LSTM_LOOKBACK = 5
_COMPACT_MODEL_PARAMS = {
    "feature_columns": _FEATURES,
    "arma_p_max": 1,
    "arma_q_max": 1,
    "lstm_hidden_dim": 8,
    "lstm_num_layers": 1,
    "lstm_lookback": _COMPACT_LSTM_LOOKBACK,
    "lstm_epochs": 2,
    "lstm_batch_size": 8,
}


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    df = make_synthetic_ohlcv_df(n_rows=_N_ROWS, seed=_CSV_SEED)
    rng = np.random.default_rng(_FEATURE_NOISE_SEED)
    for col in _FEATURES:
        df[col] = rng.normal(0.0, 1.0, len(df))
    (tmp_path / f"{_TICKER}.csv").write_text(df.to_csv())
    return tmp_path


@pytest.fixture
def pretrained_artifact(csv_dir: Path, tmp_path: Path) -> Path:
    """Train a compact HybridReturnModel standalone, save, return its dir."""
    seed_globally()
    cfg = StandaloneModelConfig.model_validate(
        {
            "name": "synth_hybrid_ret",
            "seed": _CONFIG_SEED,
            "data": {
                "source": {"name": "csv", "params": {"data_dir": str(csv_dir)}},
                "tickers": [_TICKER],
                "start": "2020-01-02",
                "end": "2025-12-31",
                "interval": "daily",
            },
            "model": {"name": "hybrid_return", "params": _COMPACT_MODEL_PARAMS},
        }
    )
    trained = train_model_standalone(cfg)
    artifact_dir = tmp_path / "artifact"
    save_model_artifact(artifact_dir, model=trained.model, manifest=trained.manifest, config=cfg)
    return artifact_dir


def _experiment_payload(csv_dir: Path, artifact_dir: Path) -> dict[str, object]:
    return {
        "name": "exp",
        "seed": _CONFIG_SEED,
        "data": {
            "source": {"name": "csv", "params": {"data_dir": str(csv_dir)}},
            "tickers": [_TICKER],
            "start": "2020-01-02",
            "end": "2025-12-31",
            "interval": "daily",
        },
        "strategy": {
            "name": "ReturnForecast",
            "params": {
                "feature_columns": _FEATURES,
                "lstm_lookback": _COMPACT_LSTM_LOOKBACK,
                "position_scale": _POSITION_SCALE,
            },
        },
        "pretrained_leaves": {"return_model": str(artifact_dir)},
    }


class TestBuilderLoadsPretrainedLeaves:
    def test_builder_injects_loaded_leaf_into_strategy(
        self, csv_dir: Path, pretrained_artifact: Path
    ) -> None:
        payload = _experiment_payload(csv_dir, pretrained_artifact)
        cfg = ExperimentConfig.model_validate(payload)
        experiment = build_experiment(cfg)
        # Strategy has the loaded leaf — ``_pretrained_leaves`` is a per-
        # subclass convention (see the ctor notes on each concrete
        # strategy), not declared on IStrategy itself.
        pretrained_map = getattr(experiment.strategy, "_pretrained_leaves", {})
        assert "return_model" in pretrained_map

    def test_builder_produces_pretrained_leaf_records(
        self, csv_dir: Path, pretrained_artifact: Path
    ) -> None:
        payload = _experiment_payload(csv_dir, pretrained_artifact)
        cfg = ExperimentConfig.model_validate(payload)
        experiment = build_experiment(cfg)
        records = experiment.pretrained_leaf_records
        assert len(records) == 1
        record = records[0]
        assert record.key == "return_model"
        assert record.path == str(pretrained_artifact)
        assert record.data_hash  # non-empty
        assert isinstance(record.train_end, pd.Timestamp)

    def test_no_pretrained_leaves_produces_empty_records(self, csv_dir: Path) -> None:
        payload: dict[str, object] = {
            "name": "exp",
            "seed": _CONFIG_SEED,
            "data": {
                "source": {"name": "csv", "params": {"data_dir": str(csv_dir)}},
                "tickers": [_TICKER],
                "start": "2020-01-02",
                "end": "2025-12-31",
                "interval": "daily",
            },
            "strategy": {
                "name": "ReturnForecast",
                "params": {
                    "feature_columns": _FEATURES,
                    "lstm_lookback": _COMPACT_LSTM_LOOKBACK,
                },
            },
        }
        cfg = ExperimentConfig.model_validate(payload)
        experiment = build_experiment(cfg)
        assert experiment.pretrained_leaf_records == ()

    def test_strategy_injection_marks_leaf_metadata_pretrained(
        self, csv_dir: Path, pretrained_artifact: Path
    ) -> None:
        payload = _experiment_payload(csv_dir, pretrained_artifact)
        cfg = ExperimentConfig.model_validate(payload)
        experiment = build_experiment(cfg)
        # After post-ctor the strategy isn't trained yet; training_metadata is
        # None on the strategy. But the leaf has training_metadata already
        # (from the artifact). Train on a sample window then confirm.
        sample = make_synthetic_ohlcv_df(n_rows=_SAMPLE_ROWS, seed=_SAMPLE_OHLCV_SEED)
        rng = np.random.default_rng(_SAMPLE_FEATURE_SEED)
        for col in _FEATURES:
            sample[col] = rng.normal(0.0, 1.0, len(sample))
        # Shift sample to AFTER the leaf's train_end so strict-overlap passes
        leaf_end = experiment.pretrained_leaf_records[0].train_end
        sample.index = pd.date_range(
            leaf_end + pd.Timedelta(days=_LEAF_GAP_DAYS), periods=len(sample), freq="B"
        )
        experiment.strategy.train(sample)
        tracked = experiment.strategy.get_all_training_metadata()
        leaf_entries = tracked[1:]
        assert len(leaf_entries) > 0
        assert all(t.is_pretrained for t in leaf_entries)
