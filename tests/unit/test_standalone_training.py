"""End-to-end ``train_model_standalone`` + ``save_model_artifact`` round-trip.

Slower than the pure-logic tests (actually fits a compact HybridReturnModel)
but essential — covers the artifact format, the ``data_hash`` stability,
and the manifest provenance path. Uses a synthetic CSV so there's no
network / fixture-file dependency.

Also verifies ``load_model_artifact`` reconstructs a model whose
``training_metadata`` matches the pre-save state, which is what strategy
builders rely on when injecting via ``pretrained_leaves``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.config import StandaloneModelConfig
from src.core.types import Interval, ModelKind
from src.orchestration.model_artifact import load_model_artifact, save_model_artifact
from src.orchestration.standalone_training import train_model_standalone
from tests.conftest import (
    make_synthetic_ohlcv_df,
    seed_globally,
)

_TICKER = "SYNTH"
_FEATURES: list[str] = ["feat_a", "feat_b"]
_TRAIN_ROWS = 160
_CSV_SEED = 7
_FEATURE_NOISE_SEED = 13
_COMPACT_LSTM_HIDDEN_DIM = 8
_COMPACT_LSTM_LAYERS = 1
_COMPACT_LSTM_LOOKBACK = 5
_COMPACT_LSTM_EPOCHS = 2
_COMPACT_LSTM_BATCH = 8
_COMPACT_ARMA_P = 1
_COMPACT_ARMA_Q = 1
_CONFIG_SEED = 42


def _make_csv(tmp_path: Path) -> Path:
    """Produce a CSV fixture at tmp_path/SYNTH.csv with synthetic OHLCV + features."""
    df = make_synthetic_ohlcv_df(n_rows=_TRAIN_ROWS, seed=_CSV_SEED)
    rng = np.random.default_rng(_FEATURE_NOISE_SEED)
    for col in _FEATURES:
        df[col] = rng.normal(0.0, 1.0, len(df))
    csv = tmp_path / f"{_TICKER}.csv"
    df.to_csv(csv)
    return csv


def _build_config(tmp_path: Path, *, name: str = "test_hybrid_ret") -> StandaloneModelConfig:
    csv = _make_csv(tmp_path)
    return StandaloneModelConfig.model_validate(
        {
            "name": name,
            "seed": _CONFIG_SEED,
            "data": {
                "source": {"name": "csv", "params": {"data_dir": str(csv.parent)}},
                "tickers": [_TICKER],
                "start": "2020-01-02",
                "end": "2025-12-31",
                "interval": "daily",
            },
            "model": {
                "name": "hybrid_return",
                "params": {
                    "feature_columns": _FEATURES,
                    "arma_p_max": _COMPACT_ARMA_P,
                    "arma_q_max": _COMPACT_ARMA_Q,
                    "lstm_hidden_dim": _COMPACT_LSTM_HIDDEN_DIM,
                    "lstm_num_layers": _COMPACT_LSTM_LAYERS,
                    "lstm_lookback": _COMPACT_LSTM_LOOKBACK,
                    "lstm_epochs": _COMPACT_LSTM_EPOCHS,
                    "lstm_batch_size": _COMPACT_LSTM_BATCH,
                },
            },
            "model_kind": "predictor",
        }
    )


class TestTrainModelStandalone:
    def test_happy_path_returns_fitted_model_and_manifest(self, tmp_path: Path) -> None:
        seed_globally()
        cfg = _build_config(tmp_path)
        result = train_model_standalone(cfg)

        # Model has training_metadata populated by its own fit()
        assert result.model.training_metadata is not None
        assert result.model.training_metadata.interval == Interval.DAILY
        assert tuple(result.model.training_metadata.feature_columns) == tuple(_FEATURES)

        # Manifest carries the fields we rely on
        assert result.manifest.name == cfg.name
        assert result.manifest.model_name == "hybrid_return"
        assert result.manifest.model_kind == ModelKind.PREDICTOR
        assert result.manifest.seed == 42
        assert result.manifest.data_hash  # non-empty SHA

    def test_same_seed_produces_same_data_hash(self, tmp_path: Path) -> None:
        """Data hash is a SHA over raw OHLCV bytes; seed / model init don't
        touch it, so two runs on the same CSV produce identical hashes."""
        cfg = _build_config(tmp_path)
        seed_globally()
        r1 = train_model_standalone(cfg)
        seed_globally()
        r2 = train_model_standalone(cfg)
        assert r1.manifest.data_hash == r2.manifest.data_hash

    def test_unsupported_model_raises_not_implemented(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        cfg = StandaloneModelConfig.model_validate(
            {
                "name": "arma_only",
                "seed": 42,
                "data": {
                    "source": {"name": "csv", "params": {"data_dir": str(csv.parent)}},
                    "tickers": [_TICKER],
                    "start": "2020-01-02",
                    "end": "2025-12-31",
                    "interval": "daily",
                },
                "model": {"name": "arma", "params": {"p_max": 1, "q_max": 1}},
            }
        )
        with pytest.raises(NotImplementedError, match="not supported"):
            train_model_standalone(cfg)


class TestSaveLoadArtifactRoundTrip:
    def test_roundtrip_preserves_training_metadata(self, tmp_path: Path) -> None:
        seed_globally()
        cfg = _build_config(tmp_path, name="rt_test")
        result = train_model_standalone(cfg)

        artifact_dir = tmp_path / "artifact"
        save_model_artifact(artifact_dir, model=result.model, manifest=result.manifest, config=cfg)

        reloaded_model, reloaded_manifest = load_model_artifact(artifact_dir)

        # Manifest equal
        assert reloaded_manifest == result.manifest

        # Model training_metadata equal
        assert reloaded_model.training_metadata is not None
        assert reloaded_model.training_metadata == result.model.training_metadata

    def test_save_to_non_empty_dir_raises(self, tmp_path: Path) -> None:
        seed_globally()
        cfg = _build_config(tmp_path, name="rt_test")
        result = train_model_standalone(cfg)
        artifact_dir = tmp_path / "artifact"
        save_model_artifact(artifact_dir, model=result.model, manifest=result.manifest, config=cfg)
        with pytest.raises(FileExistsError, match="non-empty"):
            save_model_artifact(
                artifact_dir, model=result.model, manifest=result.manifest, config=cfg
            )

    def test_artifact_writes_yaml_config_alongside_manifest(self, tmp_path: Path) -> None:
        seed_globally()
        cfg = _build_config(tmp_path, name="rt_test")
        result = train_model_standalone(cfg)
        artifact_dir = tmp_path / "artifact"
        save_model_artifact(artifact_dir, model=result.model, manifest=result.manifest, config=cfg)
        assert (artifact_dir / "manifest.json").is_file()
        assert (artifact_dir / "config.yaml").is_file()
        # Frozen config YAML round-trips via StandaloneModelConfig.model_validate
        import yaml

        with (artifact_dir / "config.yaml").open() as f:
            reparsed = StandaloneModelConfig.model_validate(yaml.safe_load(f))
        assert reparsed.name == cfg.name
        assert reparsed.model.name == "hybrid_return"


class TestSliceAndHashDeterminism:
    def test_slice_respects_train_start_train_end(self, tmp_path: Path) -> None:
        """Fingerprint of the sliced window must differ from the full window."""
        seed_globally()
        cfg_full = _build_config(tmp_path, name="full")
        full_result = train_model_standalone(cfg_full)

        # Now re-run with a narrower train window → fingerprint must differ
        seed_globally()
        narrow_payload = cfg_full.model_dump(mode="json")
        narrow_payload["name"] = "narrow"
        narrow_payload["train_start"] = pd.Timestamp("2020-03-02").isoformat()
        narrow_payload["train_end"] = pd.Timestamp("2020-06-01").isoformat()
        cfg_narrow = StandaloneModelConfig.model_validate(narrow_payload)
        narrow_result = train_model_standalone(cfg_narrow)

        assert full_result.manifest.data_hash != narrow_result.manifest.data_hash
