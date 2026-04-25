"""Smoke tests for the ``scripts/experiment.py`` click CLI.

These exercise the ~140 lines of subcommand glue (config-override,
error wrapping, artifact-listing table rendering) that otherwise lacks
coverage. The ``run`` subcommand is left out here — it needs a full
walk-forward smoke under the gated integration tests — but
``train-model`` + ``list-models`` + the override helpers all fit a
fast unit-test shape.

Uses ``click.testing.CliRunner`` so no subprocess is spawned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from scripts.experiment import (
    _override_experiment,
    _override_standalone,
    cli,
)
from src.analysis.metrics_aggregator import AggregateStats
from src.core.config import ExperimentConfig, StandaloneModelConfig
from src.optimization import tuner as tuner_mod
from src.optimization.checkpointing import BEST_CONFIG_YAML_NAME, TRIALS_JSONL_NAME
from src.orchestration.experiment import RunOptions
from tests.conftest import (
    make_stub_aggregate_stats,
    make_synthetic_ohlcv_df,
    seed_globally,
)

_TICKER = "SYNTH"
_FEATURES: list[str] = ["feat_a", "feat_b"]
_N_ROWS = 160
_CSV_SEED = 7
_FEATURE_SEED = 13
_CONFIG_SEED = 42
_OVERRIDE_SEED = 99
_COMPACT_LSTM_HIDDEN_DIM = 8
_COMPACT_LSTM_LAYERS = 1
_COMPACT_LSTM_LOOKBACK = 5
_COMPACT_LSTM_EPOCHS = 2
_COMPACT_LSTM_BATCH = 8
_COMPACT_ARMA_P = 1
_COMPACT_ARMA_Q = 1
_BOLLINGER_WINDOW = 10
_BOLLINGER_K = 2.0
_BOLLINGER_TREND = 20
_TUNE_N_TRIALS = 3
_TUNE_STUDY_NAME = "cli_tune_smoke"


def _write_synth_csv(tmp_path: Path) -> Path:
    df = make_synthetic_ohlcv_df(n_rows=_N_ROWS, seed=_CSV_SEED)
    rng = np.random.default_rng(_FEATURE_SEED)
    for col in _FEATURES:
        df[col] = rng.normal(0.0, 1.0, len(df))
    csv = tmp_path / f"{_TICKER}.csv"
    df.to_csv(csv)
    return csv


def _write_standalone_config(tmp_path: Path, *, name: str = "cli_test") -> Path:
    csv = _write_synth_csv(tmp_path)
    payload = {
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
    cfg_path = tmp_path / "standalone.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(payload, f)
    return cfg_path


class TestTrainModelSubcommand:
    def test_creates_artifact_directory(self, tmp_path: Path) -> None:
        seed_globally()
        cfg_path = _write_standalone_config(tmp_path, name="cli_artifact")
        store_root = tmp_path / "results"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "train-model",
                "--config",
                str(cfg_path),
                "--store-root",
                str(store_root),
            ],
        )
        assert result.exit_code == 0, result.output
        artifact_dir = store_root / "models" / "cli_artifact"
        assert artifact_dir.is_dir()
        assert (artifact_dir / "manifest.json").is_file()
        assert (artifact_dir / "config.yaml").is_file()
        assert (artifact_dir / "weights").is_dir()
        assert "artifact:" in result.output
        assert "data_hash:" in result.output

    def test_missing_config_wraps_error(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "train-model",
                "--config",
                str(tmp_path / "nonexistent.yaml"),
                "--store-root",
                str(tmp_path / "results"),
            ],
        )
        assert result.exit_code != 0
        # click.Path(exists=True) produces its own error before our handler;
        # either way the subcommand must reject this non-zero.

    def test_existing_artifact_dir_rejected(self, tmp_path: Path) -> None:
        seed_globally()
        cfg_path = _write_standalone_config(tmp_path, name="dup")
        store_root = tmp_path / "results"
        runner = CliRunner()
        ok = runner.invoke(
            cli,
            ["train-model", "--config", str(cfg_path), "--store-root", str(store_root)],
        )
        assert ok.exit_code == 0, ok.output

        dup = runner.invoke(
            cli,
            ["train-model", "--config", str(cfg_path), "--store-root", str(store_root)],
        )
        assert dup.exit_code != 0
        assert "already exists" in dup.output


class TestListModelsSubcommand:
    def test_empty_when_no_models_dir(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["list-models", "--store-root", str(tmp_path / "results")])
        assert result.exit_code == 0
        assert "no models directory" in result.output

    def test_shows_header_and_row_for_trained_artifact(self, tmp_path: Path) -> None:
        seed_globally()
        cfg_path = _write_standalone_config(tmp_path, name="listed")
        store_root = tmp_path / "results"
        runner = CliRunner()
        train = runner.invoke(
            cli,
            ["train-model", "--config", str(cfg_path), "--store-root", str(store_root)],
        )
        assert train.exit_code == 0, train.output

        listing = runner.invoke(cli, ["list-models", "--store-root", str(store_root)])
        assert listing.exit_code == 0, listing.output
        assert "name" in listing.output
        assert "listed" in listing.output
        assert "hybrid_return" in listing.output
        assert "predictor" in listing.output

    def test_skips_entry_with_unreadable_manifest(self, tmp_path: Path) -> None:
        """A directory named like an artifact but with a corrupt manifest
        must not crash the listing — it should print a skip line and
        continue with the rest.
        """
        models_root = tmp_path / "results" / "models"
        broken = models_root / "broken"
        broken.mkdir(parents=True)
        (broken / "manifest.json").write_text("{not-json")

        runner = CliRunner()
        result = runner.invoke(cli, ["list-models", "--store-root", str(tmp_path / "results")])
        assert result.exit_code == 0
        assert "[skip] broken" in result.output


def _write_experiment_config(tmp_path: Path, *, name: str = "cli_tune") -> Path:
    """Write a minimal :class:`ExperimentConfig` YAML that the tuner
    accepts. Data source / walk-forward knobs are never actually touched —
    the tune test monkeypatches ``build_experiment`` + ``aggregate_folds``.
    """
    payload = {
        "name": name,
        "seed": _CONFIG_SEED,
        "data": {
            "source": {"name": "csv", "params": {"data_dir": str(tmp_path)}},
            "tickers": [_TICKER],
            "start": "2020-01-02",
            "end": "2025-12-31",
            "interval": "daily",
        },
        "strategy": {
            "name": "AdaptiveBollinger",
            "params": {
                "window": _BOLLINGER_WINDOW,
                "k": _BOLLINGER_K,
                "trend_window": _BOLLINGER_TREND,
            },
        },
    }
    cfg_path = tmp_path / "experiment.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(payload, f)
    return cfg_path


def _write_hpo_config(tmp_path: Path, *, study_name: str = _TUNE_STUDY_NAME) -> Path:
    payload = {
        "study_name": study_name,
        "n_trials": _TUNE_N_TRIALS,
        "sampler": "random",
        "objective": "sharpe",
        "seed": 1,
    }
    cfg_path = tmp_path / "hpo.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(payload, f)
    return cfg_path


class _StubExperimentResult:
    """Shape-minimal stand-in for ``ExperimentResult`` — matches the two
    attributes the tuner reads (``experiment_id`` + ``folds``).
    """

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        self.folds: tuple[object, ...] = ()
        self.manifest = None


class _StubExperiment:
    def __init__(self, experiment_id: str) -> None:
        self._experiment_id = experiment_id

    def run(self, options: RunOptions | None = None) -> _StubExperimentResult:
        return _StubExperimentResult(experiment_id=self._experiment_id)


class TestTuneSubcommand:
    """End-to-end CLI smoke for ``experiment tune``.

    Monkeypatches the per-trial ML work (``build_experiment`` +
    ``aggregate_folds``) so the test stays in CLI-glue territory:
    option parsing, CLI-to-StrategyTuner wiring, HPOReporter invocation,
    artifact layout under ``<store_root>/hpo/<study_name>/``.
    """

    def _patch_trial_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counter = {"n": 0}

        def _fake_build(cfg: ExperimentConfig) -> _StubExperiment:
            return _StubExperiment(experiment_id=f"cli_tune_exp_{counter['n']}")

        def _fake_aggregate(folds: tuple[object, ...]) -> AggregateStats:
            # Deterministic decreasing series so trial 0 wins — we only
            # care that best_config.yaml materialises.
            sharpe = 1.0 - counter["n"] * 0.1
            counter["n"] += 1
            return make_stub_aggregate_stats(sharpe=sharpe, total_return_mean=0.01)

        monkeypatch.setattr(tuner_mod, "build_experiment", _fake_build)
        monkeypatch.setattr(tuner_mod, "aggregate_folds", _fake_aggregate)

    def test_tune_smoke_produces_study_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_trial_work(monkeypatch)
        exp_cfg = _write_experiment_config(tmp_path)
        hpo_cfg = _write_hpo_config(tmp_path)
        store_root = tmp_path / "results"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "tune",
                "--config",
                str(exp_cfg),
                "--hpo-config",
                str(hpo_cfg),
                "--store-root",
                str(store_root),
            ],
        )
        assert result.exit_code == 0, result.output

        study_dir = store_root / "hpo" / _TUNE_STUDY_NAME
        assert (study_dir / "optuna_study.db").is_file()
        assert (study_dir / "experiment_config.yaml").is_file()
        assert (study_dir / "hpo_config.yaml").is_file()
        assert (study_dir / BEST_CONFIG_YAML_NAME).is_file()
        assert (study_dir / TRIALS_JSONL_NAME).is_file()
        # Reporter artifacts land under the same study dir.
        assert any(study_dir.rglob("convergence.*"))

        assert "best_value:" in result.output
        assert f"trials:      {_TUNE_N_TRIALS}" in result.output

    def test_tune_trials_override_applies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_trial_work(monkeypatch)
        exp_cfg = _write_experiment_config(tmp_path)
        hpo_cfg = _write_hpo_config(tmp_path)
        store_root = tmp_path / "results"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "tune",
                "--config",
                str(exp_cfg),
                "--hpo-config",
                str(hpo_cfg),
                "--store-root",
                str(store_root),
                "--trials",
                "2",
                "--no-report",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "trials:      2" in result.output

    def test_tune_rejects_invalid_n_jobs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_trial_work(monkeypatch)
        exp_cfg = _write_experiment_config(tmp_path)
        hpo_cfg = _write_hpo_config(tmp_path)
        store_root = tmp_path / "results"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "tune",
                "--config",
                str(exp_cfg),
                "--hpo-config",
                str(hpo_cfg),
                "--store-root",
                str(store_root),
                "--n-jobs",
                "-5",
            ],
        )
        assert result.exit_code != 0
        assert "must be -1 (auto) or a positive int" in result.output


class TestOverrideHelpers:
    """Direct-unit tests for the two override helpers that power ``--name``
    / ``--seed`` on both subcommands.
    """

    def test_override_standalone_updates_name_and_seed(self, tmp_path: Path) -> None:
        cfg_path = _write_standalone_config(tmp_path, name="orig")
        with cfg_path.open() as f:
            cfg = StandaloneModelConfig.model_validate(yaml.safe_load(f))
        overridden = _override_standalone(cfg, name="new", seed=_OVERRIDE_SEED)
        assert overridden.name == "new"
        assert overridden.seed == _OVERRIDE_SEED
        # Non-override fields preserved
        assert overridden.model.name == cfg.model.name

    def test_override_experiment_preserves_unset_fields(self) -> None:
        """Only the requested fields change; unspecified overrides are
        no-op. Uses a minimal config that doesn't hit a registry to
        avoid conftest-level fixture needs.
        """
        # Build minimal valid payload inline
        payload = {
            "name": "n",
            "seed": 1,
            "data": {
                "source": {"name": "csv", "params": {"data_dir": "/tmp"}},
                "tickers": ["X"],
                "start": "2020-01-01",
                "end": "2021-01-01",
                "interval": "daily",
            },
            "strategy": {
                "name": "AdaptiveBollinger",
                "params": {
                    "window": _BOLLINGER_WINDOW,
                    "k": _BOLLINGER_K,
                    "trend_window": _BOLLINGER_TREND,
                },
            },
        }
        cfg = ExperimentConfig.model_validate(payload)
        same = _override_experiment(cfg, name=None, seed=None)
        assert same.name == cfg.name
        assert same.seed == cfg.seed
        renamed = _override_experiment(cfg, name="renamed", seed=None)
        assert renamed.name == "renamed"
        assert renamed.seed == cfg.seed
