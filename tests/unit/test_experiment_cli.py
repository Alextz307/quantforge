"""Smoke tests for the ``scripts/experiment.py`` click CLI.

These exercise the subcommand glue (config-override, error wrapping,
tune-study artifact layout) that otherwise lacks coverage. The ``run``
subcommand is left out here — it needs a full walk-forward smoke under
the gated integration tests — but ``tune`` + the override helpers all
fit a fast unit-test shape.

Uses ``click.testing.CliRunner`` so no subprocess is spawned.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner

from scripts.experiment import (
    _apply_dotted_overrides,
    _override_experiment,
    cli,
)
from src.analysis.metrics_aggregator import AggregateStats
from src.core.config import ExperimentConfig
from src.optimization import tuner as tuner_mod
from src.optimization.checkpointing import BEST_CONFIG_YAML_NAME, TRIALS_JSONL_NAME
from src.orchestration.experiment import RunOptions
from tests.conftest import make_stub_aggregate_stats

_TICKER = "SYNTH"
_CONFIG_SEED = 42
_OVERRIDE_SEED = 99
_BOLLINGER_WINDOW = 10
_BOLLINGER_K = 2.0
_BOLLINGER_TREND = 20
_TUNE_N_TRIALS = 3
_TUNE_STUDY_NAME = "cli_tune_smoke"


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
    """Direct-unit test for the override helper that powers ``--name`` /
    ``--seed`` on the run + tune subcommands.
    """

    def test_override_experiment_preserves_unset_fields(self) -> None:
        """Only the requested fields change; unspecified overrides are
        no-op. Uses a minimal config that doesn't hit a registry to
        avoid conftest-level fixture needs.
        """
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
        renamed = _override_experiment(cfg, name="renamed", seed=_OVERRIDE_SEED)
        assert renamed.name == "renamed"
        assert renamed.seed == _OVERRIDE_SEED


class TestDottedOverride:
    """``--override key.path=value`` flow: helper round-trip + CLI integration."""

    @staticmethod
    def _minimal_cfg() -> ExperimentConfig:
        payload = {
            "name": "base",
            "seed": 1,
            "data": {
                "source": {"name": "csv", "params": {"data_dir": "/tmp"}},
                "tickers": ["SPY"],
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
        return ExperimentConfig.model_validate(payload)

    def test_helper_no_overrides_returns_same_object(self) -> None:
        cfg = self._minimal_cfg()
        assert _apply_dotted_overrides(cfg, ()) is cfg

    def test_helper_applies_overrides_via_pydantic_round_trip(self) -> None:
        cfg = self._minimal_cfg()
        out = _apply_dotted_overrides(cfg, ("data.tickers=[QQQ]", "seed=99"))
        assert out.data.tickers == ["QQQ"]
        assert out.seed == 99
        # Original untouched (round-trip semantics).
        assert cfg.data.tickers == ["SPY"]
        assert cfg.seed == 1

    def test_helper_bad_path_surfaces_clickexception(self) -> None:
        cfg = self._minimal_cfg()
        with pytest.raises(click.ClickException, match="--override failed"):
            _apply_dotted_overrides(cfg, ("dat.tickers=[QQQ]",))

    def test_helper_pydantic_violation_surfaces_clickexception(self) -> None:
        """A type-incompatible override (e.g. seed expects int, gets list)
        re-raises through pydantic and gets wrapped as a ClickException
        with a re-validation prefix.
        """
        cfg = self._minimal_cfg()
        with pytest.raises(click.ClickException, match="re-validation failed"):
            _apply_dotted_overrides(cfg, ("seed=[not, an, int]",))
