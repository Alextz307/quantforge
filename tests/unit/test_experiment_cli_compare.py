"""CliRunner smoke tests for ``experiment compare``.

Monkeypatches ``build_experiment`` in the comparison module so the test
stays in CLI-glue territory (option parsing, config loading,
ComparisonReporter invocation, artifact layout under
``<store_root>/comparisons/<out_name>/``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from scripts.experiment import cli
from src.core.config import ExperimentConfig
from src.core.persistence import COMPARISONS_SUBDIR
from src.orchestration import comparison as comparison_mod
from src.orchestration.experiment import RunOptions
from src.orchestration.types import ExperimentResult
from tests.conftest import (
    comparison_curve_seed,
    make_log_return_equity_curve,
    make_stub_experiment_result,
    make_stub_fold_record,
)

_TICKER = "SPY"
_N_FOLDS = 3
_CURVE_LENGTH = 40


def _write_cfg(tmp_path: Path, name: str) -> Path:
    payload = {
        "name": name,
        "seed": 42,
        "data": {
            "source": "csv",
            "tickers": [_TICKER],
            "start": "2020-01-02",
            "end": "2024-01-01",
            "interval": "daily",
        },
        "strategy": {
            "name": "AdaptiveBollinger",
            "params": {"window": 20, "k": 2.0, "trend_window": 200},
        },
    }
    path = tmp_path / f"{name}.yaml"
    with path.open("w") as f:
        yaml.safe_dump(payload, f)
    return path


def _stub_result(name: str, sharpe: float) -> ExperimentResult:
    folds = tuple(
        make_stub_fold_record(
            i,
            sharpe=sharpe,
            equity_curve=make_log_return_equity_curve(
                sharpe, n=_CURVE_LENGTH, seed=comparison_curve_seed(name, i)
            ),
        )
        for i in range(_N_FOLDS)
    )
    return make_stub_experiment_result(name, folds=folds)


class _StubExperiment:
    def __init__(self, name: str, sharpe: float) -> None:
        self._name = name
        self._sharpe = sharpe

    def run(self, options: RunOptions | None = None) -> ExperimentResult:
        return _stub_result(self._name, self._sharpe)


@pytest.fixture
def patched_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map each config's ``name`` to a deterministic sharpe for the stub."""

    name_to_sharpe = {"Alpha": 1.2, "Bravo": 0.7, "Charlie": 1.8}

    def _fake_build(cfg: ExperimentConfig) -> _StubExperiment:
        return _StubExperiment(name=cfg.name, sharpe=name_to_sharpe[cfg.name])

    monkeypatch.setattr(comparison_mod, "build_experiment", _fake_build)


class TestCompareSubcommandSmoke:
    def test_writes_ranking_tex_and_manifest(self, tmp_path: Path, patched_build: None) -> None:
        cfg_a = _write_cfg(tmp_path, "Alpha")
        cfg_b = _write_cfg(tmp_path, "Bravo")
        store_root = tmp_path / "results"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "--config",
                str(cfg_a),
                "--config",
                str(cfg_b),
                "--out-name",
                "smoke",
                "--store-root",
                str(store_root),
                "--significance-test",
                "bootstrap",
            ],
        )
        assert result.exit_code == 0, result.output

        cmp_dir = store_root / COMPARISONS_SUBDIR / "smoke"
        assert (cmp_dir / "manifest.json").is_file()
        assert (cmp_dir / "tables" / "ranking.tex").is_file()
        assert (cmp_dir / "tables" / "pairwise_significance.tex").is_file()
        assert (cmp_dir / "plots" / "equity_overlay.png").is_file()

    def test_no_report_flag_skips_artifact_generation(
        self, tmp_path: Path, patched_build: None
    ) -> None:
        cfg_a = _write_cfg(tmp_path, "Alpha")
        cfg_b = _write_cfg(tmp_path, "Bravo")
        store_root = tmp_path / "results"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "--config",
                str(cfg_a),
                "--config",
                str(cfg_b),
                "--out-name",
                "no_report",
                "--store-root",
                str(store_root),
                "--no-report",
            ],
        )
        assert result.exit_code == 0, result.output

        cmp_dir = store_root / COMPARISONS_SUBDIR / "no_report"
        assert cmp_dir.is_dir()
        assert not (cmp_dir / "tables").exists()
        assert not (cmp_dir / "plots").exists()

    def test_single_config_rejected_by_cli(self, tmp_path: Path) -> None:
        cfg_a = _write_cfg(tmp_path, "Alpha")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "--config",
                str(cfg_a),
                "--out-name",
                "lonely",
                "--store-root",
                str(tmp_path / "results"),
            ],
        )
        assert result.exit_code != 0
        assert "at least 2" in result.output


def _materialise_run_dir(run_dir: Path, name: str, *, data_hash: str | None = None) -> None:
    """Persist a stub :class:`ExperimentResult` to disk in the canonical layout.

    Mirrors what :meth:`Experiment.run` writes — manifest.json and
    fold_results.jsonl — so the loader has real on-disk artifacts to
    rebuild from. ``data_hash`` defaults to the conftest stub hash;
    pass an alternate value to exercise the cross-run alignment guard.
    """

    import json as _json

    from src.core import json_io
    from src.core.persistence import EXPERIMENT_MANIFEST_JSON, FOLD_RESULTS_JSONL

    run_dir.mkdir(parents=True, exist_ok=True)
    result = _stub_result(name, sharpe={"Alpha": 1.2, "Bravo": 0.7}.get(name, 1.0))
    if data_hash is not None:
        from dataclasses import replace

        result = ExperimentResult(
            experiment_id=result.experiment_id,
            folds=result.folds,
            manifest=replace(result.manifest, data_hash=data_hash),
        )
    json_io.write(run_dir / EXPERIMENT_MANIFEST_JSON, result.manifest.to_dict())
    with (run_dir / FOLD_RESULTS_JSONL).open("w", encoding="utf-8") as f:
        for fold in result.folds:
            f.write(_json.dumps(fold.to_dict(), sort_keys=True))
            f.write("\n")


@pytest.fixture
def build_experiment_must_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """If --reuse-runs is wired correctly, no experiment is built/run."""

    def _boom(_cfg: ExperimentConfig) -> object:
        raise AssertionError("build_experiment must not be called when --reuse-runs is set")

    monkeypatch.setattr(comparison_mod, "build_experiment", _boom)


class TestCompareReuseRuns:
    def test_skips_retrain_and_writes_ranking(
        self, tmp_path: Path, build_experiment_must_not_run: None
    ) -> None:
        cfg_a = _write_cfg(tmp_path, "Alpha")
        cfg_b = _write_cfg(tmp_path, "Bravo")
        run_a = tmp_path / "prior_runs" / "alpha"
        run_b = tmp_path / "prior_runs" / "bravo"
        _materialise_run_dir(run_a, "Alpha")
        _materialise_run_dir(run_b, "Bravo")
        store_root = tmp_path / "results"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "--config",
                str(cfg_a),
                "--config",
                str(cfg_b),
                "--out-name",
                "reuse_smoke",
                "--store-root",
                str(store_root),
                "--reuse-runs",
                f"{run_a},{run_b}",
            ],
        )
        assert result.exit_code == 0, result.output

        cmp_dir = store_root / COMPARISONS_SUBDIR / "reuse_smoke"
        ranking_tex = (cmp_dir / "tables" / "ranking.tex").read_text()
        assert "Alpha" in ranking_tex
        assert "Bravo" in ranking_tex

    def test_count_mismatch_rejected(self, tmp_path: Path) -> None:
        cfg_a = _write_cfg(tmp_path, "Alpha")
        cfg_b = _write_cfg(tmp_path, "Bravo")
        run_a = tmp_path / "prior_runs" / "alpha"
        _materialise_run_dir(run_a, "Alpha")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "--config",
                str(cfg_a),
                "--config",
                str(cfg_b),
                "--out-name",
                "count_mismatch",
                "--store-root",
                str(tmp_path / "results"),
                "--reuse-runs",
                str(run_a),
            ],
        )
        assert result.exit_code != 0
        assert "1 path(s) but --config has 2" in result.output

    def test_data_hash_drift_rejected(
        self, tmp_path: Path, build_experiment_must_not_run: None
    ) -> None:
        cfg_a = _write_cfg(tmp_path, "Alpha")
        cfg_b = _write_cfg(tmp_path, "Bravo")
        run_a = tmp_path / "prior_runs" / "alpha"
        run_b = tmp_path / "prior_runs" / "bravo"
        _materialise_run_dir(run_a, "Alpha", data_hash="a" * 64)
        _materialise_run_dir(run_b, "Bravo", data_hash="b" * 64)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "--config",
                str(cfg_a),
                "--config",
                str(cfg_b),
                "--out-name",
                "drift",
                "--store-root",
                str(tmp_path / "results"),
                "--reuse-runs",
                f"{run_a},{run_b}",
            ],
        )
        assert result.exit_code != 0
        assert "data_hash" in result.output
