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
        # The comparison dir is still created (run_comparison mkdir'd it +
        # per-strategy runs/ landed there), but no tables/ or plots/ subtree.
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
