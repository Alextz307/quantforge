"""End-to-end unit tests for :meth:`Experiment.run`.

These tests construct a CSV-source-backed ``Experiment`` against a tiny
synthetic bars fixture written to a tmp_path, so the run completes in
~1 second without hitting yfinance or ML leaves. The strategy is
``AdaptiveBollinger`` (minimal-ML, fast GARCH grid) so every invariant
tested here — manifest contents, fold count, experiment-id shape,
strategy-state artifact, holdout reservation — is exercised on real
components, not mocks.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from src.core import json_io
from src.core.config import ExperimentConfig
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_MANIFEST_JSON,
    EXPERIMENT_METRICS_JSON,
    EXPERIMENT_STRATEGY_SUBDIR,
    FOLD_RESULTS_JSONL,
)
from src.engine.scenarios import SlippageScenario
from src.orchestration.builder import build_experiment
from src.orchestration.experiment import RunOptions
from src.orchestration.manifest import Manifest
from src.orchestration.types import ExperimentResult
from tests.conftest import make_synthetic_ohlcv_df

_TICKER = "MINI"
_N_ROWS = 300
_START = datetime(2020, 1, 2)
_END = datetime(2022, 1, 1)
_BOLLINGER_WINDOW = 20
_BOLLINGER_TREND = 50
_GARCH_P = 1
_GARCH_Q = 1
_N_SPLITS = 2
_TEST_SIZE = 60
_GAP = 1
_HOLDOUT_PCT = 0.2
_SEED = 42
# {timestamp}_{strategy}_{sha7}_{8 hex chars}
_EXPERIMENT_ID_RE = re.compile(r"^\d{8}_\d{6}_AdaptiveBollinger_[a-f0-9]+_[a-f0-9]{8}$")


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    df = make_synthetic_ohlcv_df(n_rows=_N_ROWS, start="2020-01-02")
    df.index.name = "date"
    out = tmp_path / "csv_data"
    out.mkdir()
    df.to_csv(out / f"{_TICKER}.csv")
    return out


@pytest.fixture
def cfg_dict(csv_dir: Path) -> dict[str, Any]:
    return {
        "name": "mini_experiment",
        "seed": _SEED,
        "data": {
            "source": {"name": "csv", "params": {"data_dir": str(csv_dir)}},
            "tickers": [_TICKER],
            "start": _START,
            "end": _END,
            "interval": "daily",
        },
        "strategy": {
            "name": "AdaptiveBollinger",
            "params": {
                "window": _BOLLINGER_WINDOW,
                "trend_window": _BOLLINGER_TREND,
                "garch_p_max": _GARCH_P,
                "garch_q_max": _GARCH_Q,
            },
        },
        "validation": {
            "n_splits": _N_SPLITS,
            "test_size": _TEST_SIZE,
            "gap": _GAP,
            "holdout_pct": _HOLDOUT_PCT,
        },
        "slippage": {"scenario": "normal"},
    }


@pytest.fixture
def run_result(cfg_dict: dict[str, Any], tmp_path: Path) -> tuple[Path, ExperimentResult]:
    """Execute a single run once per test — GARCH fit is the slow step."""
    cfg = ExperimentConfig.model_validate(cfg_dict)
    exp = build_experiment(cfg)
    store = tmp_path / "experiment_results"
    result = exp.run(RunOptions(store_root=store))
    return store, result


class TestExperimentRun:
    def test_experiment_id_shape(
        self,
        run_result: tuple[Path, ExperimentResult],
    ) -> None:
        _, result = run_result
        assert _EXPERIMENT_ID_RE.match(result.experiment_id)

    def test_fold_count_matches_validator(
        self,
        run_result: tuple[Path, ExperimentResult],
    ) -> None:
        _, result = run_result
        assert len(result.folds) == _N_SPLITS

    def test_run_dir_layout(
        self,
        run_result: tuple[Path, ExperimentResult],
    ) -> None:
        store, result = run_result
        run_dir = store / "runs" / result.experiment_id
        assert run_dir.is_dir()
        assert (run_dir / EXPERIMENT_CONFIG_YAML).is_file()
        assert (run_dir / EXPERIMENT_MANIFEST_JSON).is_file()
        assert (run_dir / FOLD_RESULTS_JSONL).is_file()
        assert (run_dir / EXPERIMENT_METRICS_JSON).is_file()
        assert (run_dir / EXPERIMENT_STRATEGY_SUBDIR).is_dir()
        # write_report defaults True → reporter artifacts land alongside.
        assert (run_dir / "plots" / "equity_curves.png").is_file()
        assert (run_dir / "plots" / "fold_stability.png").is_file()
        assert (run_dir / "tables" / "metrics_summary.tex").is_file()

    def test_manifest_round_trips(
        self,
        run_result: tuple[Path, ExperimentResult],
    ) -> None:
        store, result = run_result
        run_dir = store / "runs" / result.experiment_id
        raw = json_io.read_dict(run_dir / EXPERIMENT_MANIFEST_JSON)
        manifest = Manifest.from_dict(raw)
        assert manifest.experiment_id == result.experiment_id
        assert manifest.seed == _SEED
        assert manifest.slippage_scenario == SlippageScenario.NORMAL
        assert manifest.holdout_start is not None
        assert len(manifest.data_hash) == 64  # SHA-256 hex

    def test_holdout_reservation_excludes_last_bars(
        self,
        run_result: tuple[Path, ExperimentResult],
    ) -> None:
        """Walk-forward never sees bars at or past holdout_start."""
        _, result = run_result
        boundary = result.manifest.holdout_start
        assert boundary is not None
        for fold in result.folds:
            assert fold.test_end < boundary

    def test_fold_jsonl_one_object_per_line(
        self,
        run_result: tuple[Path, ExperimentResult],
    ) -> None:
        store, result = run_result
        run_dir = store / "runs" / result.experiment_id
        lines = (run_dir / FOLD_RESULTS_JSONL).read_text().strip().splitlines()
        assert len(lines) == _N_SPLITS
        # Each line is valid JSON
        import json

        for line in lines:
            parsed = json.loads(line)
            assert "fold_index" in parsed
            assert "sharpe_ratio" in parsed

    def test_metrics_json_has_aggregates(
        self,
        run_result: tuple[Path, ExperimentResult],
    ) -> None:
        store, result = run_result
        run_dir = store / "runs" / result.experiment_id
        metrics = json_io.read_dict(run_dir / EXPERIMENT_METRICS_JSON)
        assert metrics["n_folds"] == _N_SPLITS
        for k in ("sharpe_mean", "sortino_mean", "calmar_mean", "max_drawdown_worst"):
            assert k in metrics

    def test_frozen_config_round_trips_through_model_validate(
        self,
        run_result: tuple[Path, ExperimentResult],
    ) -> None:
        store, result = run_result
        run_dir = store / "runs" / result.experiment_id
        import yaml

        with (run_dir / EXPERIMENT_CONFIG_YAML).open() as f:
            payload = yaml.safe_load(f)
        revived = ExperimentConfig.model_validate(payload)
        assert revived.name == "mini_experiment"
        assert revived.seed == _SEED


class TestNoReportFlag:
    def test_write_report_false_skips_reporter_artifacts(
        self,
        cfg_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        cfg = ExperimentConfig.model_validate(cfg_dict)
        exp = build_experiment(cfg)
        store = tmp_path / "experiment_results"
        result = exp.run(RunOptions(store_root=store, write_report=False))
        run_dir = store / "runs" / result.experiment_id
        # Core runtime artifacts still land.
        assert (run_dir / FOLD_RESULTS_JSONL).is_file()
        # Reporter artifacts do NOT.
        assert not (run_dir / "plots").exists()
        assert not (run_dir / "tables").exists()


class TestExperimentIdUniqueness:
    def test_two_runs_same_strategy_produce_different_ids(
        self,
        cfg_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Suffix disambiguates two invocations in the same second."""
        cfg = ExperimentConfig.model_validate(cfg_dict)
        store = tmp_path / "experiment_results"
        r1 = build_experiment(cfg).run(RunOptions(store_root=store))
        r2 = build_experiment(cfg).run(RunOptions(store_root=store))
        assert r1.experiment_id != r2.experiment_id


class TestMultiTickerRejected:
    def test_multi_ticker_raises_not_implemented(
        self,
        csv_dir: Path,
        cfg_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Pairs wiring is deferred; a two-ticker config must fail loudly."""
        # Write a second CSV so the config validator doesn't trip on an
        # unknown ticker before the runner even starts.
        df = make_synthetic_ohlcv_df(n_rows=_N_ROWS, start="2020-01-02", seed=99)
        df.index.name = "date"
        df.to_csv(csv_dir / "OTHER.csv")

        cfg_dict_copy = dict(cfg_dict)
        cfg_dict_copy["data"] = {
            **cfg_dict["data"],
            "tickers": [_TICKER, "OTHER"],
        }
        cfg = ExperimentConfig.model_validate(cfg_dict_copy)
        with pytest.raises(NotImplementedError, match="multi-ticker"):
            build_experiment(cfg).run(RunOptions(store_root=tmp_path / "experiment_results"))
