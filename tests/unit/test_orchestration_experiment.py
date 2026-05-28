"""
End-to-end unit tests for :meth:`Experiment.run`.

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

import pandas as pd
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
    """
    Execute a single run once per test — GARCH fit is the slow step.
    """

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
        assert (run_dir / "plots" / "equity_curves.png").is_file()
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
        assert len(manifest.data_hash) == 64

    def test_holdout_reservation_excludes_last_bars(
        self,
        run_result: tuple[Path, ExperimentResult],
    ) -> None:
        """
        Walk-forward never sees bars at or past holdout_start.
        """

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
        assert (run_dir / FOLD_RESULTS_JSONL).is_file()
        assert not (run_dir / "plots").exists()
        assert not (run_dir / "tables").exists()


class TestExperimentIdUniqueness:
    def test_two_runs_same_strategy_produce_different_ids(
        self,
        cfg_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """
        Suffix disambiguates two invocations in the same second.
        """

        cfg = ExperimentConfig.model_validate(cfg_dict)
        store = tmp_path / "experiment_results"
        r1 = build_experiment(cfg).run(RunOptions(store_root=store))
        r2 = build_experiment(cfg).run(RunOptions(store_root=store))
        assert r1.experiment_id != r2.experiment_id


class TestTickerCountValidation:
    def test_single_asset_strategy_rejects_two_tickers(
        self,
        csv_dir: Path,
        cfg_dict: dict[str, Any],
        tmp_path: Path,  # noqa: ARG002
    ) -> None:
        """
        Single-asset strategy + two tickers must fail at build time.
        """

        df = make_synthetic_ohlcv_df(n_rows=_N_ROWS, start="2020-01-02", seed=99)
        df.index.name = "date"
        df.to_csv(csv_dir / "OTHER.csv")

        cfg_dict_copy = dict(cfg_dict)
        cfg_dict_copy["data"] = {
            **cfg_dict["data"],
            "tickers": [_TICKER, "OTHER"],
        }
        cfg = ExperimentConfig.model_validate(cfg_dict_copy)
        with pytest.raises(ValueError, match="single-asset"):
            build_experiment(cfg)

    def test_three_tickers_rejected_at_fetch(
        self,
        csv_dir: Path,
        cfg_dict: dict[str, Any],
        tmp_path: Path,  # noqa: ARG002
    ) -> None:
        """
        Three-ticker configs are not in scope for any registered strategy.
        """

        for extra in ("OTHER", "THIRD"):
            df = make_synthetic_ohlcv_df(n_rows=_N_ROWS, start="2020-01-02", seed=99)
            df.index.name = "date"
            df.to_csv(csv_dir / f"{extra}.csv")

        cfg_dict_copy = dict(cfg_dict)
        cfg_dict_copy["data"] = {
            **cfg_dict["data"],
            "tickers": [_TICKER, "OTHER", "THIRD"],
        }
        cfg = ExperimentConfig.model_validate(cfg_dict_copy)
        with pytest.raises(ValueError, match="single-asset"):
            build_experiment(cfg)


_PAIRS_TICKER_A = "PAIRA"
_PAIRS_TICKER_B = "PAIRB"
_PAIRS_N_ROWS = 600
_PAIRS_END = datetime(2022, 6, 1)
_PAIRS_TEST_SIZE = 80
_PAIRS_HEDGE_TRUE = 0.85


@pytest.fixture
def pairs_csv_dir(tmp_path: Path) -> Path:
    """
    Two synthetic cointegrated tickers written as CSV for the pairs run.

    Leg A is GBM-style geometric noise; leg B = ``hedge * A + spread`` where
    ``spread`` is a mean-reverting OU process. The Engle-Granger test only
    needs cointegration to *exist*, not to be high-quality — these fixtures
    pass the default p-value threshold (0.05) on every seed we've tried.
    """

    import numpy as np

    rng = np.random.default_rng(seed=2026)
    n = _PAIRS_N_ROWS
    log_a = np.cumsum(rng.normal(0.0001, 0.012, size=n))
    price_a = 100.0 * np.exp(log_a)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.85 * spread[i - 1] + rng.normal(0.0, 0.5)
    price_b = (price_a * _PAIRS_HEDGE_TRUE) + spread + 50.0

    out = tmp_path / "pairs_csv"
    out.mkdir()
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    for ticker, prices in ((_PAIRS_TICKER_A, price_a), (_PAIRS_TICKER_B, price_b)):
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.005,
                "low": prices * 0.995,
                "close": prices,
                "volume": 1_000_000.0,
            },
            index=pd.Index(dates, name="date"),
        )
        df.to_csv(out / f"{ticker}.csv")
    return out


class TestPairsExperimentEndToEnd:
    def test_pairs_walk_forward_runs_to_completion(
        self,
        pairs_csv_dir: Path,
        tmp_path: Path,
    ) -> None:
        """
        A PairsTrading config with two tickers walks forward without error.

        Validates the full chain: multi-ticker fetch → wide-format frame
        → walk-forward dispatch to ``engine.run_pairs`` → equity curve
        consumes both legs.
        """

        cfg_dict = {
            "name": "pairs_smoke",
            "seed": 7,
            "data": {
                "source": {"name": "csv", "params": {"data_dir": str(pairs_csv_dir)}},
                "tickers": [_PAIRS_TICKER_A, _PAIRS_TICKER_B],
                "start": datetime(2020, 1, 2),
                "end": _PAIRS_END,
                "interval": "daily",
            },
            "strategy": {
                "name": "PairsTrading",
                "params": {
                    "entry_zscore": 2.0,
                    "exit_zscore": 0.5,
                    "stop_loss_zscore": 4.0,
                    "zscore_lookback": 30,
                    "p_value_threshold": 0.5,
                },
            },
            "validation": {
                "n_splits": 2,
                "test_size": _PAIRS_TEST_SIZE,
                "gap": 1,
                "holdout_pct": 0.0,
            },
            "slippage": {"scenario": "normal"},
        }
        cfg = ExperimentConfig.model_validate(cfg_dict)
        store = tmp_path / "exp"
        result = build_experiment(cfg).run(RunOptions(store_root=store, write_report=False))

        assert len(result.folds) == 2
        for fold in result.folds:
            assert len(fold.equity_curve) > 0
            assert fold.equity_curve[0] > 0.0


class TestFetchPairBars:
    def test_two_ticker_inner_join_renames_columns(
        self,
        pairs_csv_dir: Path,
    ) -> None:
        """
        `fetch_bars` for two tickers returns a wide-format frame.
        """

        from src.core.config import ExperimentConfig
        from src.core.registry import data_source_registry
        from src.orchestration.experiment import fetch_bars

        cfg = ExperimentConfig.model_validate(
            {
                "name": "join_check",
                "data": {
                    "source": {"name": "csv", "params": {"data_dir": str(pairs_csv_dir)}},
                    "tickers": [_PAIRS_TICKER_A, _PAIRS_TICKER_B],
                    "start": datetime(2020, 1, 2),
                    "end": _PAIRS_END,
                    "interval": "daily",
                },
                "strategy": {"name": "PairsTrading", "params": {}},
            }
        )
        source = data_source_registry.create_from_config(cfg.data.source)
        bars = fetch_bars(source, cfg, build_experiment(cfg).strategy)
        for col in (
            "open_a",
            "high_a",
            "low_a",
            "close_a",
            "volume_a",
            "open_b",
            "high_b",
            "low_b",
            "close_b",
            "volume_b",
        ):
            assert col in bars.columns, col
        assert len(bars) > 0


class TestWalkForwardPairsSplit:
    def test_split_pairs_frame_extracts_leg_subframes(self) -> None:
        """
        The wide → two-OHLCV split helper round-trips.
        """

        from src.engine.walk_forward import split_pairs_frame

        n = 6
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        wide = pd.DataFrame(
            {
                "open_a": range(n),
                "high_a": range(n),
                "low_a": range(n),
                "close_a": range(n),
                "volume_a": range(n),
                "open_b": [-i for i in range(n)],
                "high_b": [-i for i in range(n)],
                "low_b": [-i for i in range(n)],
                "close_b": [-i for i in range(n)],
                "volume_b": [-i for i in range(n)],
            },
            index=idx,
        )
        bars_a, bars_b = split_pairs_frame(wide)
        assert list(bars_a.columns) == ["open", "high", "low", "close", "volume"]
        assert list(bars_b.columns) == ["open", "high", "low", "close", "volume"]
        assert (bars_a["close"] == range(n)).all()
        assert (bars_b["close"] == [-i for i in range(n)]).all()

    def test_split_pairs_frame_missing_leg_raises(self) -> None:
        from src.engine.walk_forward import split_pairs_frame

        bars = pd.DataFrame(
            {"open_a": [1.0], "high_a": [1.0], "low_a": [1.0], "close_a": [1.0], "volume_a": [1.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        with pytest.raises(ValueError, match="wide-format columns"):
            split_pairs_frame(bars)
