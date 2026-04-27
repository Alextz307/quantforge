"""Integration test for the in-process regime-analysis driver.

Exercises the end-to-end ``run_regime_report`` path against a persisted
mini-experiment artifact tree, with a single fold covering the entire
synthetic bar range so the regime split is deterministic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.config import (
    ComponentConfig,
    DataConfig,
    ExperimentConfig,
    SlippageConfigSpec,
    ValidationConfig,
    write_frozen_yaml,
)
from src.core.exceptions import LeakageError
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_MANIFEST_JSON,
    FOLD_RESULTS_JSONL,
    REGIME_REPORTS_SUBDIR,
)
from src.core.regime_config import RegimeConfig
from src.core.types import Interval
from src.engine.scenarios import SlippageScenario
from src.orchestration.manifest import Manifest
from src.orchestration.regime_run import (
    load_run_from_disk,
    run_regime_report,
)
from src.orchestration.types import FoldRecord

N_BARS = 500
TICKER = "TEST"


def _make_csv_bars(path: Path) -> pd.DataFrame:
    """Write a synthetic OHLCV CSV that the csv data source can fetch."""
    rng = np.random.default_rng(31)
    dates = pd.date_range(start="2020-01-01", periods=N_BARS, freq="B")
    drift = np.where(np.arange(N_BARS) < N_BARS // 2, 0.003, -0.003)
    noise = rng.normal(0.0, 0.001, size=N_BARS)
    log_returns = drift + noise
    close = 100.0 * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.full(N_BARS, 1_000_000.0),
        },
        index=dates,
    )
    df.index.name = "Date"
    df.to_csv(path)
    return df


def _persist_minimal_run(
    run_dir: Path, csv_dir: Path, csv_filename: str
) -> tuple[ExperimentConfig, pd.DataFrame]:
    """Create a fake `experiment_results/runs/<id>/` tree on disk."""
    _make_csv_bars(csv_dir / csv_filename)

    cfg = ExperimentConfig(
        name="regime_smoke",
        seed=42,
        data=DataConfig(
            source=ComponentConfig(name="csv", params={"data_dir": str(csv_dir)}),
            tickers=[TICKER],
            start=datetime(2020, 1, 1),
            end=datetime(2030, 1, 1),
            interval=Interval.DAILY,
        ),
        validation=ValidationConfig(n_splits=1, test_size=50, gap=5),
        strategy=ComponentConfig(name="AdaptiveBollinger", params={"window": 20, "k": 2.0}),
        slippage=SlippageConfigSpec(scenario=SlippageScenario.NORMAL),
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    write_frozen_yaml(run_dir / EXPERIMENT_CONFIG_YAML, cfg)

    # Manifest data hash must match what the data source will return at
    # regime-analysis time, so go through the actual fetch path here
    # instead of fingerprinting the raw CSV (which has un-normalised
    # uppercase columns).
    from src.core.registry import data_source_registry
    from src.data.fingerprint import fingerprint_bars

    source = data_source_registry.create_from_config(cfg.data.source)
    bars_normalised = source.fetch(TICKER, cfg.data.start, cfg.data.end, cfg.data.interval)
    data_hash = fingerprint_bars(bars_normalised)

    manifest = Manifest(
        experiment_id="exp_smoke",
        name="regime_smoke",
        created_at=datetime(2026, 4, 25, tzinfo=UTC),
        git_sha="abcdef12",
        seed=42,
        data_hash=data_hash,
        slippage_scenario=SlippageScenario.NORMAL,
    )
    (run_dir / EXPERIMENT_MANIFEST_JSON).write_text(json.dumps(manifest.to_dict()))

    # One fold spanning the whole range, dummy metrics.
    fold = FoldRecord(
        fold_index=0,
        train_start=bars_normalised.index[0],
        train_end=bars_normalised.index[200],
        test_start=bars_normalised.index[200],
        test_end=bars_normalised.index[-1] + pd.Timedelta(days=1),
        total_return=0.05,
        annualized_return=0.10,
        annualized_volatility=0.15,
        sharpe_ratio=0.66,
        sortino_ratio=0.70,
        calmar_ratio=0.50,
        max_drawdown=-0.08,
        win_rate=0.55,
        trade_count=12,
        equity_curve=(1.0, 1.05),
    )
    (run_dir / FOLD_RESULTS_JSONL).write_text(json.dumps(fold.to_dict()) + "\n")
    return cfg, bars_normalised


def test_load_run_from_disk_round_trips(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "exp_smoke"
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    _persist_minimal_run(run_dir, csv_dir, f"{TICKER}.csv")

    loaded = load_run_from_disk(run_dir)
    assert loaded.experiment_id == "exp_smoke"
    assert loaded.config.strategy.name == "AdaptiveBollinger"
    assert loaded.manifest.git_sha == "abcdef12"
    assert len(loaded.folds) == 1
    assert loaded.folds[0].fold_index == 0


def test_load_run_from_disk_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run directory not found"):
        load_run_from_disk(tmp_path / "does_not_exist")


def test_load_run_from_disk_missing_artifact_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "exp_smoke"
    run_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="missing artifact"):
        load_run_from_disk(run_dir)


def test_run_regime_report_writes_under_regime_reports_subdir(tmp_path: Path) -> None:
    store_root = tmp_path
    run_dir = store_root / "runs" / "exp_smoke"
    csv_dir = store_root / "csv"
    csv_dir.mkdir()
    _persist_minimal_run(run_dir, csv_dir, f"{TICKER}.csv")

    regime_cfg = RegimeConfig.model_validate(
        {"detector": {"name": "trend", "params": {"window": 50}}}
    )
    report, out_dir = run_regime_report(
        run_dir=run_dir,
        regime_cfg=regime_cfg,
        out_name="trend_smoke",
        store_root=store_root,
    )

    assert out_dir == store_root / REGIME_REPORTS_SUBDIR / "trend_smoke"
    assert report.experiment_id == "exp_smoke"
    assert report.detector_name == "trend"
    # The single fold spans the whole bar range; with a 50-bar warmup it
    # ends up with mixed bull/bear bars over its test window — the
    # resulting label depends on the synthetic drift split, but at least
    # one assignment must exist (regime or mixed).
    assert len(report.per_regime_stats) >= 1 or len(report.mixed_fold_indices) >= 1


def test_run_regime_report_data_hash_drift_raises(tmp_path: Path) -> None:
    """Manifest hash drift is fatal — re-fetched bars don't match the saved run."""
    store_root = tmp_path
    run_dir = store_root / "runs" / "exp_smoke"
    csv_dir = store_root / "csv"
    csv_dir.mkdir()
    _persist_minimal_run(run_dir, csv_dir, f"{TICKER}.csv")

    # Mutate the CSV file AFTER persisting the manifest → drift. Keep the
    # OHLC ordering invariant intact (high >= max(open, close), low <= min)
    # by nudging volume only — the validator would otherwise short-circuit
    # the drift check with a DataQualityError.
    csv_path = csv_dir / f"{TICKER}.csv"
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df["Volume"] = df["Volume"] * 1.5
    df.to_csv(csv_path)

    regime_cfg = RegimeConfig.model_validate({"detector": "trend"})
    with pytest.raises(LeakageError, match="data_hash drift"):
        run_regime_report(
            run_dir=run_dir,
            regime_cfg=regime_cfg,
            out_name="drift_smoke",
            store_root=store_root,
        )
