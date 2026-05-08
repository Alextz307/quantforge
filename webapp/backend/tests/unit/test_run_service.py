"""Unit tests for services/run_service.py."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from webapp.backend.app.infrastructure.store import RunNotFoundError
from webapp.backend.app.services.run_service import (
    PlotNotFoundError,
    get_folds,
    get_run,
    list_runs,
    resolve_plot,
)
from webapp.backend.tests.conftest import (
    PLOT_BYTES,
    PLOT_FILENAME,
    make_synthetic_run,
)

NEWER_ID = "20260301_120000_AdaptiveBollinger_aaa1111_aaaaaaaa"
OLDER_ID = "20260101_120000_AdaptiveBollinger_bbb2222_bbbbbbbb"
EXPECTED_FOLD_COUNT = 3
NEWER_TS = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
OLDER_TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_list_runs_sorts_newest_first(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    runs = root / "thesis_demo" / "runs"
    make_synthetic_run(runs, experiment_id=OLDER_ID, created_at=OLDER_TS)
    make_synthetic_run(runs, experiment_id=NEWER_ID, created_at=NEWER_TS)

    summaries = list_runs(root)

    assert [s.experiment_id for s in summaries] == [NEWER_ID, OLDER_ID]


def test_list_runs_populates_strategy_and_universe_from_config(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(
        root / "studies" / "main" / "runs",
        experiment_id=NEWER_ID,
        strategy="PairsTrading",
        tickers=["IVV", "VOO"],
    )

    summary = list_runs(root)[0]

    assert summary.strategy == "PairsTrading"
    assert summary.tickers == ["IVV", "VOO"]
    assert summary.store == "studies/main/runs"


def test_list_runs_skips_runs_missing_config(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "thesis_demo" / "runs", experiment_id=NEWER_ID)
    make_synthetic_run(root / "thesis_demo" / "runs", experiment_id=OLDER_ID, write_config=False)

    summaries = list_runs(root)

    assert [s.experiment_id for s in summaries] == [NEWER_ID]


def test_list_runs_tolerates_missing_metrics(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "thesis_demo" / "runs", experiment_id=NEWER_ID, write_metrics=False)

    summary = list_runs(root)[0]

    assert summary.sharpe_mean is None
    assert summary.calmar_mean is None


def test_get_run_returns_full_detail(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(
        root / "thesis_demo" / "runs",
        experiment_id=NEWER_ID,
        strategy="AdaptiveBollinger",
        tickers=["SPY"],
    )

    detail = get_run(root, NEWER_ID)

    assert detail.experiment_id == NEWER_ID
    assert detail.strategy == "AdaptiveBollinger"
    assert detail.tickers == ["SPY"]
    assert detail.git_sha == "abc1234"
    assert detail.metrics["sharpe_mean"] == pytest.approx(0.5)
    assert PLOT_FILENAME in detail.plots


def test_get_run_lazy_renders_plots_when_missing(tmp_path: Path) -> None:
    """HPO-trial / comparison-leg runs skip plot generation; webapp renders on first access."""
    root = tmp_path / "experiment_results"
    nested_runs_dir = (
        root / "studies" / "main" / "hpo" / "AdaptiveBollinger" / "trials_artifacts" / "runs"
    )
    run_dir = make_synthetic_run(nested_runs_dir, experiment_id=NEWER_ID, write_plot=False)

    detail = get_run(root, NEWER_ID)

    assert "equity_curves.png" in detail.plots
    assert "fold_stability.png" in detail.plots
    plot_path = run_dir / "plots" / "equity_curves.png"
    assert plot_path.is_file()
    mtime_before = plot_path.stat().st_mtime_ns

    get_run(root, NEWER_ID)

    assert plot_path.stat().st_mtime_ns == mtime_before, "second call should not re-render"


def test_get_run_raises_for_unknown_id(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "thesis_demo" / "runs", experiment_id=NEWER_ID)

    with pytest.raises(RunNotFoundError):
        get_run(root, "missing_id")


def test_get_folds_returns_one_row_per_fold(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(
        root / "thesis_demo" / "runs", experiment_id=NEWER_ID, n_folds=EXPECTED_FOLD_COUNT
    )

    folds = get_folds(root, NEWER_ID)

    assert len(folds) == EXPECTED_FOLD_COUNT
    assert [f.fold_index for f in folds] == list(range(EXPECTED_FOLD_COUNT))
    assert all(f.equity_curve for f in folds)


def test_resolve_plot_returns_path_for_existing_file(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "thesis_demo" / "runs", experiment_id=NEWER_ID)

    path = resolve_plot(root, NEWER_ID, PLOT_FILENAME)

    assert path.is_file()
    assert path.read_bytes() == PLOT_BYTES


def test_resolve_plot_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "thesis_demo" / "runs", experiment_id=NEWER_ID)

    with pytest.raises(PlotNotFoundError):
        resolve_plot(root, NEWER_ID, "../../../../etc/passwd")


def test_resolve_plot_raises_for_missing_file(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "thesis_demo" / "runs", experiment_id=NEWER_ID)

    with pytest.raises(PlotNotFoundError):
        resolve_plot(root, NEWER_ID, "does_not_exist.png")
