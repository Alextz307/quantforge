"""
Unit tests for services/run_service.py.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.analysis.feature_importance import (
    FeatureImportance,
    FoldImportance,
    ImportanceMethod,
    build_importance_artifact,
)
from src.core import json_io
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    FEATURE_IMPORTANCE_DIVERGED_JSON,
    FEATURE_IMPORTANCE_JSON,
)
from webapp.backend.app.infrastructure.store import RunNotFoundError
from webapp.backend.app.services.run_service import (
    PlotNotFoundError,
    get_feature_importance,
    get_folds,
    get_run,
    list_runs,
    resolve_plot,
)
from webapp.backend.tests.conftest import (
    PLOT_BYTES,
    PLOT_FILENAME,
    make_synthetic_run,
    make_viewer_user,
)

_FEATURE_A = "rsi_14"
_FEATURE_B = "vol_20"
_ASSET = "QQQ"
_EXPECTED_ENTRY_COUNT = 4
_EXPECTED_PERMUTATION_COUNT = 2
_RSI_PERMUTATION_MEAN = 0.45
_RSI_PERMUTATION_STD = 0.0707106781

_BOTH_METHOD_FOLDS = (
    FoldImportance(
        fold_index=0,
        scores=(
            FeatureImportance(_FEATURE_A, 0.40, 0.0, ImportanceMethod.PERMUTATION),
            FeatureImportance(_FEATURE_B, 0.10, 0.0, ImportanceMethod.PERMUTATION),
            FeatureImportance(_FEATURE_A, 0.60, 0.0, ImportanceMethod.XGB_GAIN),
            FeatureImportance(_FEATURE_B, 0.30, 0.0, ImportanceMethod.XGB_GAIN),
        ),
    ),
    FoldImportance(
        fold_index=1,
        scores=(
            FeatureImportance(_FEATURE_A, 0.50, 0.0, ImportanceMethod.PERMUTATION),
            FeatureImportance(_FEATURE_B, 0.20, 0.0, ImportanceMethod.PERMUTATION),
            FeatureImportance(_FEATURE_A, 0.70, 0.0, ImportanceMethod.XGB_GAIN),
            FeatureImportance(_FEATURE_B, 0.40, 0.0, ImportanceMethod.XGB_GAIN),
        ),
    ),
)

_WITH_ASSET_FOLDS = (
    FoldImportance(
        fold_index=0,
        scores=(
            FeatureImportance(_FEATURE_A, 0.40, 0.0, ImportanceMethod.PERMUTATION),
            FeatureImportance(_FEATURE_A, 0.60, 0.0, ImportanceMethod.XGB_GAIN),
            FeatureImportance(_ASSET, 0.25, 0.0, ImportanceMethod.ASSET_PERMUTATION),
            FeatureImportance(_ASSET, 1.20, 0.0, ImportanceMethod.ASSET_XGB_GAIN),
        ),
    ),
)

# Two folds, both NaN -> aggregate mean AND across-fold std are NaN (null in JSON).
_NAN_SCORE = FeatureImportance(_FEATURE_A, float("nan"), 0.0, ImportanceMethod.PERMUTATION)
_NAN_FOLDS = (
    FoldImportance(0, (_NAN_SCORE,)),
    FoldImportance(1, (_NAN_SCORE,)),
)

# build_importance_artifact nulls non-finite scores, so hand-build a raw
# Infinity payload to exercise the read-side guard on a legacy artifact.
_INF_AGGREGATED_PAYLOAD: dict[str, object] = {
    "n_folds": 1,
    "per_fold": [],
    "aggregated": [
        {
            "feature": _FEATURE_A,
            "importance": float("inf"),
            "std": 0.0,
            "n_folds": 1,
            "method": ImportanceMethod.PERMUTATION.value,
        }
    ],
}


def _write_importance(run_dir: Path, folds: tuple[FoldImportance, ...]) -> None:
    json_io.write(run_dir / FEATURE_IMPORTANCE_JSON, build_importance_artifact(folds))


NEWER_ID = "20260301_120000_AdaptiveBollinger_aaa1111_aaaaaaaa"
OLDER_ID = "20260101_120000_AdaptiveBollinger_bbb2222_bbbbbbbb"
EXPECTED_FOLD_COUNT = 3
NEWER_TS = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
OLDER_TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_list_runs_sorts_newest_first(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    runs = root / "flat_store" / "runs"
    make_synthetic_run(runs, experiment_id=OLDER_ID, created_at=OLDER_TS)
    make_synthetic_run(runs, experiment_id=NEWER_ID, created_at=NEWER_TS)

    summaries = list_runs(root, conn=db_conn, user=make_viewer_user(db_conn), all_users=False)

    assert [s.experiment_id for s in summaries] == [NEWER_ID, OLDER_ID]


def test_list_runs_populates_strategy_and_universe_from_config(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(
        root / "studies" / "main" / "runs",
        experiment_id=NEWER_ID,
        strategy="PairsTrading",
        tickers=["IVV", "VOO"],
    )

    summary = list_runs(root, conn=db_conn, user=make_viewer_user(db_conn), all_users=False)[0]

    assert summary.strategy == "PairsTrading"
    assert summary.tickers == ["IVV", "VOO"]
    assert summary.store == "studies/main/runs"


def test_list_runs_skips_runs_missing_config(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "flat_store" / "runs", experiment_id=NEWER_ID)
    make_synthetic_run(root / "flat_store" / "runs", experiment_id=OLDER_ID, write_config=False)

    summaries = list_runs(root, conn=db_conn, user=make_viewer_user(db_conn), all_users=False)

    assert [s.experiment_id for s in summaries] == [NEWER_ID]


def test_list_runs_tolerates_missing_metrics(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "flat_store" / "runs", experiment_id=NEWER_ID, write_metrics=False)

    summary = list_runs(root, conn=db_conn, user=make_viewer_user(db_conn), all_users=False)[0]

    assert summary.sharpe_mean is None


def test_get_run_returns_full_detail(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(
        root / "flat_store" / "runs",
        experiment_id=NEWER_ID,
        strategy="AdaptiveBollinger",
        tickers=["SPY"],
    )

    detail = get_run(root, NEWER_ID, conn=db_conn, user=make_viewer_user(db_conn))

    assert detail.experiment_id == NEWER_ID
    assert detail.strategy == "AdaptiveBollinger"
    assert detail.tickers == ["SPY"]
    assert detail.git_sha == "abc1234"
    assert detail.metrics["sharpe_mean"] == pytest.approx(0.5)
    assert PLOT_FILENAME in detail.plots


def test_get_run_does_not_render_plots_on_mount(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    """
    Detail-page mount returns immediately; plot rendering is deferred to resolve_plot.
    """

    root = tmp_path / "experiment_results"
    nested_runs_dir = (
        root / "studies" / "main" / "hpo" / "AdaptiveBollinger" / "trials_artifacts" / "runs"
    )
    run_dir = make_synthetic_run(nested_runs_dir, experiment_id=NEWER_ID, write_plot=False)

    detail = get_run(root, NEWER_ID, conn=db_conn, user=make_viewer_user(db_conn))

    assert detail.plots == []
    assert not (run_dir / "plots" / "equity_curves.png").exists()


def test_resolve_plot_lazy_renders_when_missing(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    """
    Direct plot fetch triggers lazy rendering once.
    """

    root = tmp_path / "experiment_results"
    nested_runs_dir = (
        root / "studies" / "main" / "hpo" / "AdaptiveBollinger" / "trials_artifacts" / "runs"
    )
    run_dir = make_synthetic_run(nested_runs_dir, experiment_id=NEWER_ID, write_plot=False)
    viewer = make_viewer_user(db_conn)

    path = resolve_plot(root, NEWER_ID, "equity_curves.png", conn=db_conn, user=viewer)
    assert path.is_file()
    assert path == run_dir / "plots" / "equity_curves.png"
    mtime_before = path.stat().st_mtime_ns

    resolve_plot(root, NEWER_ID, "equity_curves.png", conn=db_conn, user=viewer)

    assert path.stat().st_mtime_ns == mtime_before, "second call should not re-render"


def test_get_run_raises_for_unknown_id(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "flat_store" / "runs", experiment_id=NEWER_ID)

    with pytest.raises(RunNotFoundError):
        get_run(root, "missing_id", conn=db_conn, user=make_viewer_user(db_conn))


def test_get_folds_returns_one_row_per_fold(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(
        root / "flat_store" / "runs", experiment_id=NEWER_ID, n_folds=EXPECTED_FOLD_COUNT
    )

    folds = get_folds(root, NEWER_ID, conn=db_conn, user=make_viewer_user(db_conn))

    assert len(folds) == EXPECTED_FOLD_COUNT
    assert [f.fold_index for f in folds] == list(range(EXPECTED_FOLD_COUNT))
    assert all(f.equity_curve for f in folds)


def test_resolve_plot_returns_path_for_existing_file(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "flat_store" / "runs", experiment_id=NEWER_ID)

    path = resolve_plot(root, NEWER_ID, PLOT_FILENAME, conn=db_conn, user=make_viewer_user(db_conn))

    assert path.is_file()
    assert path.read_bytes() == PLOT_BYTES


def test_resolve_plot_rejects_traversal(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "flat_store" / "runs", experiment_id=NEWER_ID)

    with pytest.raises(PlotNotFoundError):
        resolve_plot(
            root,
            NEWER_ID,
            "../../../../etc/passwd",
            conn=db_conn,
            user=make_viewer_user(db_conn),
        )


def test_resolve_plot_raises_for_missing_file(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "flat_store" / "runs", experiment_id=NEWER_ID)

    with pytest.raises(PlotNotFoundError):
        resolve_plot(
            root,
            NEWER_ID,
            "does_not_exist.png",
            conn=db_conn,
            user=make_viewer_user(db_conn),
        )


def test_get_feature_importance_returns_aggregated_entries(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    run_dir = make_synthetic_run(root / "flat_store" / "runs", experiment_id=NEWER_ID)
    _write_importance(run_dir, _BOTH_METHOD_FOLDS)

    response = get_feature_importance(root, NEWER_ID, conn=db_conn, user=make_viewer_user(db_conn))

    assert len(response.entries) == _EXPECTED_ENTRY_COUNT
    assert response.message is None
    permutation = [e for e in response.entries if e.method == ImportanceMethod.PERMUTATION]
    assert len(permutation) == _EXPECTED_PERMUTATION_COUNT
    rsi = next(e for e in permutation if e.feature == _FEATURE_A)
    assert rsi.importance == pytest.approx(_RSI_PERMUTATION_MEAN)
    assert rsi.std == pytest.approx(_RSI_PERMUTATION_STD)


def test_get_feature_importance_excludes_per_asset_entries(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    run_dir = make_synthetic_run(root / "flat_store" / "runs", experiment_id=NEWER_ID)
    _write_importance(run_dir, _WITH_ASSET_FOLDS)

    response = get_feature_importance(root, NEWER_ID, conn=db_conn, user=make_viewer_user(db_conn))

    assert {e.method for e in response.entries} == {
        ImportanceMethod.PERMUTATION,
        ImportanceMethod.XGB_GAIN,
    }
    assert all(e.feature == _FEATURE_A for e in response.entries)


def test_get_feature_importance_serves_entries_without_config(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    # A present artifact must render even with config.yaml gone: it implies a
    # feature-consuming strategy, so the response must not re-read the config.
    root = tmp_path / "experiment_results"
    run_dir = make_synthetic_run(
        root / "flat_store" / "runs", experiment_id=NEWER_ID, strategy="MomentumGatekeeper"
    )
    _write_importance(run_dir, _BOTH_METHOD_FOLDS)
    (run_dir / EXPERIMENT_CONFIG_YAML).unlink()

    response = get_feature_importance(root, NEWER_ID, conn=db_conn, user=make_viewer_user(db_conn))

    assert len(response.entries) == _EXPECTED_ENTRY_COUNT
    assert response.computable is True


def test_get_feature_importance_missing_artifact_returns_empty_with_message(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "flat_store" / "runs", experiment_id=NEWER_ID)

    response = get_feature_importance(root, NEWER_ID, conn=db_conn, user=make_viewer_user(db_conn))

    assert response.entries == []
    assert response.message is not None


def test_get_feature_importance_marks_feature_strategy_computable(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(
        root / "flat_store" / "runs", experiment_id=NEWER_ID, strategy="MomentumGatekeeper"
    )

    response = get_feature_importance(root, NEWER_ID, conn=db_conn, user=make_viewer_user(db_conn))

    assert response.entries == []
    assert response.computable is True


def test_get_feature_importance_marks_rule_based_not_computable(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(
        root / "flat_store" / "runs", experiment_id=NEWER_ID, strategy="AdaptiveBollinger"
    )

    response = get_feature_importance(root, NEWER_ID, conn=db_conn, user=make_viewer_user(db_conn))

    assert response.computable is False


def test_get_feature_importance_surfaces_diverged_pointer(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    run_dir = make_synthetic_run(
        root / "flat_store" / "runs", experiment_id=NEWER_ID, strategy="MomentumGatekeeper"
    )
    json_io.write(run_dir / FEATURE_IMPORTANCE_DIVERGED_JSON, {"diverged_run_id": "exp_refit_123"})

    response = get_feature_importance(root, NEWER_ID, conn=db_conn, user=make_viewer_user(db_conn))

    assert response.entries == []
    assert response.computable is True
    assert response.diverged_run_id == "exp_refit_123"


def test_get_feature_importance_maps_nan_to_none(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    run_dir = make_synthetic_run(root / "flat_store" / "runs", experiment_id=NEWER_ID)
    _write_importance(run_dir, _NAN_FOLDS)

    response = get_feature_importance(root, NEWER_ID, conn=db_conn, user=make_viewer_user(db_conn))

    assert len(response.entries) == 1
    assert response.entries[0].importance is None
    assert response.entries[0].std is None


def test_get_feature_importance_maps_inf_to_none(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    run_dir = make_synthetic_run(root / "flat_store" / "runs", experiment_id=NEWER_ID)
    json_io.write(run_dir / FEATURE_IMPORTANCE_JSON, _INF_AGGREGATED_PAYLOAD)

    response = get_feature_importance(root, NEWER_ID, conn=db_conn, user=make_viewer_user(db_conn))

    assert len(response.entries) == 1
    assert response.entries[0].importance is None


def test_get_feature_importance_raises_for_unknown_id(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "flat_store" / "runs", experiment_id=NEWER_ID)

    with pytest.raises(RunNotFoundError):
        get_feature_importance(root, "missing_id", conn=db_conn, user=make_viewer_user(db_conn))
