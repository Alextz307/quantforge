"""
Unit tests for services/comparison_service.py.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.core.persistence import COMPARISONS_SUBDIR
from webapp.backend.app.infrastructure.store import ComparisonNotFoundError
from webapp.backend.app.services.comparison_service import (
    PlotNotFoundError,
    get_comparison,
    list_comparisons,
    resolve_plot,
)
from webapp.backend.tests.conftest import (
    PLOT_BYTES,
    PLOT_FILENAME,
    make_synthetic_comparison,
    make_viewer_user,
)

NEWER_NAME = "compare_newer"
OLDER_NAME = "compare_older"
NEWER_TS = datetime(2026, 4, 1, tzinfo=UTC)
OLDER_TS = datetime(2026, 1, 1, tzinfo=UTC)
EXPECTED_SHARPE = 0.5
EXPECTED_STRATEGY_COUNT = 2


def test_list_comparisons_sorts_newest_first(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    parent = root / "flat_store" / COMPARISONS_SUBDIR
    make_synthetic_comparison(parent, name=OLDER_NAME, created_at=OLDER_TS)
    make_synthetic_comparison(parent, name=NEWER_NAME, created_at=NEWER_TS)

    summaries = list_comparisons(
        root, conn=db_conn, user=make_viewer_user(db_conn), all_users=False
    )

    assert [s.name for s in summaries] == [NEWER_NAME, OLDER_NAME]


def test_list_comparisons_surfaces_strategies_and_store(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_comparison(
        root / "studies" / "main" / COMPARISONS_SUBDIR,
        name=NEWER_NAME,
        strategies={"A": "id_a", "B": "id_b"},
    )

    summary = list_comparisons(root, conn=db_conn, user=make_viewer_user(db_conn), all_users=False)[
        0
    ]

    assert summary.strategies == ["A", "B"]
    assert len(summary.strategies) == EXPECTED_STRATEGY_COUNT
    assert summary.store == "studies/main/comparisons"


def test_get_comparison_returns_full_detail(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_comparison(
        root / "flat_store" / COMPARISONS_SUBDIR,
        name=NEWER_NAME,
        strategies={"A": "id_a"},
    )

    detail = get_comparison(root, NEWER_NAME, conn=db_conn, user=make_viewer_user(db_conn))

    assert detail.name == NEWER_NAME
    assert detail.git_sha == "abc1234"
    assert len(detail.per_strategy_stats) == 1
    row = detail.per_strategy_stats[0]
    assert row.strategy == "A"
    assert row.experiment_id == "id_a"
    assert row.sharpe_mean == pytest.approx(EXPECTED_SHARPE)
    assert PLOT_FILENAME in detail.plots


def test_get_comparison_raises_for_unknown_name(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_comparison(root / "flat_store" / COMPARISONS_SUBDIR, name=NEWER_NAME)

    with pytest.raises(ComparisonNotFoundError):
        get_comparison(root, "missing_compare", conn=db_conn, user=make_viewer_user(db_conn))


def test_resolve_plot_returns_path_for_existing_file(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_comparison(root / "flat_store" / COMPARISONS_SUBDIR, name=NEWER_NAME)

    path = resolve_plot(
        root, NEWER_NAME, PLOT_FILENAME, conn=db_conn, user=make_viewer_user(db_conn)
    )

    assert path.is_file()
    assert path.read_bytes() == PLOT_BYTES


def test_resolve_plot_rejects_traversal(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_comparison(root / "flat_store" / COMPARISONS_SUBDIR, name=NEWER_NAME)

    with pytest.raises(PlotNotFoundError):
        resolve_plot(
            root,
            NEWER_NAME,
            "../../../etc/passwd",
            conn=db_conn,
            user=make_viewer_user(db_conn),
        )
