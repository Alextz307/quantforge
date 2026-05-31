"""
Unit tests for services/study_service.py.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from webapp.backend.app.infrastructure.store import StudyNotFoundError
from webapp.backend.app.services.study_service import get_study, list_studies
from webapp.backend.tests.conftest import make_synthetic_study, make_viewer_user

NEWER_NAME = "study_newer"
OLDER_NAME = "study_older"
NEWER_TS = datetime(2026, 5, 1, tzinfo=UTC)
OLDER_TS = datetime(2026, 2, 1, tzinfo=UTC)
EXPECTED_TOTAL_LEGS = 3
EXPECTED_COMPLETED_LEGS = 2
EXPECTED_COMPLETION_PCT = pytest.approx(EXPECTED_COMPLETED_LEGS / EXPECTED_TOTAL_LEGS * 100.0)


def test_list_studies_sorts_newest_first(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    parent = root / "studies"
    make_synthetic_study(parent, name=OLDER_NAME, started_at=OLDER_TS)
    make_synthetic_study(parent, name=NEWER_NAME, started_at=NEWER_TS)

    summaries = list_studies(root, conn=db_conn, user=make_viewer_user(db_conn), all_users=False)

    assert [s.name for s in summaries] == [NEWER_NAME, OLDER_NAME]


def test_list_studies_surfaces_completion(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_study(
        root / "studies",
        name=NEWER_NAME,
        legs=(
            ("AdaptiveBollinger", "spy_daily_5y", True),
            ("AdaptiveBollinger", "spy_daily_10y", True),
            ("AdaptiveBollinger", "qqq_daily_5y", False),
        ),
    )

    summary = list_studies(root, conn=db_conn, user=make_viewer_user(db_conn), all_users=False)[0]

    assert summary.total_legs == EXPECTED_TOTAL_LEGS
    assert summary.completed_legs == EXPECTED_COMPLETED_LEGS
    assert summary.completion_pct == EXPECTED_COMPLETION_PCT


def test_get_study_returns_full_detail(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_study(
        root / "studies",
        name=NEWER_NAME,
        cross_strategy_compares_done=("spy_daily_5y",),
    )

    detail = get_study(root, NEWER_NAME, conn=db_conn, user=make_viewer_user(db_conn))

    assert detail.name == NEWER_NAME
    assert detail.spec_name == "demo_spec"
    assert detail.cross_strategy_compares_done == ["spy_daily_5y"]
    assert {leg.universe for leg in detail.legs} == {"spy_daily_5y", "spy_daily_10y"}
    completed_leg = next(leg for leg in detail.legs if leg.is_complete)
    assert completed_leg.run_experiment_id is not None
    assert "tune" in completed_leg.steps_completed


def test_get_study_raises_for_unknown_name(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_study(root / "studies", name=NEWER_NAME)

    with pytest.raises(StudyNotFoundError):
        get_study(root, "missing_study", conn=db_conn, user=make_viewer_user(db_conn))
