"""
Unit tests for services/hpo_service.py.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.core.persistence import HPO_SUBDIR
from src.optimization.tuner import storage_url_for
from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.db import bootstrap_schema, get_connection
from webapp.backend.app.infrastructure.job_store import NewJob, insert_job, mark_running
from webapp.backend.app.infrastructure.store import HpoStudyNotFoundError
from webapp.backend.app.schemas.jobs import JobKind
from webapp.backend.app.services.hpo_service import (
    find_live_job_for,
    get_hpo_study,
    get_param_importance,
    list_hpo_studies,
    list_trials,
)
from webapp.backend.app.services.user_service import create_user
from webapp.backend.tests.conftest import make_synthetic_hpo_study, make_viewer_user

NEWER_NAME = "Hpo__newer"
OLDER_NAME = "Hpo__older"
NEWER_TS = datetime(2026, 5, 1, tzinfo=UTC)
OLDER_TS = datetime(2026, 2, 1, tzinfo=UTC)
EXPECTED_BEST_VALUE = 0.95
EXPECTED_BEST_TRIAL_NUMBER = 2
EXPECTED_N_TRIALS = 4
EXPECTED_N_COMPLETE = 3
AFTER_TRIAL_FILTER = 1


def test_list_hpo_studies_sorts_newest_first(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    parent = root / "studies" / "main" / HPO_SUBDIR
    make_synthetic_hpo_study(parent, name=OLDER_NAME, created_at=OLDER_TS)
    make_synthetic_hpo_study(parent, name=NEWER_NAME, created_at=NEWER_TS)

    summaries = list_hpo_studies(
        root, conn=db_conn, user=make_viewer_user(db_conn), all_users=False
    )

    assert [s.name for s in summaries] == [NEWER_NAME, OLDER_NAME]


def test_list_hpo_studies_surfaces_best_and_store(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=NEWER_NAME,
        n_trials=EXPECTED_N_TRIALS,
        n_complete=EXPECTED_N_COMPLETE,
        best_value=EXPECTED_BEST_VALUE,
        best_trial_number=EXPECTED_BEST_TRIAL_NUMBER,
    )

    summary = list_hpo_studies(root, conn=db_conn, user=make_viewer_user(db_conn), all_users=False)[
        0
    ]

    assert summary.n_trials == EXPECTED_N_TRIALS
    assert summary.n_complete == EXPECTED_N_COMPLETE
    assert summary.best_value == pytest.approx(EXPECTED_BEST_VALUE)
    assert summary.best_trial_number == EXPECTED_BEST_TRIAL_NUMBER
    assert summary.store == "studies/main/hpo"


def test_get_hpo_study_returns_best_config(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=NEWER_NAME,
    )

    detail = get_hpo_study(
        root,
        f"studies~main~hpo~{NEWER_NAME}",
        conn=db_conn,
        user=make_viewer_user(db_conn),
    )

    assert detail.name == NEWER_NAME
    assert detail.best_config["name"] == "demo"
    strategy = detail.best_config["strategy"]
    assert isinstance(strategy, dict)
    assert strategy["name"] == "AdaptiveBollinger"


def test_get_hpo_study_handles_missing_best_config(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=NEWER_NAME,
        n_complete=0,
        write_best_config=False,
    )

    detail = get_hpo_study(
        root,
        f"studies~main~hpo~{NEWER_NAME}",
        conn=db_conn,
        user=make_viewer_user(db_conn),
    )

    assert detail.best_config == {}
    assert detail.best_value is None
    assert detail.best_trial_number is None


def test_get_hpo_study_raises_for_unknown_name(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(root / "studies" / "main" / HPO_SUBDIR, name=NEWER_NAME)

    with pytest.raises(HpoStudyNotFoundError):
        get_hpo_study(root, "missing_hpo", conn=db_conn, user=make_viewer_user(db_conn))


def test_list_trials_returns_all_by_default(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=NEWER_NAME,
        n_trials=EXPECTED_N_TRIALS,
        n_complete=EXPECTED_N_COMPLETE,
    )

    rows = list_trials(
        root,
        f"studies~main~hpo~{NEWER_NAME}",
        conn=db_conn,
        user=make_viewer_user(db_conn),
    )

    assert [r.number for r in rows] == list(range(EXPECTED_N_TRIALS))
    assert sum(1 for r in rows if r.state == "COMPLETE") == EXPECTED_N_COMPLETE


def test_list_trials_filters_by_after_trial(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=NEWER_NAME,
        n_trials=EXPECTED_N_TRIALS,
    )

    rows = list_trials(
        root,
        f"studies~main~hpo~{NEWER_NAME}",
        conn=db_conn,
        user=make_viewer_user(db_conn),
        after_trial=AFTER_TRIAL_FILTER,
    )

    assert [r.number for r in rows] == list(range(AFTER_TRIAL_FILTER + 1, EXPECTED_N_TRIALS))


def test_list_trials_raises_for_unknown_name(tmp_path: Path, db_conn: sqlite3.Connection) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(root / "studies" / "main" / HPO_SUBDIR, name=NEWER_NAME)

    with pytest.raises(HpoStudyNotFoundError):
        list_trials(root, "missing_hpo", conn=db_conn, user=make_viewer_user(db_conn))


_FAKE_PID = 12345
_TEST_PASSWORD = "password123"

_IMPORTANCE_STUDY_NAME = "Hpo__importance"
_IMPORTANCE_TRIAL_PARAMS = (
    (20, 1.5, 0.50),
    (30, 2.0, 0.70),
    (40, 2.5, 0.60),
    (25, 1.7, 0.55),
)


def test_get_param_importance_returns_message_when_too_few_completes(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=_IMPORTANCE_STUDY_NAME,
        n_trials=1,
        n_complete=1,
    )

    response = get_param_importance(
        root,
        f"studies~main~hpo~{_IMPORTANCE_STUDY_NAME}",
        conn=db_conn,
        user=make_viewer_user(db_conn),
    )

    assert response.importance == {}
    assert response.message is not None
    assert "completed trials" in response.message


def test_get_param_importance_returns_message_when_db_missing(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=_IMPORTANCE_STUDY_NAME,
        n_trials=EXPECTED_N_TRIALS,
        n_complete=EXPECTED_N_COMPLETE,
    )

    response = get_param_importance(
        root,
        f"studies~main~hpo~{_IMPORTANCE_STUDY_NAME}",
        conn=db_conn,
        user=make_viewer_user(db_conn),
    )

    assert response.importance == {}
    assert response.message is not None
    assert "DB" in response.message


def test_get_param_importance_with_real_optuna_study(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    """
    Seeds a real Optuna SQLite with multiple completed trials.
    """

    import optuna
    from optuna.distributions import FloatDistribution, IntDistribution
    from optuna.trial import create_trial

    root = tmp_path / "experiment_results"
    study_dir = make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=_IMPORTANCE_STUDY_NAME,
        n_trials=len(_IMPORTANCE_TRIAL_PARAMS),
        n_complete=len(_IMPORTANCE_TRIAL_PARAMS),
    )
    storage_url = storage_url_for(study_dir)
    distributions = {
        "window": IntDistribution(10, 50),
        "k": FloatDistribution(1.0, 3.0),
    }
    study = optuna.create_study(
        study_name=_IMPORTANCE_STUDY_NAME, storage=storage_url, direction="maximize"
    )
    for window, k, value in _IMPORTANCE_TRIAL_PARAMS:
        study.add_trial(
            create_trial(
                params={"window": window, "k": k},
                distributions=distributions,
                value=value,
            )
        )

    viewer = make_viewer_user(db_conn)
    wire_id = f"studies~main~hpo~{_IMPORTANCE_STUDY_NAME}"
    response = get_param_importance(root, wire_id, conn=db_conn, user=viewer)

    assert response.message is None
    assert set(response.importance.keys()) == {"window", "k"}
    assert all(v >= 0 for v in response.importance.values())
    assert sum(response.importance.values()) == pytest.approx(1.0, abs=1e-6)

    repeat = get_param_importance(root, wire_id, conn=db_conn, user=viewer)

    assert repeat.importance == response.importance


def test_get_param_importance_raises_for_unknown_name(
    tmp_path: Path, db_conn: sqlite3.Connection
) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(root / "studies" / "main" / HPO_SUBDIR, name=NEWER_NAME)

    with pytest.raises(HpoStudyNotFoundError):
        get_param_importance(root, "missing_hpo", conn=db_conn, user=make_viewer_user(db_conn))


def test_find_live_job_for_returns_none_when_no_jobs(tmp_path: Path) -> None:
    conn: sqlite3.Connection = get_connection(tmp_path / "webapp.sqlite")
    try:
        bootstrap_schema(conn)
        assert find_live_job_for(conn, "hpo~any_study") is None
    finally:
        conn.close()


def test_find_live_job_for_returns_running_tune_job_id(tmp_path: Path) -> None:
    conn: sqlite3.Connection = get_connection(tmp_path / "webapp.sqlite")
    try:
        bootstrap_schema(conn)
        user = create_user(conn, username="alice", password=_TEST_PASSWORD, role=Role.USER)
        job = insert_job(
            conn,
            NewJob(
                user_id=user.id,
                kind=JobKind.TUNE,
                command=("placeholder",),
                config_path=Path("/tmp/cfg.yaml"),
                log_path=Path("/tmp/job.log"),
            ),
        )
        conn.execute(
            "UPDATE jobs SET experiment_id = ? WHERE id = ?",
            ("demo_study", job.id),
        )
        conn.commit()
        mark_running(conn, job.id, _FAKE_PID)

        assert find_live_job_for(conn, "hpo~demo_study") == job.id
        assert find_live_job_for(conn, "hpo~other_study") is None
    finally:
        conn.close()
