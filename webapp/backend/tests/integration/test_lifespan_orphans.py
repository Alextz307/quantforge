"""
Lifespan-startup orphan reconciliation when a stale RUNNING row exists.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.db import open_db
from webapp.backend.app.infrastructure.job_store import (
    NewJob,
    insert_job,
    list_jobs,
    mark_running,
)
from webapp.backend.app.main import create_app
from webapp.backend.app.schemas.jobs import JobKind, JobStatus
from webapp.backend.app.services.user_service import create_user

DEAD_PID = 99_999_999


@pytest.fixture
def staged_orphan(_isolated_job_paths: None, db_conn: sqlite3.Connection) -> Iterator[str]:
    """
    Pre-populate the DB with a RUNNING job whose PID is dead.
    """

    user = create_user(db_conn, username="alex", password="password123", role=Role.USER)
    job = insert_job(
        db_conn,
        NewJob(
            user_id=user.id,
            kind=JobKind.RUN,
            command=("placeholder",),
            config_path=Path("/tmp/cfg.yaml"),
            log_path=Path("/tmp/job.log"),
        ),
    )
    mark_running(db_conn, job.id, DEAD_PID)
    yield job.id


def test_lifespan_marks_orphan_failed_on_startup(staged_orphan: str) -> None:
    get_settings.cache_clear()
    with TestClient(create_app()):
        with open_db() as conn:
            jobs = list_jobs(conn)
    [orphan] = [j for j in jobs if j.id == staged_orphan]
    assert orphan.status is JobStatus.FAILED
