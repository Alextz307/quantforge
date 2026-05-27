"""SQL-layer behaviour for the jobs table."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.job_store import (
    IllegalStatusTransitionError,
    JobNotFoundError,
    NewJob,
    delete_job,
    get_job,
    insert_job,
    list_jobs,
    list_running_jobs,
    mark_running,
    mark_terminal,
)
from webapp.backend.app.schemas.jobs import JobKind, JobStatus
from webapp.backend.app.services.user_service import create_user

USER_PASSWORD = "password123"
RUN_PID = 4321
EXIT_OK = 0
EXIT_FAIL = 137


def _create_user(conn: sqlite3.Connection, username: str, role: Role = Role.USER) -> int:
    return create_user(conn, username=username, password=USER_PASSWORD, role=role).id


def _new_job(user_id: int, *, command: str = "x") -> NewJob:
    return NewJob(
        user_id=user_id,
        kind=JobKind.RUN,
        command=(command,),
        config_path=Path("/tmp/cfg.yaml"),
        log_path=Path("/tmp/job.log"),
    )


def test_insert_round_trip(db_conn: sqlite3.Connection) -> None:
    uid = _create_user(db_conn, "alice")
    inserted = insert_job(db_conn, _new_job(uid))
    assert inserted.status is JobStatus.QUEUED
    assert inserted.user_id == uid
    fetched = get_job(db_conn, inserted.id)
    assert fetched == inserted


def test_get_job_raises_when_missing(db_conn: sqlite3.Connection) -> None:
    with pytest.raises(JobNotFoundError):
        get_job(db_conn, "nope")


def test_list_jobs_filters_by_user(db_conn: sqlite3.Connection) -> None:
    a = _create_user(db_conn, "alice")
    b = _create_user(db_conn, "bob")
    insert_job(db_conn, _new_job(a))
    insert_job(db_conn, _new_job(b))
    insert_job(db_conn, _new_job(a))

    alice_jobs = list_jobs(db_conn, user_id=a)
    bob_jobs = list_jobs(db_conn, user_id=b)
    all_jobs = list_jobs(db_conn)

    assert len(alice_jobs) == 2
    assert len(bob_jobs) == 1
    assert len(all_jobs) == 3


def test_list_jobs_orders_queued_first_then_started_newest_first(
    db_conn: sqlite3.Connection,
) -> None:
    """Queued jobs surface above started ones; started jobs are newest-first.

    Pins the SQL ORDER BY contract so the UI doesn't need a sort widget.
    """

    uid = _create_user(db_conn, "alice")
    earliest = insert_job(db_conn, _new_job(uid, command="earliest"))
    middle = insert_job(db_conn, _new_job(uid, command="middle"))
    latest = insert_job(db_conn, _new_job(uid, command="latest"))
    queued = insert_job(db_conn, _new_job(uid, command="still_queued"))
    # Start in non-id order so the test would fail under the legacy ``ORDER BY id DESC``.
    mark_running(db_conn, earliest.id, RUN_PID)
    mark_running(db_conn, latest.id, RUN_PID + 1)
    mark_running(db_conn, middle.id, RUN_PID + 2)

    ordering = [j.id for j in list_jobs(db_conn, user_id=uid)]

    # Queued (no started_at) comes first; started jobs follow in started_at-desc.
    # mark_running stamps started_at to now(), so the most recently marked
    # (``middle``) is the freshest start.
    assert ordering == [queued.id, middle.id, latest.id, earliest.id]


def test_mark_running_then_terminal_transitions(db_conn: sqlite3.Connection) -> None:
    uid = _create_user(db_conn, "alice")
    job = insert_job(db_conn, _new_job(uid))

    running = mark_running(db_conn, job.id, RUN_PID)
    assert running.status is JobStatus.RUNNING
    assert running.pid == RUN_PID
    assert running.started_at is not None

    completed = mark_terminal(
        db_conn,
        job.id,
        status=JobStatus.COMPLETED,
        exit_code=EXIT_OK,
        experiment_id="xyz",
    )
    assert completed.status is JobStatus.COMPLETED
    assert completed.exit_code == EXIT_OK
    assert completed.experiment_id == "xyz"
    assert completed.finished_at is not None


def test_double_terminal_is_illegal(db_conn: sqlite3.Connection) -> None:
    uid = _create_user(db_conn, "alice")
    job = insert_job(db_conn, _new_job(uid))
    mark_running(db_conn, job.id, RUN_PID)
    mark_terminal(db_conn, job.id, status=JobStatus.COMPLETED, exit_code=EXIT_OK)
    with pytest.raises(IllegalStatusTransitionError):
        mark_terminal(db_conn, job.id, status=JobStatus.FAILED, exit_code=EXIT_FAIL)


def test_running_to_running_is_illegal(db_conn: sqlite3.Connection) -> None:
    uid = _create_user(db_conn, "alice")
    job = insert_job(db_conn, _new_job(uid))
    mark_running(db_conn, job.id, RUN_PID)
    with pytest.raises(IllegalStatusTransitionError):
        mark_running(db_conn, job.id, RUN_PID + 1)


def test_mark_terminal_rejects_non_terminal_status(db_conn: sqlite3.Connection) -> None:
    uid = _create_user(db_conn, "alice")
    job = insert_job(db_conn, _new_job(uid))
    mark_running(db_conn, job.id, RUN_PID)
    with pytest.raises(IllegalStatusTransitionError):
        mark_terminal(db_conn, job.id, status=JobStatus.RUNNING, exit_code=None)


def test_list_running_jobs_filters_status(db_conn: sqlite3.Connection) -> None:
    uid = _create_user(db_conn, "alice")
    queued_job = insert_job(db_conn, _new_job(uid))
    running_job = insert_job(db_conn, _new_job(uid))
    mark_running(db_conn, running_job.id, RUN_PID)
    completed_job = insert_job(db_conn, _new_job(uid))
    mark_running(db_conn, completed_job.id, RUN_PID + 1)
    mark_terminal(db_conn, completed_job.id, status=JobStatus.COMPLETED, exit_code=EXIT_OK)

    running = list_running_jobs(db_conn)
    assert {j.id for j in running} == {running_job.id}
    assert queued_job.id not in {j.id for j in running}


def test_delete_job_round_trip(db_conn: sqlite3.Connection) -> None:
    uid = _create_user(db_conn, "alice")
    job = insert_job(db_conn, _new_job(uid))
    assert delete_job(db_conn, job.id) is True
    assert delete_job(db_conn, job.id) is False
    with pytest.raises(JobNotFoundError):
        get_job(db_conn, job.id)
