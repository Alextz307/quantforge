"""Service-layer behaviour: ownership, validation, reconcile, config write."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
import yaml

from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.job_store import (
    NewJob,
    insert_job,
    list_jobs,
    list_running_jobs,
    mark_running,
    mark_terminal,
)
from webapp.backend.app.infrastructure.process_manager import (
    JobEventBroker,
    ProcessManager,
)
from webapp.backend.app.schemas.jobs import JobKind, JobStatus, JobSubmission
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.job_service import (
    JobNotOwnedError,
    JobNotRunningError,
    cancel_job,
    get_job_for,
    list_jobs_for,
    reconcile_orphans,
    submit_job,
)
from webapp.backend.app.services.user_service import create_user

USER_PASSWORD = "password123"
SUBMISSION_PAYLOAD: dict[str, object] = {"name": "test-run", "seed": 42}
FAKE_PID = 12345


def _user(conn: sqlite3.Connection, username: str, role: Role = Role.USER) -> UserPublic:
    return create_user(conn, username=username, password=USER_PASSWORD, role=role)


def _new_job_for(user: UserPublic) -> NewJob:
    return NewJob(
        user_id=user.id,
        kind=JobKind.RUN,
        command=("placeholder",),
        config_path=Path("/tmp/cfg.yaml"),
        log_path=Path("/tmp/job.log"),
    )


def _stub_manager() -> ProcessManager:
    """Real ProcessManager with spawn + cancel replaced by AsyncMock."""
    manager = ProcessManager(JobEventBroker(), on_complete=AsyncMock())
    manager.spawn = AsyncMock(return_value=FAKE_PID)  # type: ignore[method-assign]
    manager.cancel = AsyncMock(return_value=True)  # type: ignore[method-assign]
    return manager


def test_submit_writes_yaml_and_persists_running(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    submission = JobSubmission(kind=JobKind.RUN, config_payload=SUBMISSION_PAYLOAD)
    store_root = tmp_path / "store"
    job_temp_dir = tmp_path / "jobs"

    row = asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=store_root,
            job_temp_dir=job_temp_dir,
        )
    )

    assert row.status is JobStatus.RUNNING
    assert row.pid == FAKE_PID
    config_yaml = job_temp_dir / f"{row.id}.yaml"
    parsed = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    assert parsed == SUBMISSION_PAYLOAD
    cast(AsyncMock, manager.spawn).assert_awaited_once()


def test_list_jobs_for_filters_by_user(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    bob = _user(db_conn, "bob")
    insert_job(db_conn, _new_job_for(alice))
    insert_job(db_conn, _new_job_for(bob))
    insert_job(db_conn, _new_job_for(alice))

    alice_view = list_jobs_for(db_conn, user=alice)
    assert {j.user_id for j in alice_view} == {alice.id}
    assert len(alice_view) == 2


def test_list_jobs_admin_all_users(db_conn: sqlite3.Connection) -> None:
    admin = _user(db_conn, "boss", role=Role.ADMIN)
    alice = _user(db_conn, "alice")
    insert_job(db_conn, _new_job_for(alice))
    insert_job(db_conn, _new_job_for(admin))

    full = list_jobs_for(db_conn, user=admin, all_users=True)
    assert {j.user_id for j in full} == {alice.id, admin.id}


def test_list_jobs_non_admin_all_users_rejected(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    with pytest.raises(JobNotOwnedError):
        list_jobs_for(db_conn, user=alice, all_users=True)


def test_get_job_blocks_other_user(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    bob = _user(db_conn, "bob")
    job = insert_job(db_conn, _new_job_for(alice))
    with pytest.raises(JobNotOwnedError):
        get_job_for(db_conn, user=bob, job_id=job.id)


def test_get_job_admin_can_view_any(db_conn: sqlite3.Connection) -> None:
    admin = _user(db_conn, "boss", role=Role.ADMIN)
    alice = _user(db_conn, "alice")
    job = insert_job(db_conn, _new_job_for(alice))
    fetched = get_job_for(db_conn, user=admin, job_id=job.id)
    assert fetched.id == job.id


def test_cancel_blocks_terminal_jobs(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    job = insert_job(db_conn, _new_job_for(alice))
    mark_running(db_conn, job.id, FAKE_PID)
    mark_terminal(db_conn, job.id, status=JobStatus.COMPLETED, exit_code=0)
    with pytest.raises(JobNotRunningError):
        asyncio.run(
            cancel_job(
                conn=db_conn,
                manager=_stub_manager(),
                user=alice,
                job_id=job.id,
            )
        )


def test_cancel_blocks_other_user(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    bob = _user(db_conn, "bob")
    job = insert_job(db_conn, _new_job_for(alice))
    mark_running(db_conn, job.id, FAKE_PID)
    with pytest.raises(JobNotOwnedError):
        asyncio.run(
            cancel_job(
                conn=db_conn,
                manager=_stub_manager(),
                user=bob,
                job_id=job.id,
            )
        )


def test_cancel_invokes_manager(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    job = insert_job(db_conn, _new_job_for(alice))
    mark_running(db_conn, job.id, FAKE_PID)
    manager = _stub_manager()
    asyncio.run(
        cancel_job(
            conn=db_conn,
            manager=manager,
            user=alice,
            job_id=job.id,
        )
    )
    cast(AsyncMock, manager.cancel).assert_awaited_once_with(job.id)


def test_reconcile_marks_dead_pid_failed(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    job = insert_job(db_conn, _new_job_for(alice))
    mark_running(db_conn, job.id, _dead_pid())

    reconciled = reconcile_orphans(db_conn)
    assert reconciled == 1

    [orphan] = list_jobs(db_conn, user_id=alice.id)
    assert orphan.status is JobStatus.FAILED


def test_reconcile_leaves_alive_pid_running(db_conn: sqlite3.Connection) -> None:
    alice = _user(db_conn, "alice")
    job = insert_job(db_conn, _new_job_for(alice))
    mark_running(db_conn, job.id, os.getpid())  # current process is alive

    assert reconcile_orphans(db_conn) == 0
    [running] = list_running_jobs(db_conn)
    assert running.id == job.id


def _dead_pid() -> int:
    """A PID well above any plausible PID_MAX (Linux ~32k, macOS ~99k)."""
    return 99_999_999
