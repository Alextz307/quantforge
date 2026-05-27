"""SQLite CRUD for the jobs table; FSM ``queued → running → {completed, failed, cancelled}``."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from webapp.backend.app.schemas.jobs import TERMINAL_STATUSES, JobKind, JobRow, JobStatus


class JobNotFoundError(LookupError):
    pass


class IllegalStatusTransitionError(ValueError):
    pass


@dataclass(frozen=True)
class NewJob:
    user_id: int
    kind: JobKind
    command: tuple[str, ...]
    config_path: Path
    log_path: Path


_JOB_SELECT = (
    "SELECT j.id, j.user_id, j.kind, j.status, j.started_at, j.finished_at, "
    "j.exit_code, j.experiment_id, j.log_path, j.pid, u.username AS launched_by_username "
    "FROM jobs j LEFT JOIN users u ON u.id = j.user_id"
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_job(row: sqlite3.Row) -> JobRow:
    started_at = row["started_at"]
    finished_at = row["finished_at"]
    username = row["launched_by_username"]
    return JobRow(
        id=str(row["id"]),
        user_id=int(row["user_id"]),
        kind=JobKind(str(row["kind"])),
        status=JobStatus(str(row["status"])),
        started_at=datetime.fromisoformat(str(started_at)) if started_at else None,
        finished_at=datetime.fromisoformat(str(finished_at)) if finished_at else None,
        exit_code=int(row["exit_code"]) if row["exit_code"] is not None else None,
        experiment_id=str(row["experiment_id"]) if row["experiment_id"] else None,
        log_path=str(row["log_path"]),
        pid=int(row["pid"]) if row["pid"] is not None else None,
        launched_by_username=str(username) if username is not None else None,
    )


def insert_job(conn: sqlite3.Connection, new_job: NewJob) -> JobRow:
    job_id = uuid.uuid4().hex
    command_str = " ".join(new_job.command)
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, command, config_path, log_path, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            job_id,
            new_job.user_id,
            new_job.kind.value,
            command_str,
            str(new_job.config_path),
            str(new_job.log_path),
            JobStatus.QUEUED.value,
        ),
    )
    conn.commit()
    return get_job(conn, job_id)


def get_job(conn: sqlite3.Connection, job_id: str) -> JobRow:
    row = conn.execute(f"{_JOB_SELECT} WHERE j.id = ?", (job_id,)).fetchone()
    if row is None:
        raise JobNotFoundError(f"job {job_id} not found")
    return _row_to_job(row)


def list_jobs(conn: sqlite3.Connection, *, user_id: int | None = None) -> list[JobRow]:
    """List jobs newest-first by ``started_at``. ``user_id=None`` returns every job (admin view).

    Ordering: ``started_at DESC NULLS FIRST, id DESC``. Queued jobs (no
    ``started_at``) surface at the top — they're the freshest activity even
    though the subprocess hasn't spawned — and ``id`` breaks ties for stable
    pagination across the polling refetch.
    """
    order_clause = "ORDER BY j.started_at DESC NULLS FIRST, j.id DESC"
    if user_id is None:
        rows = conn.execute(f"{_JOB_SELECT} {order_clause}").fetchall()
    else:
        rows = conn.execute(
            f"{_JOB_SELECT} WHERE j.user_id = ? {order_clause}",
            (user_id,),
        ).fetchall()
    return [_row_to_job(row) for row in rows]


def list_running_jobs(conn: sqlite3.Connection) -> list[JobRow]:
    """Used by lifespan-startup orphan reconcile."""
    rows = conn.execute(
        f"{_JOB_SELECT} WHERE j.status = ?",
        (JobStatus.RUNNING.value,),
    ).fetchall()
    return [_row_to_job(row) for row in rows]


def mark_running(conn: sqlite3.Connection, job_id: str, pid: int) -> JobRow:
    job = get_job(conn, job_id)
    if job.status is not JobStatus.QUEUED:
        raise IllegalStatusTransitionError(
            f"cannot mark job {job_id} running from status {job.status.value}"
        )
    conn.execute(
        "UPDATE jobs SET status = ?, pid = ?, started_at = ? WHERE id = ?",
        (JobStatus.RUNNING.value, pid, _now_iso(), job_id),
    )
    conn.commit()
    return get_job(conn, job_id)


def mark_terminal(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    status: JobStatus,
    exit_code: int | None,
    experiment_id: str | None = None,
) -> JobRow:
    if status not in TERMINAL_STATUSES:
        raise IllegalStatusTransitionError(
            f"mark_terminal called with non-terminal status {status.value}"
        )
    job = get_job(conn, job_id)
    if job.status in TERMINAL_STATUSES:
        raise IllegalStatusTransitionError(
            f"job {job_id} already terminal at status {job.status.value}"
        )
    conn.execute(
        "UPDATE jobs SET status = ?, exit_code = ?, finished_at = ?, experiment_id = ? "
        "WHERE id = ?",
        (status.value, exit_code, _now_iso(), experiment_id, job_id),
    )
    conn.commit()
    return get_job(conn, job_id)


def delete_job(conn: sqlite3.Connection, job_id: str) -> bool:
    cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.commit()
    return cursor.rowcount > 0
