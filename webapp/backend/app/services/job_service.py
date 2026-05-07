"""Job-submission glue: persist row, write config YAML, spawn subprocess, enforce ownership."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

import yaml

from src.core.fs import ensure_parent_dir
from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.job_store import (
    IllegalStatusTransitionError,
    JobNotFoundError,
    NewJob,
    get_job,
    insert_job,
    list_jobs,
    list_running_jobs,
    mark_running,
    mark_terminal,
)
from webapp.backend.app.infrastructure.process_manager import (
    ProcessManager,
    build_run_command,
)
from webapp.backend.app.schemas.jobs import (
    JobKind,
    JobRow,
    JobStatus,
    JobSubmission,
)
from webapp.backend.app.schemas.users import UserPublic

logger = logging.getLogger(__name__)


class JobNotOwnedError(PermissionError):
    pass


class JobNotRunningError(ValueError):
    pass


def _config_path(job_temp_dir: Path, job_id: str) -> Path:
    return job_temp_dir / f"{job_id}.yaml"


def _log_path(job_temp_dir: Path, job_id: str) -> Path:
    return job_temp_dir / f"{job_id}.log"


def _write_config_yaml(path: Path, payload: dict[str, object]) -> None:
    ensure_parent_dir(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


async def submit_job(
    *,
    conn: sqlite3.Connection,
    manager: ProcessManager,
    user: UserPublic,
    submission: JobSubmission,
    store_root: Path,
    job_temp_dir: Path,
) -> JobRow:
    """Persist a queued row, write the config YAML, spawn the CLI, mark running."""
    if submission.kind is not JobKind.RUN:
        raise ValueError(f"unsupported job kind: {submission.kind.value}")
    job_temp_dir.mkdir(parents=True, exist_ok=True)
    # Two-phase: insert with placeholders, then UPDATE with paths derived from
    # the row id once we have it. Lifespan reconcile recovers a crash here.
    placeholder_command = ("placeholder",)
    placeholder_config = job_temp_dir / "_pending.yaml"
    placeholder_log = job_temp_dir / "_pending.log"
    row = insert_job(
        conn,
        NewJob(
            user_id=user.id,
            kind=submission.kind,
            command=placeholder_command,
            config_path=placeholder_config,
            log_path=placeholder_log,
        ),
    )
    config_path = _config_path(job_temp_dir, row.id)
    log_path = _log_path(job_temp_dir, row.id)
    command = build_run_command(config_path=config_path, job_id=row.id, store_root=store_root)
    conn.execute(
        "UPDATE jobs SET command = ?, config_path = ?, log_path = ? WHERE id = ?",
        (" ".join(command), str(config_path), str(log_path), row.id),
    )
    conn.commit()
    _write_config_yaml(config_path, submission.config_payload)
    pid = await manager.spawn(
        job_id=row.id,
        command=command,
        log_path=log_path,
        store_root=store_root,
    )
    return mark_running(conn, row.id, pid)


def list_jobs_for(
    conn: sqlite3.Connection, *, user: UserPublic, all_users: bool = False
) -> list[JobRow]:
    """Per-user view by default; admins may pass ``all_users=True``."""
    if all_users:
        if user.role is not Role.ADMIN:
            raise JobNotOwnedError("only admins can list all users' jobs")
        return list_jobs(conn)
    return list_jobs(conn, user_id=user.id)


def get_job_for(conn: sqlite3.Connection, *, user: UserPublic, job_id: str) -> JobRow:
    job = get_job(conn, job_id)
    _enforce_ownership(job, user)
    return job


async def cancel_job(
    *,
    conn: sqlite3.Connection,
    manager: ProcessManager,
    user: UserPublic,
    job_id: str,
) -> JobRow:
    job = get_job(conn, job_id)
    _enforce_ownership(job, user)
    if job.status is not JobStatus.RUNNING:
        raise JobNotRunningError(f"job {job_id} is in status {job.status.value}, cannot cancel")
    await manager.cancel(job_id)
    return job


def reconcile_orphans(conn: sqlite3.Connection) -> int:
    """Mark RUNNING rows whose PID is no longer alive as FAILED; returns the count."""
    orphans = 0
    for job in list_running_jobs(conn):
        if job.pid is None or not _pid_alive(job.pid):
            try:
                mark_terminal(
                    conn,
                    job.id,
                    status=JobStatus.FAILED,
                    exit_code=None,
                )
            except IllegalStatusTransitionError:
                continue
            orphans += 1
    if orphans:
        logger.warning("reconciled %d orphaned RUNNING job(s)", orphans)
    return orphans


def _enforce_ownership(job: JobRow, user: UserPublic) -> None:
    if user.role is Role.ADMIN:
        return
    if job.user_id != user.id:
        raise JobNotOwnedError(f"job {job.id} not owned by user {user.id}")


def _pid_alive(pid: int) -> bool:
    # EPERM (PermissionError) means the PID was recycled to another user — also dead.
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


__all__ = [
    "JobNotFoundError",
    "JobNotOwnedError",
    "JobNotRunningError",
    "cancel_job",
    "get_job_for",
    "list_jobs_for",
    "reconcile_orphans",
    "submit_job",
]
