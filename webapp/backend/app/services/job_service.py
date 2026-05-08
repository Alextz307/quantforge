"""Job-submission glue: persist row, write config YAML, spawn subprocess, enforce ownership."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

import yaml

from src.core.fs import ensure_parent_dir
from src.core.persistence import HPO_SUBDIR
from src.optimization.checkpointing import TRIALS_JSONL_NAME
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
    TrialTailSpec,
    build_run_command,
    build_tune_command,
)
from webapp.backend.app.schemas.configs import ConfigKind, ValidationErrorItem
from webapp.backend.app.schemas.jobs import (
    JobKind,
    JobRow,
    JobStatus,
    JobSubmission,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.config_service import validate as validate_config
from webapp.backend.app.services.strategy_service import describe_strategy

logger = logging.getLogger(__name__)

_JOB_KIND_TO_CONFIG_KIND: dict[JobKind, ConfigKind] = {
    JobKind.RUN: ConfigKind.EXPERIMENT,
    JobKind.TUNE: ConfigKind.EXPERIMENT,
}


class JobNotOwnedError(PermissionError):
    pass


class JobNotRunningError(ValueError):
    pass


class JobConfigInvalidError(ValueError):
    """Raised when ``submit_job`` rejects a config_payload before persisting state.

    Carries the structured Pydantic errors so the router can surface them
    inline on the form rather than failing the spawned subprocess.
    """

    def __init__(self, errors: list[ValidationErrorItem]) -> None:
        super().__init__(f"invalid config payload ({len(errors)} error(s))")
        self.errors = errors


def _config_path(job_temp_dir: Path, job_id: str) -> Path:
    return job_temp_dir / f"{job_id}.yaml"


def _experiment_config_path(job_temp_dir: Path, job_id: str) -> Path:
    return job_temp_dir / f"{job_id}.exp.yaml"


def _hpo_config_path(job_temp_dir: Path, job_id: str) -> Path:
    return job_temp_dir / f"{job_id}.hpo.yaml"


def _log_path(job_temp_dir: Path, job_id: str) -> Path:
    return job_temp_dir / f"{job_id}.log"


def _trial_jsonl_path(store_root: Path, study_name: str) -> Path:
    return store_root / HPO_SUBDIR / study_name / TRIALS_JSONL_NAME


def _write_config_yaml(path: Path, payload: dict[str, object]) -> None:
    ensure_parent_dir(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


_STANDARD_FEATURES_BLOCK: dict[str, object] = {
    "name": "standard",
    "params": {"keep_ohlc": True},
}


def _maybe_inject_standard_features(payload: dict[str, object]) -> None:
    """Auto-add a canonical ``features:`` block for strategies that need it.

    VolatilityTargeting / ReturnForecast consume pre-engineered feature
    columns (signaled by a required ``feature_columns`` ctor param). The
    webapp form has no features-block UI, so a submission without
    ``features:`` would crash at runtime with a ``KeyError`` on the first
    missing engineered column. We inject the project's canonical default
    (``standard`` pipeline with ``keep_ohlc=true``, mirroring
    ``config/strategies/{volatility_targeting,return_forecast}.yaml``).

    User-supplied ``features`` blocks pass through unchanged.
    """
    if payload.get("features") is not None:
        return
    strategy = payload.get("strategy")
    if not isinstance(strategy, dict):
        return
    name = strategy.get("name")
    if not isinstance(name, str):
        return
    try:
        schema = describe_strategy(name)
    except KeyError:
        return
    needs_features = any(p.required and p.name == "feature_columns" for p in schema.params)
    if needs_features:
        payload["features"] = dict(_STANDARD_FEATURES_BLOCK)


async def submit_job(
    *,
    conn: sqlite3.Connection,
    manager: ProcessManager,
    user: UserPublic,
    submission: JobSubmission,
    store_root: Path,
    job_temp_dir: Path,
) -> JobRow:
    """Persist a queued row, write the config YAML(s), spawn the CLI, mark running.

    For RUN jobs: one ExperimentConfig YAML at ``<job_id>.yaml``.
    For TUNE jobs: an experiment YAML at ``<job_id>.exp.yaml`` plus an
    HPOConfig YAML at ``<job_id>.hpo.yaml`` — both validated through the
    shared ``config_service.validate`` machinery.
    """
    config_kind = _JOB_KIND_TO_CONFIG_KIND[submission.kind]
    validation = validate_config(config_kind, submission.config_payload)
    if not validation.valid:
        raise JobConfigInvalidError(validation.errors)
    if submission.kind is JobKind.TUNE:
        assert submission.hpo_payload is not None  # validator-enforced
        hpo_validation = validate_config(ConfigKind.HPO, submission.hpo_payload)
        if not hpo_validation.valid:
            raise JobConfigInvalidError(
                [
                    ValidationErrorItem(loc=["hpo_payload", *err.loc], msg=err.msg, type=err.type)
                    for err in hpo_validation.errors
                ]
            )
    _maybe_inject_standard_features(submission.config_payload)
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
    log_path = _log_path(job_temp_dir, row.id)
    if submission.kind is JobKind.RUN:
        config_path = _config_path(job_temp_dir, row.id)
        command = build_run_command(config_path=config_path, job_id=row.id, store_root=store_root)
        return await _persist_and_spawn(
            conn=conn,
            manager=manager,
            row=row,
            kind=JobKind.RUN,
            command=command,
            primary_config_path=config_path,
            log_path=log_path,
            store_root=store_root,
            configs_to_write={config_path: submission.config_payload},
        )

    assert submission.kind is JobKind.TUNE
    assert submission.hpo_payload is not None
    experiment_config_path = _experiment_config_path(job_temp_dir, row.id)
    hpo_config_path = _hpo_config_path(job_temp_dir, row.id)
    study_name = _extract_study_name(submission.hpo_payload)
    command = build_tune_command(
        experiment_config_path=experiment_config_path,
        hpo_config_path=hpo_config_path,
        store_root=store_root,
    )
    return await _persist_and_spawn(
        conn=conn,
        manager=manager,
        row=row,
        kind=JobKind.TUNE,
        command=command,
        primary_config_path=experiment_config_path,
        log_path=log_path,
        store_root=store_root,
        configs_to_write={
            experiment_config_path: submission.config_payload,
            hpo_config_path: submission.hpo_payload,
        },
        experiment_id=study_name,
        trial_tail=TrialTailSpec(
            study_name=study_name,
            trial_jsonl_path=_trial_jsonl_path(store_root, study_name),
        ),
    )


async def _persist_and_spawn(
    *,
    conn: sqlite3.Connection,
    manager: ProcessManager,
    row: JobRow,
    kind: JobKind,
    command: tuple[str, ...],
    primary_config_path: Path,
    log_path: Path,
    store_root: Path,
    configs_to_write: dict[Path, dict[str, object]],
    experiment_id: str | None = None,
    trial_tail: TrialTailSpec | None = None,
) -> JobRow:
    conn.execute(
        "UPDATE jobs SET command = ?, config_path = ?, log_path = ?, experiment_id = ? "
        "WHERE id = ?",
        (" ".join(command), str(primary_config_path), str(log_path), experiment_id, row.id),
    )
    conn.commit()
    for path, payload in configs_to_write.items():
        _write_config_yaml(path, payload)
    pid = await manager.spawn(
        job_id=row.id,
        kind=kind,
        command=command,
        log_path=log_path,
        store_root=store_root,
        trial_tail=trial_tail,
    )
    return mark_running(conn, row.id, pid)


def _extract_study_name(hpo_payload: dict[str, object]) -> str:
    """Extract validated ``study_name`` from an HPO payload.

    HPOConfig validation already ran (in submit_job), so ``study_name``
    is a non-empty path-safe string. Defensive re-check rather than
    threading the parsed model through.
    """
    raw = hpo_payload.get("study_name")
    if not isinstance(raw, str) or not raw:
        raise JobConfigInvalidError(
            [
                ValidationErrorItem(
                    loc=["hpo_payload", "study_name"],
                    msg="field required",
                    type="missing",
                )
            ]
        )
    return raw


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
    "JobConfigInvalidError",
    "JobNotFoundError",
    "JobNotOwnedError",
    "JobNotRunningError",
    "cancel_job",
    "get_job_for",
    "list_jobs_for",
    "reconcile_orphans",
    "submit_job",
]
