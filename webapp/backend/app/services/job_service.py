"""
Job-submission glue: persist row, write config YAML, spawn subprocess, enforce ownership.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

from src.core.config import StudyLeg, StudySpec
from src.core.fs import ensure_parent_dir
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    read_experiment_manifest,
)
from src.optimization.checkpointing import BEST_CONFIG_YAML_NAME
from src.orchestration.study import make_leg_id
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
    build_compare_command,
    build_holdout_command,
    build_run_command,
    build_study_command,
    build_tune_command,
)
from webapp.backend.app.infrastructure.store import (
    HpoStudyNotFoundError,
    RunNotFoundError,
    find_hpo_study_dir_by_wire_id,
    find_run_dir,
    iter_hpo_study_dirs,
    iter_run_dirs,
)
from webapp.backend.app.schemas.configs import ConfigKind, ValidationErrorItem
from webapp.backend.app.schemas.jobs import (
    TERMINAL_STATUSES,
    ComparePayload,
    HoldoutPayload,
    JobKind,
    JobRow,
    JobStatus,
    JobSubmission,
    StudyPayload,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services._dir_cache import cached_artifact_dirs
from webapp.backend.app.services.config_service import validate as validate_config
from webapp.backend.app.services.hpo_service import best_config_reserves_holdout
from webapp.backend.app.services.strategy_service import describe_strategy
from webapp.backend.app.services.study_service import find_live_study_job_for
from webapp.backend.app.services.study_spec_uploads import find_upload_path

logger = logging.getLogger(__name__)


class JobNotOwnedError(PermissionError):
    pass


class JobNotRunningError(ValueError):
    pass


class JobConfigInvalidError(ValueError):
    """
    Raised when ``submit_job`` rejects a config_payload before persisting state.

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


def _write_config_yaml(path: Path, payload: dict[str, object]) -> None:
    ensure_parent_dir(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


_STANDARD_FEATURES_BLOCK: dict[str, object] = {
    "name": "standard",
    "params": {"keep_ohlc": True},
}


def _maybe_inject_standard_features(payload: dict[str, object]) -> None:
    """
    Auto-add a canonical ``features:`` block for strategies that need it.

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


@dataclass(frozen=True)
class _SpawnPlan:
    """
    Everything ``_persist_and_spawn`` needs to spawn one job kind.

    ``configs_to_write`` is empty for kinds that reference existing
    on-disk artifacts (compare/holdout) instead of building fresh YAMLs.
    ``experiment_id`` is pre-committed at submission time for kinds whose
    artifact directory name is known up front (TUNE/COMPARE/HOLDOUT);
    RUN resolves it post-completion via a manifest scan.
    """

    command: tuple[str, ...]
    primary_config_path: Path
    configs_to_write: dict[Path, dict[str, object]] = field(default_factory=dict)
    experiment_id: str | None = None
    artifact_id: str | None = None


@dataclass(frozen=True)
class _HandlerCtx:
    """
    Per-submission context threaded through validate() + plan().

    Bundles environment knobs that one or more kinds need but that vary
    per-request: ``user`` (for ownership-scoped resource lookups -
    currently only study-spec uploads), ``conn`` (collision detection
    against live jobs), ``config_root``, ``store_root``, and the
    user-uploads root. Kinds that don't need a given field just don't
    read it. Bundling them keeps every handler signature stable as new
    cross-cutting parameters get added.
    """

    user: UserPublic
    conn: sqlite3.Connection
    store_root: Path
    config_root: Path
    study_spec_uploads_dir: Path


class _JobHandler(Protocol):
    """
    Per-kind submit_job hook: validate pre-insert, then build a SpawnPlan.
    """

    def validate(self, submission: JobSubmission, ctx: _HandlerCtx) -> None:
        """
        Raise :class:`JobConfigInvalidError` on any pre-insert failure.

        Runs before the placeholder row is inserted so a rejected
        submission leaves no orphan state behind.
        """

    def plan(
        self, submission: JobSubmission, row: JobRow, job_temp_dir: Path, ctx: _HandlerCtx
    ) -> _SpawnPlan: ...


def _extract_study_name(hpo_payload: dict[str, object]) -> str:
    """
    Extract validated ``study_name`` from an HPO payload.

    HPOConfig validation already ran (in the tune handler), so
    ``study_name`` is a non-empty path-safe string. Defensive re-check
    rather than threading the parsed model through.
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


def _resolve_compare_inputs(
    payload: ComparePayload, store_root: Path
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """
    Resolve each ``run_ids[i]`` to ``(config.yaml, run_dir)`` in matching order.

    The CLI pairs ``--config`` and ``--reuse-runs`` positionally, so the
    two returned tuples MUST share order with ``payload.run_ids``. Errors
    accumulate across all bad indices so the UI can highlight every
    problematic row in a single 422 response.
    """

    errors: list[ValidationErrorItem] = []
    config_paths: list[Path] = []
    reuse_run_dirs: list[Path] = []
    for idx, run_id in enumerate(payload.run_ids):
        loc = ["compare_payload", "run_ids", str(idx)]
        try:
            run_dir = find_run_dir(store_root, run_id)
        except RunNotFoundError:
            errors.append(
                ValidationErrorItem(loc=loc, msg=f"run not found: {run_id}", type="value_error")
            )
            continue
        config_path = run_dir / EXPERIMENT_CONFIG_YAML
        if not config_path.is_file():
            errors.append(
                ValidationErrorItem(
                    loc=loc,
                    msg=f"run is missing config.yaml: {run_id}",
                    type="value_error",
                )
            )
            continue
        config_paths.append(config_path)
        reuse_run_dirs.append(run_dir)
    if errors:
        raise JobConfigInvalidError(errors)
    return tuple(config_paths), tuple(reuse_run_dirs)


def _resolve_holdout_source(payload: HoldoutPayload, store_root: Path) -> tuple[Path, str]:
    """
    Resolve the source dir and derive the artifact (out_name) name.

    Returns ``(source_path, artifact_name)`` where ``artifact_name`` is
    ``payload.out_name`` if set, else the source basename (matches the
    CLI's default).
    """

    loc = ["holdout_payload", "source_id"]
    if payload.source_kind == "run":
        try:
            source_path = find_run_dir(store_root, payload.source_id)
        except RunNotFoundError as exc:
            raise JobConfigInvalidError(
                [
                    ValidationErrorItem(
                        loc=loc,
                        msg=f"run not found: {payload.source_id}",
                        type="value_error",
                    )
                ]
            ) from exc
        try:
            manifest = read_experiment_manifest(source_path)
        except FileNotFoundError as exc:
            raise JobConfigInvalidError(
                [
                    ValidationErrorItem(
                        loc=loc,
                        msg=f"run is missing manifest.json: {payload.source_id}",
                        type="value_error",
                    )
                ]
            ) from exc
        if manifest.holdout_start is None:
            raise JobConfigInvalidError(
                [
                    ValidationErrorItem(
                        loc=loc,
                        msg=(
                            f"run {payload.source_id} has no holdout boundary "
                            "(manifest.holdout_start is null)"
                        ),
                        type="value_error",
                    )
                ]
            )
    else:
        try:
            source_path = find_hpo_study_dir_by_wire_id(store_root, payload.source_id)
        except HpoStudyNotFoundError as exc:
            raise JobConfigInvalidError(
                [
                    ValidationErrorItem(
                        loc=loc,
                        msg=f"hpo study not found: {payload.source_id}",
                        type="value_error",
                    )
                ]
            ) from exc
        if not (source_path / BEST_CONFIG_YAML_NAME).is_file():
            raise JobConfigInvalidError(
                [
                    ValidationErrorItem(
                        loc=loc,
                        msg=(
                            f"hpo study {payload.source_id} has no best_config.yaml "
                            "(study has no completed trials yet)"
                        ),
                        type="value_error",
                    )
                ]
            )
        if not best_config_reserves_holdout(source_path):
            raise JobConfigInvalidError(
                [
                    ValidationErrorItem(
                        loc=loc,
                        msg=(
                            f"hpo study {payload.source_id} reserved no holdout region "
                            "(best_config's validation.holdout_pct is 0 and no holdout_start "
                            "is pinned); honest OOS evaluation is undefined. Re-run the HPO "
                            "with validation.holdout_pct > 0."
                        ),
                        type="value_error",
                    )
                ]
            )
    artifact_name = payload.out_name if payload.out_name is not None else source_path.name
    return source_path, artifact_name


_STUDY_CONFIG_SUBDIR = "study"


def _library_spec_path(spec_name: str, config_root: Path) -> Path:
    return config_root / _STUDY_CONFIG_SUBDIR / f"{spec_name}.yaml"


def _resolve_spec_path(payload: StudyPayload, ctx: _HandlerCtx) -> Path:
    """
    Pick the on-disk spec file: user uploads first, library second.

    Uploads can never shadow library entries - the upload-save endpoint
    rejects slugs matching ``config/study/<slug>.yaml`` - so this two-step
    lookup is unambiguous. Caller is responsible for surfacing a 422 if
    neither location resolves.
    """

    upload_path = find_upload_path(ctx.study_spec_uploads_dir, ctx.user.id, payload.spec_name)
    if upload_path is not None:
        return upload_path
    return _library_spec_path(payload.spec_name, ctx.config_root)


def _resolve_study_spec(payload: StudyPayload, ctx: _HandlerCtx) -> tuple[StudySpec, Path]:
    """
    Load and schema-validate the spec referenced by ``payload.spec_name``.

    Returns the parsed ``StudySpec`` together with the resolved on-disk
    path (the same path the CLI subprocess receives via ``--spec``).
    Raises :class:`JobConfigInvalidError` if the file is missing, the
    YAML is malformed, or the parsed payload fails ``StudySpec``
    validation. Schema errors are flattened into ``ValidationErrorItem``s
    rooted at ``["study_payload", "spec_name", ...]`` so the form can
    highlight the offending field.
    """

    spec_path = _resolve_spec_path(payload, ctx)
    try:
        raw = spec_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise JobConfigInvalidError(
            [
                ValidationErrorItem(
                    loc=["study_payload", "spec_name"],
                    msg=f"study spec not found: {payload.spec_name}",
                    type="value_error",
                )
            ]
        ) from exc
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise JobConfigInvalidError(
            [
                ValidationErrorItem(
                    loc=["study_payload", "spec_name"],
                    msg=f"study spec is not valid YAML: {exc}",
                    type="value_error",
                )
            ]
        ) from exc
    if not isinstance(parsed, dict):
        raise JobConfigInvalidError(
            [
                ValidationErrorItem(
                    loc=["study_payload", "spec_name"],
                    msg="study spec must be a YAML mapping",
                    type="value_error",
                )
            ]
        )
    result = validate_config(ConfigKind.STUDY, parsed)
    if not result.valid:
        raise JobConfigInvalidError(
            [
                ValidationErrorItem(
                    loc=["study_payload", "spec_name", *err.loc],
                    msg=err.msg,
                    type=err.type,
                )
                for err in result.errors
            ]
        )
    return StudySpec.model_validate(parsed), spec_path


def _validate_only_legs(only_legs: list[str], spec_legs: list[StudyLeg]) -> None:
    """
    Each ``only_legs`` entry must match a leg id derived from the spec.
    """

    if not only_legs:
        return
    valid_ids: set[str] = {
        make_leg_id(leg.strategy, universe) for leg in spec_legs for universe in leg.universes
    }
    errors: list[ValidationErrorItem] = []
    for idx, leg_id in enumerate(only_legs):
        if leg_id not in valid_ids:
            errors.append(
                ValidationErrorItem(
                    loc=["study_payload", "only_legs", str(idx)],
                    msg=f"unknown leg id: {leg_id}",
                    type="value_error",
                )
            )
    if errors:
        raise JobConfigInvalidError(errors)


class _RunHandler:
    def validate(self, submission: JobSubmission, ctx: _HandlerCtx) -> None:
        assert submission.config_payload is not None  # JobSubmission validator-enforced
        result = validate_config(ConfigKind.EXPERIMENT, submission.config_payload)
        if not result.valid:
            raise JobConfigInvalidError(result.errors)

    def plan(
        self, submission: JobSubmission, row: JobRow, job_temp_dir: Path, ctx: _HandlerCtx
    ) -> _SpawnPlan:
        assert submission.config_payload is not None
        config_path = _config_path(job_temp_dir, row.id)
        return _SpawnPlan(
            command=build_run_command(
                config_path=config_path, job_id=row.id, store_root=ctx.store_root
            ),
            primary_config_path=config_path,
            configs_to_write={config_path: submission.config_payload},
        )


class _TuneHandler:
    def validate(self, submission: JobSubmission, ctx: _HandlerCtx) -> None:
        assert submission.config_payload is not None
        assert submission.hpo_payload is not None
        exp_result = validate_config(ConfigKind.EXPERIMENT, submission.config_payload)
        if not exp_result.valid:
            raise JobConfigInvalidError(exp_result.errors)
        hpo_result = validate_config(ConfigKind.HPO, submission.hpo_payload)
        if not hpo_result.valid:
            raise JobConfigInvalidError(
                [
                    ValidationErrorItem(loc=["hpo_payload", *err.loc], msg=err.msg, type=err.type)
                    for err in hpo_result.errors
                ]
            )
        _extract_study_name(submission.hpo_payload)

    def plan(
        self, submission: JobSubmission, row: JobRow, job_temp_dir: Path, ctx: _HandlerCtx
    ) -> _SpawnPlan:
        assert submission.config_payload is not None
        assert submission.hpo_payload is not None
        experiment_config_path = _experiment_config_path(job_temp_dir, row.id)
        hpo_config_path = _hpo_config_path(job_temp_dir, row.id)
        study_name = _extract_study_name(submission.hpo_payload)
        return _SpawnPlan(
            command=build_tune_command(
                experiment_config_path=experiment_config_path,
                hpo_config_path=hpo_config_path,
                store_root=ctx.store_root,
            ),
            primary_config_path=experiment_config_path,
            configs_to_write={
                experiment_config_path: submission.config_payload,
                hpo_config_path: submission.hpo_payload,
            },
            experiment_id=study_name,
        )


class _CompareHandler:
    def validate(self, submission: JobSubmission, ctx: _HandlerCtx) -> None:
        assert submission.compare_payload is not None
        # Pre-insert file checks; the same resolver runs again in plan().
        # Cheap (stat-only) and isolating the call here keeps validate() and
        # plan() symmetric across handlers.
        _resolve_compare_inputs(submission.compare_payload, ctx.store_root)

    def plan(
        self, submission: JobSubmission, row: JobRow, job_temp_dir: Path, ctx: _HandlerCtx
    ) -> _SpawnPlan:
        assert submission.compare_payload is not None
        payload = submission.compare_payload
        config_paths, reuse_run_dirs = _resolve_compare_inputs(payload, ctx.store_root)
        # No webapp-written temp YAMLs: the CLI consumes each run's frozen
        # config.yaml directly so --config and --reuse-runs stay in matching
        # positional order.
        return _SpawnPlan(
            command=build_compare_command(
                config_paths=config_paths,
                reuse_run_dirs=reuse_run_dirs,
                out_name=payload.out_name,
                significance_test=payload.significance_test,
                n_jobs=payload.n_jobs,
                write_report=payload.write_report,
                publish_label=payload.publish_label,
                store_root=ctx.store_root,
            ),
            primary_config_path=config_paths[0],
            experiment_id=payload.out_name,
            artifact_id=payload.out_name,
        )


class _HoldoutHandler:
    def validate(self, submission: JobSubmission, ctx: _HandlerCtx) -> None:
        assert submission.holdout_payload is not None
        _resolve_holdout_source(submission.holdout_payload, ctx.store_root)

    def plan(
        self, submission: JobSubmission, row: JobRow, job_temp_dir: Path, ctx: _HandlerCtx
    ) -> _SpawnPlan:
        assert submission.holdout_payload is not None
        payload = submission.holdout_payload
        source_path, artifact_name = _resolve_holdout_source(payload, ctx.store_root)
        return _SpawnPlan(
            command=build_holdout_command(
                source_kind=payload.source_kind,
                source_path=source_path,
                out_name=payload.out_name,
                write_report=payload.write_report,
                publish_label=payload.publish_label,
                store_root=ctx.store_root,
            ),
            primary_config_path=source_path,
            experiment_id=artifact_name,
            artifact_id=artifact_name,
        )


class _StudyHandler:
    def validate(self, submission: JobSubmission, ctx: _HandlerCtx) -> None:
        assert submission.study_payload is not None
        payload = submission.study_payload
        spec, _ = _resolve_study_spec(payload, ctx)
        _validate_only_legs(payload.only_legs, spec.legs)
        output_name = spec.output_dir.name
        live = find_live_study_job_for(ctx.conn, output_name)
        if live is not None:
            raise JobConfigInvalidError(
                [
                    ValidationErrorItem(
                        loc=["study_payload", "spec_name"],
                        msg=(
                            f"study '{output_name}' is already running under job "
                            f"{live!r}; wait for it to finish or cancel it before "
                            "launching another with the same output_dir."
                        ),
                        type="value_error",
                    )
                ]
            )

    def plan(
        self, submission: JobSubmission, row: JobRow, job_temp_dir: Path, ctx: _HandlerCtx
    ) -> _SpawnPlan:
        assert submission.study_payload is not None
        payload = submission.study_payload
        spec, spec_path = _resolve_study_spec(payload, ctx)
        output_name = spec.output_dir.name
        return _SpawnPlan(
            command=build_study_command(
                spec_path=spec_path,
                force_rerun=payload.force_rerun,
                only_legs=tuple(payload.only_legs),
                skip_compares=payload.skip_compares,
                skip_holdout_eval=payload.skip_holdout_eval,
                store_root=ctx.store_root,
            ),
            primary_config_path=spec_path,
            experiment_id=output_name,
            artifact_id=output_name,
        )


_HANDLERS: dict[JobKind, _JobHandler] = {
    JobKind.RUN: _RunHandler(),
    JobKind.TUNE: _TuneHandler(),
    JobKind.COMPARE: _CompareHandler(),
    JobKind.HOLDOUT: _HoldoutHandler(),
    JobKind.STUDY: _StudyHandler(),
}


async def submit_job(
    *,
    conn: sqlite3.Connection,
    manager: ProcessManager,
    user: UserPublic,
    submission: JobSubmission,
    store_root: Path,
    config_root: Path,
    job_temp_dir: Path,
    study_spec_uploads_dir: Path,
) -> JobRow:
    """
    Persist a queued row, write the config YAML(s), spawn the CLI, mark running.

    Per-kind logic lives in :data:`_HANDLERS`. Validation runs before the
    placeholder row is inserted so a rejected submission leaves no orphan
    state behind.
    """

    ctx = _HandlerCtx(
        user=user,
        conn=conn,
        store_root=store_root,
        config_root=config_root,
        study_spec_uploads_dir=study_spec_uploads_dir,
    )
    handler = _HANDLERS[submission.kind]
    handler.validate(submission, ctx)
    if submission.kind in (JobKind.RUN, JobKind.TUNE):
        assert submission.config_payload is not None
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
    plan = handler.plan(submission, row, job_temp_dir, ctx)
    return await _persist_and_spawn(
        conn=conn,
        manager=manager,
        row=row,
        kind=submission.kind,
        plan=plan,
        log_path=log_path,
        store_root=store_root,
    )


async def _persist_and_spawn(
    *,
    conn: sqlite3.Connection,
    manager: ProcessManager,
    row: JobRow,
    kind: JobKind,
    plan: _SpawnPlan,
    log_path: Path,
    store_root: Path,
) -> JobRow:
    conn.execute(
        "UPDATE jobs SET command = ?, config_path = ?, log_path = ?, experiment_id = ? "
        "WHERE id = ?",
        (
            " ".join(plan.command),
            str(plan.primary_config_path),
            str(log_path),
            plan.experiment_id,
            row.id,
        ),
    )
    conn.commit()

    for path, payload in plan.configs_to_write.items():
        _write_config_yaml(path, payload)

    pid = await manager.spawn(
        job_id=row.id,
        kind=kind,
        command=plan.command,
        log_path=log_path,
        store_root=store_root,
        artifact_id=plan.artifact_id,
    )
    return mark_running(conn, row.id, pid)


def list_jobs_for(
    conn: sqlite3.Connection,
    *,
    user: UserPublic,
    store_root: Path,
    all_users: bool = False,
) -> list[JobRow]:
    """
    Per-user view by default; admins may pass ``all_users=True``.

    Terminal jobs whose ``experiment_id`` no longer resolves to an artifact
    on disk are filtered out so the UI doesn't show entries that would
    error on click. Non-terminal jobs are kept regardless (their artifact
    may not exist yet).
    """

    if all_users:
        if user.role is not Role.ADMIN:
            raise JobNotOwnedError("only admins can list all users' jobs")
        rows = list_jobs(conn)
    else:
        rows = list_jobs(conn, user_id=user.id)

    # Gather all valid IDs in two tree walks rather than calling find_*_dir
    # per terminal job (each of which would re-walk the whole tree). Reuses
    # the shared TTL cache so consecutive job-polls skip the walks entirely.
    # Only built if any terminal job actually has an experiment_id to validate.
    needs_validation = any(
        row.status in TERMINAL_STATUSES and row.experiment_id is not None for row in rows
    )
    if not needs_validation:
        return list(rows)
    valid_run_ids = {p.name for p in cached_artifact_dirs(store_root, "run", iter_run_dirs)}
    valid_hpo_ids = {p.name for p in cached_artifact_dirs(store_root, "hpo", iter_hpo_study_dirs)}
    return [row for row in rows if _job_artifact_present(row, valid_run_ids, valid_hpo_ids)]


def _job_artifact_present(job: JobRow, valid_run_ids: set[str], valid_hpo_ids: set[str]) -> bool:
    if job.status not in TERMINAL_STATUSES:
        return True
    if job.experiment_id is None:
        return True
    if job.kind is JobKind.RUN:
        return job.experiment_id in valid_run_ids
    if job.kind is JobKind.TUNE:
        return job.experiment_id in valid_hpo_ids
    # COMPARE / HOLDOUT artifacts aren't covered by the cached id sets;
    # skip filtering for them - a missing dir surfaces a 404 on click,
    # which is the right UX for now.
    return True


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
    """
    Mark RUNNING rows whose PID is no longer alive as FAILED; returns the count.
    """

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
    # EPERM (PermissionError) means the PID was recycled to another user - also dead.
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
