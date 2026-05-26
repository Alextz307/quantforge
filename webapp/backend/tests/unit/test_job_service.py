"""Service-layer behaviour: ownership, validation, reconcile, config write."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
import yaml

from src.core import json_io
from src.core.persistence import (
    EXPERIMENT_MANIFEST_JSON,
    HPO_SUBDIR,
    RUNS_SUBDIR,
)
from src.optimization.checkpointing import BEST_CONFIG_YAML_NAME
from src.orchestration.comparison import SignificanceTest
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
from webapp.backend.app.schemas.jobs import (
    ComparePayload,
    HoldoutPayload,
    JobKind,
    JobStatus,
    JobSubmission,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.job_service import (
    JobConfigInvalidError,
    JobNotOwnedError,
    JobNotRunningError,
    cancel_job,
    get_job_for,
    list_jobs_for,
    reconcile_orphans,
    submit_job,
)
from webapp.backend.app.services.user_service import create_user

from ..conftest import make_synthetic_run, make_valid_experiment_payload

USER_PASSWORD = "password123"
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
    payload = make_valid_experiment_payload()
    submission = JobSubmission(kind=JobKind.RUN, config_payload=payload)
    store_root = tmp_path / "store"
    job_temp_dir = tmp_path / "jobs"

    row = asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=store_root,
            config_root=tmp_path / "config",
            job_temp_dir=job_temp_dir,
            study_spec_uploads_dir=tmp_path / "uploads",
        )
    )

    assert row.status is JobStatus.RUNNING
    assert row.pid == FAKE_PID
    config_yaml = job_temp_dir / f"{row.id}.yaml"
    parsed = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    assert parsed == payload
    cast(AsyncMock, manager.spawn).assert_awaited_once()


def test_submit_auto_injects_features_for_strategies_that_need_them(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """``VolatilityTargeting`` / ``ReturnForecast`` consume pre-engineered
    feature columns. The webapp form has no features-block UI, so the
    submitted payload is missing it; ``submit_job`` injects the canonical
    standard pipeline so the run doesn't crash with KeyError."""
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    payload = make_valid_experiment_payload()
    payload["strategy"] = {
        "name": "ReturnForecast",
        "params": {"feature_columns": ["rsi_14", "macd_signal"]},
    }
    submission = JobSubmission(kind=JobKind.RUN, config_payload=payload)
    job_temp_dir = tmp_path / "jobs"

    row = asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=tmp_path / "store",
            config_root=tmp_path / "config",
            job_temp_dir=job_temp_dir,
            study_spec_uploads_dir=tmp_path / "uploads",
        )
    )

    parsed = yaml.safe_load((job_temp_dir / f"{row.id}.yaml").read_text(encoding="utf-8"))
    assert parsed["features"] == {"name": "standard", "params": {"keep_ohlc": True}}


def test_submit_does_not_inject_features_for_self_contained_strategies(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """``AdaptiveBollinger`` / ``MomentumGatekeeper`` engineer features
    internally — no top-level ``features:`` block needed, none injected."""
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    payload = make_valid_experiment_payload()  # AdaptiveBollinger by default
    submission = JobSubmission(kind=JobKind.RUN, config_payload=payload)
    job_temp_dir = tmp_path / "jobs"

    row = asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=tmp_path / "store",
            config_root=tmp_path / "config",
            job_temp_dir=job_temp_dir,
            study_spec_uploads_dir=tmp_path / "uploads",
        )
    )

    parsed = yaml.safe_load((job_temp_dir / f"{row.id}.yaml").read_text(encoding="utf-8"))
    assert "features" not in parsed


def test_submit_respects_user_supplied_features_block(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    payload = make_valid_experiment_payload()
    payload["strategy"] = {
        "name": "ReturnForecast",
        "params": {"feature_columns": ["rsi_14"]},
    }
    user_features = {"name": "standard", "params": {"keep_ohlc": False}}
    payload["features"] = user_features
    submission = JobSubmission(kind=JobKind.RUN, config_payload=payload)
    job_temp_dir = tmp_path / "jobs"

    row = asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=tmp_path / "store",
            config_root=tmp_path / "config",
            job_temp_dir=job_temp_dir,
            study_spec_uploads_dir=tmp_path / "uploads",
        )
    )

    parsed = yaml.safe_load((job_temp_dir / f"{row.id}.yaml").read_text(encoding="utf-8"))
    assert parsed["features"] == user_features


def test_submit_rejects_invalid_payload_before_persisting(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    bad = make_valid_experiment_payload()
    del bad["data"]
    submission = JobSubmission(kind=JobKind.RUN, config_payload=bad)

    with pytest.raises(JobConfigInvalidError) as excinfo:
        asyncio.run(
            submit_job(
                conn=db_conn,
                manager=manager,
                user=user,
                submission=submission,
                store_root=tmp_path / "store",
                config_root=tmp_path / "config",
                job_temp_dir=tmp_path / "jobs",
                study_spec_uploads_dir=tmp_path / "uploads",
            )
        )

    assert any(err.loc == ["data"] for err in excinfo.value.errors)
    # Nothing persisted, nothing spawned.
    assert list_jobs(db_conn) == []
    cast(AsyncMock, manager.spawn).assert_not_awaited()


def test_list_jobs_for_filters_by_user(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    alice = _user(db_conn, "alice")
    bob = _user(db_conn, "bob")
    insert_job(db_conn, _new_job_for(alice))
    insert_job(db_conn, _new_job_for(bob))
    insert_job(db_conn, _new_job_for(alice))

    alice_view = list_jobs_for(db_conn, user=alice, store_root=tmp_path)
    assert {j.user_id for j in alice_view} == {alice.id}
    assert len(alice_view) == 2


def test_list_jobs_admin_all_users(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    admin = _user(db_conn, "boss", role=Role.ADMIN)
    alice = _user(db_conn, "alice")
    insert_job(db_conn, _new_job_for(alice))
    insert_job(db_conn, _new_job_for(admin))

    full = list_jobs_for(db_conn, user=admin, store_root=tmp_path, all_users=True)
    assert {j.user_id for j in full} == {alice.id, admin.id}


def test_list_jobs_non_admin_all_users_rejected(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    alice = _user(db_conn, "alice")
    with pytest.raises(JobNotOwnedError):
        list_jobs_for(db_conn, user=alice, store_root=tmp_path, all_users=True)


def test_list_jobs_hides_terminal_jobs_with_missing_artifacts(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Completed jobs whose experiment_id no longer resolves on disk are filtered."""
    alice = _user(db_conn, "alice")
    store_root = tmp_path / "store"
    store_root.mkdir()

    # Job 1: terminal + experiment_id missing on disk → filtered.
    orphan = insert_job(db_conn, _new_job_for(alice))
    mark_running(db_conn, orphan.id, FAKE_PID)
    mark_terminal(
        db_conn, orphan.id, status=JobStatus.COMPLETED, exit_code=0, experiment_id="ghost_id"
    )

    # Job 2: queued/running, no experiment_id yet → kept.
    live = insert_job(db_conn, _new_job_for(alice))

    # Job 3: terminal with no experiment_id (failed early) → kept.
    failed = insert_job(db_conn, _new_job_for(alice))
    mark_running(db_conn, failed.id, FAKE_PID + 1)
    mark_terminal(db_conn, failed.id, status=JobStatus.FAILED, exit_code=1)

    view = list_jobs_for(db_conn, user=alice, store_root=store_root)
    ids = {j.id for j in view}
    assert orphan.id not in ids
    assert live.id in ids
    assert failed.id in ids


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


def test_submit_tune_writes_both_yamls_and_persists_study_name(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """TUNE jobs split the payload across ``<id>.exp.yaml`` + ``<id>.hpo.yaml``
    and stamp ``experiment_id = study_name`` immediately so ``find_live_job_for``
    can resolve in-flight studies before the subprocess exits."""
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    exp_payload = make_valid_experiment_payload()
    hpo_payload: dict[str, object] = {"study_name": "demo_study", "n_trials": 2, "n_jobs": 1}
    submission = JobSubmission(
        kind=JobKind.TUNE, config_payload=exp_payload, hpo_payload=hpo_payload
    )
    job_temp_dir = tmp_path / "jobs"

    row = asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=tmp_path / "store",
            config_root=tmp_path / "config",
            job_temp_dir=job_temp_dir,
            study_spec_uploads_dir=tmp_path / "uploads",
        )
    )

    assert row.status is JobStatus.RUNNING
    assert row.kind is JobKind.TUNE
    assert row.experiment_id == "demo_study"
    exp_path = job_temp_dir / f"{row.id}.exp.yaml"
    hpo_path = job_temp_dir / f"{row.id}.hpo.yaml"
    assert yaml.safe_load(exp_path.read_text(encoding="utf-8")) == exp_payload
    parsed_hpo = yaml.safe_load(hpo_path.read_text(encoding="utf-8"))
    assert parsed_hpo["study_name"] == "demo_study"
    assert parsed_hpo["n_trials"] == 2


def test_submit_tune_rejects_invalid_hpo_payload(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """An HPOConfig with a path-separator in ``study_name`` is rejected
    before any DB row or YAML lands on disk."""
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    submission = JobSubmission(
        kind=JobKind.TUNE,
        config_payload=make_valid_experiment_payload(),
        hpo_payload={"study_name": "bad/name", "n_trials": 2},
    )
    job_temp_dir = tmp_path / "jobs"

    with pytest.raises(JobConfigInvalidError) as excinfo:
        asyncio.run(
            submit_job(
                conn=db_conn,
                manager=manager,
                user=user,
                submission=submission,
                store_root=tmp_path / "store",
                config_root=tmp_path / "config",
                job_temp_dir=job_temp_dir,
                study_spec_uploads_dir=tmp_path / "uploads",
            )
        )

    assert any("hpo_payload" in err.loc for err in excinfo.value.errors)
    assert list_jobs(db_conn, user_id=user.id) == []


def test_job_submission_validator_rejects_run_with_hpo_payload() -> None:
    with pytest.raises(ValueError, match="hpo_payload must be omitted"):
        JobSubmission(
            kind=JobKind.RUN,
            config_payload=make_valid_experiment_payload(),
            hpo_payload={"study_name": "x"},
        )


def test_job_submission_validator_requires_hpo_payload_for_tune() -> None:
    with pytest.raises(ValueError, match="hpo_payload is required"):
        JobSubmission(
            kind=JobKind.TUNE,
            config_payload=make_valid_experiment_payload(),
        )


# Compare + holdout fixtures + tests --------------------------------------------------------

# All run ids share a 64-char hash suffix shape so ``find_run_dir``'s glob
# resolves without ambiguity in a freshly-seeded store.
_COMPARE_RUN_A = "20260101_120000_AdaptiveBollinger_abc1234_deadbeef"
_COMPARE_RUN_B = "20260201_090000_AdaptiveBollinger_def5678_cafebabe"
_HOLDOUT_RUN_ID = "20260301_080000_AdaptiveBollinger_aaa0000_bbbb1111"
_HPO_STUDY = "demo_study"
_HPO_STUDY_WIRE_ID = f"hpo~{_HPO_STUDY}"


def _seed_run_with_holdout(store_root: Path, experiment_id: str, holdout_start: str) -> Path:
    """Synthetic run whose ``manifest.holdout_start`` is non-null."""
    run_dir = make_synthetic_run(
        store_root / RUNS_SUBDIR,
        experiment_id=experiment_id,
        created_at=datetime(2026, 3, 1, tzinfo=UTC),
    )
    # ``make_synthetic_run`` stamps ``holdout_start=None``; patch in place.
    manifest = json_io.read_dict(run_dir / EXPERIMENT_MANIFEST_JSON)
    manifest["holdout_start"] = holdout_start
    json_io.write(run_dir / EXPERIMENT_MANIFEST_JSON, manifest)
    return run_dir


def _seed_hpo_study(
    store_root: Path,
    name: str,
    *,
    write_best_config: bool = True,
    holdout_pct: float = 0.2,
) -> Path:
    study_dir = store_root / HPO_SUBDIR / name
    study_dir.mkdir(parents=True, exist_ok=True)
    json_io.write_jsonl(study_dir / "trials.jsonl", [])
    if write_best_config:
        best_cfg: dict[str, object] = {
            "strategy": {"name": "AdaptiveBollinger", "params": {}},
            "validation": {"holdout_pct": holdout_pct},
        }
        (study_dir / BEST_CONFIG_YAML_NAME).write_text(
            yaml.safe_dump(best_cfg),
            encoding="utf-8",
        )
    return study_dir


def test_submit_compare_spawns_with_reuse_runs_and_persists_out_name(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    store_root = tmp_path / "store"
    make_synthetic_run(store_root / RUNS_SUBDIR, experiment_id=_COMPARE_RUN_A)
    make_synthetic_run(store_root / RUNS_SUBDIR, experiment_id=_COMPARE_RUN_B)
    submission = JobSubmission(
        kind=JobKind.COMPARE,
        compare_payload=ComparePayload(
            run_ids=[_COMPARE_RUN_A, _COMPARE_RUN_B],
            out_name="my_compare",
            significance_test=SignificanceTest.BOOTSTRAP,
        ),
    )

    row = asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=store_root,
            config_root=tmp_path / "config",
            job_temp_dir=tmp_path / "jobs",
            study_spec_uploads_dir=tmp_path / "uploads",
        )
    )

    assert row.status is JobStatus.RUNNING
    assert row.kind is JobKind.COMPARE
    # Artifact name is pre-committed at submission so live-job lookups work
    # before the subprocess writes the comparison dir.
    assert row.experiment_id == "my_compare"
    spawn = cast(AsyncMock, manager.spawn)
    spawn.assert_awaited_once()
    assert spawn.await_args is not None
    spawn_kwargs = spawn.await_args.kwargs
    command = spawn_kwargs["command"]
    assert "compare" in command
    assert "--reuse-runs" in command
    reuse_value = command[command.index("--reuse-runs") + 1]
    # Reuse runs string carries both run dirs in matching order.
    assert _COMPARE_RUN_A in reuse_value
    assert _COMPARE_RUN_B in reuse_value
    assert reuse_value.index(_COMPARE_RUN_A) < reuse_value.index(_COMPARE_RUN_B)
    assert spawn_kwargs["artifact_id"] == "my_compare"


def test_submit_compare_rejects_unknown_run_id(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    store_root = tmp_path / "store"
    make_synthetic_run(store_root / RUNS_SUBDIR, experiment_id=_COMPARE_RUN_A)
    submission = JobSubmission(
        kind=JobKind.COMPARE,
        compare_payload=ComparePayload(
            run_ids=[_COMPARE_RUN_A, "ghost_run_id_not_on_disk"],
            out_name="bad_compare",
        ),
    )

    with pytest.raises(JobConfigInvalidError) as excinfo:
        asyncio.run(
            submit_job(
                conn=db_conn,
                manager=manager,
                user=user,
                submission=submission,
                store_root=store_root,
                config_root=tmp_path / "config",
                job_temp_dir=tmp_path / "jobs",
                study_spec_uploads_dir=tmp_path / "uploads",
            )
        )

    # Errors are positional: ghost id sits at index 1.
    locs = [err.loc for err in excinfo.value.errors]
    assert ["compare_payload", "run_ids", "1"] in locs
    assert list_jobs(db_conn, user_id=user.id) == []
    cast(AsyncMock, manager.spawn).assert_not_awaited()


def test_compare_payload_rejects_fewer_than_two_runs() -> None:
    with pytest.raises(ValueError):
        ComparePayload(run_ids=[_COMPARE_RUN_A], out_name="single_run_compare")


def test_compare_payload_rejects_invalid_slug() -> None:
    with pytest.raises(ValueError):
        ComparePayload(run_ids=[_COMPARE_RUN_A, _COMPARE_RUN_B], out_name="has spaces")


def test_submit_holdout_from_run_spawns_with_run_dir_flag(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    store_root = tmp_path / "store"
    run_dir = _seed_run_with_holdout(store_root, _HOLDOUT_RUN_ID, "2024-01-01T00:00:00")
    submission = JobSubmission(
        kind=JobKind.HOLDOUT,
        holdout_payload=HoldoutPayload(
            source_kind="run", source_id=_HOLDOUT_RUN_ID, out_name="my_holdout"
        ),
    )

    row = asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=store_root,
            config_root=tmp_path / "config",
            job_temp_dir=tmp_path / "jobs",
            study_spec_uploads_dir=tmp_path / "uploads",
        )
    )

    assert row.status is JobStatus.RUNNING
    assert row.kind is JobKind.HOLDOUT
    assert row.experiment_id == "my_holdout"
    spawn = cast(AsyncMock, manager.spawn)
    assert spawn.await_args is not None
    command = spawn.await_args.kwargs["command"]
    assert "holdout-eval" in command
    assert "--run-dir" in command
    assert str(run_dir) in command
    assert "--hpo-best" not in command


def test_submit_holdout_from_hpo_spawns_with_hpo_best_flag(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    store_root = tmp_path / "store"
    study_dir = _seed_hpo_study(store_root, _HPO_STUDY)
    submission = JobSubmission(
        kind=JobKind.HOLDOUT,
        holdout_payload=HoldoutPayload(source_kind="hpo", source_id=_HPO_STUDY_WIRE_ID),
    )

    row = asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=store_root,
            config_root=tmp_path / "config",
            job_temp_dir=tmp_path / "jobs",
            study_spec_uploads_dir=tmp_path / "uploads",
        )
    )

    # out_name defaults to the source basename (= study name).
    assert row.experiment_id == _HPO_STUDY
    spawn = cast(AsyncMock, manager.spawn)
    assert spawn.await_args is not None
    command = spawn.await_args.kwargs["command"]
    assert "--hpo-best" in command
    assert str(study_dir) in command


def test_submit_holdout_rejects_run_without_holdout_start(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """A run with ``manifest.holdout_start == null`` cannot be evaluated."""
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    store_root = tmp_path / "store"
    # ``make_synthetic_run`` defaults ``holdout_start=None`` — the rejected case.
    make_synthetic_run(store_root / RUNS_SUBDIR, experiment_id=_HOLDOUT_RUN_ID)
    submission = JobSubmission(
        kind=JobKind.HOLDOUT,
        holdout_payload=HoldoutPayload(source_kind="run", source_id=_HOLDOUT_RUN_ID),
    )

    with pytest.raises(JobConfigInvalidError) as excinfo:
        asyncio.run(
            submit_job(
                conn=db_conn,
                manager=manager,
                user=user,
                submission=submission,
                store_root=store_root,
                config_root=tmp_path / "config",
                job_temp_dir=tmp_path / "jobs",
                study_spec_uploads_dir=tmp_path / "uploads",
            )
        )

    locs = [err.loc for err in excinfo.value.errors]
    assert ["holdout_payload", "source_id"] in locs
    assert "no holdout boundary" in excinfo.value.errors[0].msg


def test_submit_holdout_rejects_hpo_without_best_config(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """HPO study with no completed trials → no ``best_config.yaml`` → rejected."""
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    store_root = tmp_path / "store"
    _seed_hpo_study(store_root, _HPO_STUDY, write_best_config=False)
    submission = JobSubmission(
        kind=JobKind.HOLDOUT,
        holdout_payload=HoldoutPayload(source_kind="hpo", source_id=_HPO_STUDY_WIRE_ID),
    )

    with pytest.raises(JobConfigInvalidError) as excinfo:
        asyncio.run(
            submit_job(
                conn=db_conn,
                manager=manager,
                user=user,
                submission=submission,
                store_root=store_root,
                config_root=tmp_path / "config",
                job_temp_dir=tmp_path / "jobs",
                study_spec_uploads_dir=tmp_path / "uploads",
            )
        )

    assert "best_config.yaml" in excinfo.value.errors[0].msg


def test_submit_holdout_rejects_hpo_whose_best_config_reserves_no_holdout(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Best_config exists but ``validation.holdout_pct=0`` → 422 with structured loc.

    Mirrors the CLI's manifest-level guard but fires at the API boundary so the
    user doesn't pay a "submit → wait → fail" round trip.
    """
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    store_root = tmp_path / "store"
    _seed_hpo_study(store_root, _HPO_STUDY, holdout_pct=0.0)
    submission = JobSubmission(
        kind=JobKind.HOLDOUT,
        holdout_payload=HoldoutPayload(source_kind="hpo", source_id=_HPO_STUDY_WIRE_ID),
    )

    with pytest.raises(JobConfigInvalidError) as excinfo:
        asyncio.run(
            submit_job(
                conn=db_conn,
                manager=manager,
                user=user,
                submission=submission,
                store_root=store_root,
                config_root=tmp_path / "config",
                job_temp_dir=tmp_path / "jobs",
                study_spec_uploads_dir=tmp_path / "uploads",
            )
        )

    assert "reserved no holdout region" in excinfo.value.errors[0].msg
    assert excinfo.value.errors[0].loc == ["holdout_payload", "source_id"]
    assert list_jobs(db_conn, user_id=user.id) == []


def test_job_submission_validator_rejects_compare_with_config_payload() -> None:
    with pytest.raises(ValueError, match="config_payload must be omitted"):
        JobSubmission(
            kind=JobKind.COMPARE,
            config_payload=make_valid_experiment_payload(),
            compare_payload=ComparePayload(run_ids=[_COMPARE_RUN_A, _COMPARE_RUN_B], out_name="x"),
        )


def test_job_submission_validator_requires_compare_payload_for_compare() -> None:
    with pytest.raises(ValueError, match="compare_payload is required"):
        JobSubmission(kind=JobKind.COMPARE)


def test_job_submission_validator_requires_holdout_payload_for_holdout() -> None:
    with pytest.raises(ValueError, match="holdout_payload is required"):
        JobSubmission(kind=JobKind.HOLDOUT)


def test_job_submission_validator_rejects_holdout_with_compare_payload() -> None:
    with pytest.raises(ValueError, match="compare_payload must be omitted"):
        JobSubmission(
            kind=JobKind.HOLDOUT,
            compare_payload=ComparePayload(run_ids=[_COMPARE_RUN_A, _COMPARE_RUN_B], out_name="x"),
            holdout_payload=HoldoutPayload(source_kind="run", source_id=_HOLDOUT_RUN_ID),
        )
