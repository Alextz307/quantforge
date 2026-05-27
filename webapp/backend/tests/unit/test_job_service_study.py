"""Service-layer behaviour for STUDY jobs: spec resolution, only_legs, collision."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.job_store import NewJob, insert_job, mark_running
from webapp.backend.app.infrastructure.process_manager import (
    JobEventBroker,
    ProcessManager,
)
from webapp.backend.app.schemas.jobs import JobKind, JobStatus, JobSubmission, StudyPayload
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.job_service import (
    JobConfigInvalidError,
    submit_job,
)
from webapp.backend.app.services.study_spec_uploads import save_upload
from webapp.backend.app.services.user_service import create_user

USER_PASSWORD = "password123"
FAKE_PID = 54321
_SPEC_NAME = "tiny_study"
_OUTPUT_DIR = "studies/tiny"
_STRATEGY = "AdaptiveBollinger"
_UNIVERSE = "spy_daily_5y"


def _user(conn: sqlite3.Connection, username: str) -> UserPublic:
    return create_user(conn, username=username, password=USER_PASSWORD, role=Role.USER)


def _stub_manager() -> ProcessManager:
    manager = ProcessManager(JobEventBroker(), on_complete=AsyncMock())
    manager.spawn = AsyncMock(return_value=FAKE_PID)  # type: ignore[method-assign]
    manager.cancel = AsyncMock(return_value=True)  # type: ignore[method-assign]
    return manager


def _minimal_spec_dict(
    *,
    name: str = _SPEC_NAME,
    output_dir: str = _OUTPUT_DIR,
    strategy: str = _STRATEGY,
    universes: tuple[str, ...] = (_UNIVERSE,),
) -> dict[str, object]:
    return {
        "name": name,
        "output_dir": output_dir,
        "legs": [
            {
                "strategy": strategy,
                "strategy_config": "config/strategies/adaptive_bollinger.yaml",
                "hpo_config": "config/hpo/adaptive_bollinger.yaml",
                "universes": list(universes),
            }
        ],
    }


def _write_spec(config_root: Path, spec: dict[str, object], *, name: str = _SPEC_NAME) -> Path:
    study_dir = config_root / "study"
    study_dir.mkdir(parents=True, exist_ok=True)
    spec_path = study_dir / f"{name}.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return spec_path


def _study_submission(**kwargs: object) -> JobSubmission:
    payload = StudyPayload(spec_name=_SPEC_NAME, **kwargs)  # type: ignore[arg-type]
    return JobSubmission(kind=JobKind.STUDY, study_payload=payload)


def test_submit_study_persists_running_with_output_dir_as_experiment_id(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    config_root = tmp_path / "config"
    _write_spec(config_root, _minimal_spec_dict())
    submission = _study_submission()

    row = asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=tmp_path / "store",
            config_root=config_root,
            job_temp_dir=tmp_path / "jobs",
            study_spec_uploads_dir=tmp_path / "uploads",
        )
    )

    assert row.status is JobStatus.RUNNING
    assert row.pid == FAKE_PID
    assert row.experiment_id == "tiny"


def test_submit_study_command_includes_flags(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    config_root = tmp_path / "config"
    _write_spec(config_root, _minimal_spec_dict())
    submission = JobSubmission(
        kind=JobKind.STUDY,
        study_payload=StudyPayload(
            spec_name=_SPEC_NAME,
            force_rerun=True,
            skip_compares=True,
            skip_holdout_eval=True,
            only_legs=[f"{_STRATEGY}__{_UNIVERSE}"],
        ),
    )

    asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=tmp_path / "store",
            config_root=config_root,
            job_temp_dir=tmp_path / "jobs",
            study_spec_uploads_dir=tmp_path / "uploads",
        )
    )

    spawn_call = manager.spawn.await_args  # type: ignore[attr-defined]
    command = spawn_call.kwargs["command"]
    assert "study" in command
    assert "run" in command
    assert "--force-rerun" in command
    assert "--skip-compares" in command
    assert "--skip-holdout-eval" in command
    assert "--only-leg" in command
    assert f"{_STRATEGY}__{_UNIVERSE}" in command


def test_submit_study_rejects_missing_spec_file(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    submission = _study_submission()

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

    locs = [err.loc for err in excinfo.value.errors]
    assert ["study_payload", "spec_name"] in locs
    assert "not found" in excinfo.value.errors[0].msg


def test_submit_study_rejects_malformed_yaml(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    config_root = tmp_path / "config"
    (config_root / "study").mkdir(parents=True)
    (config_root / "study" / f"{_SPEC_NAME}.yaml").write_text(
        "name: tiny\nlegs: [unbalanced", encoding="utf-8"
    )
    submission = _study_submission()

    with pytest.raises(JobConfigInvalidError) as excinfo:
        asyncio.run(
            submit_job(
                conn=db_conn,
                manager=manager,
                user=user,
                submission=submission,
                store_root=tmp_path / "store",
                config_root=config_root,
                job_temp_dir=tmp_path / "jobs",
                study_spec_uploads_dir=tmp_path / "uploads",
            )
        )

    assert any("YAML" in err.msg for err in excinfo.value.errors)


def test_submit_study_rejects_schema_invalid(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    """A spec missing ``legs`` surfaces the Pydantic error under ``study_payload.spec_name``."""
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    config_root = tmp_path / "config"
    (config_root / "study").mkdir(parents=True)
    (config_root / "study" / f"{_SPEC_NAME}.yaml").write_text(
        yaml.safe_dump({"name": "tiny", "output_dir": "out"}), encoding="utf-8"
    )
    submission = _study_submission()

    with pytest.raises(JobConfigInvalidError) as excinfo:
        asyncio.run(
            submit_job(
                conn=db_conn,
                manager=manager,
                user=user,
                submission=submission,
                store_root=tmp_path / "store",
                config_root=config_root,
                job_temp_dir=tmp_path / "jobs",
                study_spec_uploads_dir=tmp_path / "uploads",
            )
        )

    locs = [err.loc for err in excinfo.value.errors]
    assert any(loc[:2] == ["study_payload", "spec_name"] and "legs" in loc for loc in locs)


def test_submit_study_rejects_unknown_only_leg(db_conn: sqlite3.Connection, tmp_path: Path) -> None:
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    config_root = tmp_path / "config"
    _write_spec(config_root, _minimal_spec_dict())
    submission = JobSubmission(
        kind=JobKind.STUDY,
        study_payload=StudyPayload(spec_name=_SPEC_NAME, only_legs=["bogus_leg_id"]),
    )

    with pytest.raises(JobConfigInvalidError) as excinfo:
        asyncio.run(
            submit_job(
                conn=db_conn,
                manager=manager,
                user=user,
                submission=submission,
                store_root=tmp_path / "store",
                config_root=config_root,
                job_temp_dir=tmp_path / "jobs",
                study_spec_uploads_dir=tmp_path / "uploads",
            )
        )

    locs = [err.loc for err in excinfo.value.errors]
    assert any(loc[:2] == ["study_payload", "only_legs"] for loc in locs)


def test_submit_study_rejects_when_running_collision_exists(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """A non-terminal STUDY job with the same ``experiment_id`` blocks resubmission."""
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    config_root = tmp_path / "config"
    _write_spec(config_root, _minimal_spec_dict())
    seeded = insert_job(
        db_conn,
        NewJob(
            user_id=user.id,
            kind=JobKind.STUDY,
            command=("placeholder",),
            config_path=Path("/tmp/cfg.yaml"),
            log_path=Path("/tmp/job.log"),
        ),
    )
    db_conn.execute("UPDATE jobs SET experiment_id = ? WHERE id = ?", ("tiny", seeded.id))
    db_conn.commit()
    mark_running(db_conn, seeded.id, FAKE_PID)
    submission = _study_submission()

    with pytest.raises(JobConfigInvalidError) as excinfo:
        asyncio.run(
            submit_job(
                conn=db_conn,
                manager=manager,
                user=user,
                submission=submission,
                store_root=tmp_path / "store",
                config_root=config_root,
                job_temp_dir=tmp_path / "jobs",
                study_spec_uploads_dir=tmp_path / "uploads",
            )
        )

    locs = [err.loc for err in excinfo.value.errors]
    assert ["study_payload", "spec_name"] in locs
    assert "already running" in excinfo.value.errors[0].msg


def test_submit_study_resolves_user_upload_when_present(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """An upload owned by the submitting user takes precedence over the library."""
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    config_root = tmp_path / "config"
    config_root.mkdir()
    (config_root / "strategies").mkdir()
    (config_root / "hpo").mkdir()
    (config_root / "universes").mkdir()
    (config_root / "strategies" / "adaptive_bollinger.yaml").write_text("body\n")
    (config_root / "hpo" / "adaptive_bollinger.yaml").write_text("body\n")
    (config_root / "universes" / f"{_UNIVERSE}.yaml").write_text("body\n")

    upload_yaml = yaml.safe_dump(
        {
            "name": _SPEC_NAME,
            "output_dir": _OUTPUT_DIR,
            "legs": [
                {
                    "strategy": _STRATEGY,
                    "strategy_config": str(
                        config_root / "strategies" / "adaptive_bollinger.yaml"
                    ),
                    "hpo_config": str(config_root / "hpo" / "adaptive_bollinger.yaml"),
                    "universes": [_UNIVERSE],
                }
            ],
        }
    )
    uploads_root = tmp_path / "uploads"
    save_upload(
        db_conn,
        user=user,
        slug=_SPEC_NAME,
        yaml_text=upload_yaml,
        uploads_root=uploads_root,
        config_root=config_root,
    )

    submission = _study_submission()
    asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=tmp_path / "store",
            config_root=config_root,
            job_temp_dir=tmp_path / "jobs",
            study_spec_uploads_dir=uploads_root,
        )
    )

    spawn_call = manager.spawn.await_args  # type: ignore[attr-defined]
    command = spawn_call.kwargs["command"]
    spec_arg = command[command.index("--spec") + 1]
    assert Path(spec_arg) == uploads_root / str(user.id) / f"{_SPEC_NAME}.yaml"


def test_submit_study_falls_back_to_library_when_no_upload(
    db_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """No matching upload for caller → resolver uses ``config/study/<slug>.yaml``."""
    user = _user(db_conn, "alice")
    manager = _stub_manager()
    config_root = tmp_path / "config"
    _write_spec(config_root, _minimal_spec_dict())
    submission = _study_submission()

    asyncio.run(
        submit_job(
            conn=db_conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=tmp_path / "store",
            config_root=config_root,
            job_temp_dir=tmp_path / "jobs",
            study_spec_uploads_dir=tmp_path / "uploads",
        )
    )

    spawn_call = manager.spawn.await_args  # type: ignore[attr-defined]
    command = spawn_call.kwargs["command"]
    spec_arg = command[command.index("--spec") + 1]
    assert Path(spec_arg) == config_root / "study" / f"{_SPEC_NAME}.yaml"
