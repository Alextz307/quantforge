"""End-to-end HTTP behaviour of the jobs router.

Real subprocess spawning is exercised: each test submits a payload that
bypasses the actual ``experiment run`` command via a monkeypatched
``build_run_command``, so we hit the full ProcessManager + JobStore +
LogTailer stack against a fast Python ``-c`` script.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from webapp.backend.app.core.types import Role
from webapp.backend.app.schemas.jobs import JobStatus
from webapp.backend.app.services.user_service import create_user

from ..conftest import (
    SECONDARY_PASSWORD,
    SECONDARY_USERNAME,
    TEST_PASSWORD,
    TEST_USERNAME,
    make_valid_experiment_payload,
    make_valid_job_submission,
    make_valid_tune_submission,
)

JOBS_PATH = "/api/jobs"

PAYLOAD_LINES = ["payload-line-1", "payload-line-2"]
COMPLETION_TIMEOUT_S = 15.0
POLL_INTERVAL_S = 0.05


def _quick_command_factory(
    lines: list[str],
) -> Callable[..., tuple[str, ...]]:
    """Build a ``build_run_command`` replacement that prints + exits cleanly."""
    body = "; ".join(f"print({line!r})" for line in lines)

    def fake_build(**_kwargs: object) -> tuple[str, ...]:
        return (sys.executable, "-c", body)

    return fake_build


def _slow_command_factory() -> Callable[..., tuple[str, ...]]:
    def fake_build(**_kwargs: object) -> tuple[str, ...]:
        return (sys.executable, "-c", "import time; time.sleep(60)")

    return fake_build


def _await_terminal(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + COMPLETION_TIMEOUT_S
    while time.monotonic() < deadline:
        resp = client.get(f"{JOBS_PATH}/{job_id}")
        assert resp.status_code == HTTPStatus.OK
        body: dict[str, object] = resp.json()
        if body["status"] in (
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        ):
            return body
        time.sleep(POLL_INTERVAL_S)
    pytest.fail(f"job {job_id} did not reach terminal state in {COMPLETION_TIMEOUT_S}s")


def test_post_job_returns_201_and_running(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_run_command",
        _quick_command_factory(PAYLOAD_LINES),
    )

    resp = authed_jobs_client.post(JOBS_PATH, json=make_valid_job_submission())
    assert resp.status_code == HTTPStatus.CREATED
    body = resp.json()
    assert body["status"] in (JobStatus.RUNNING.value, JobStatus.COMPLETED.value)
    assert body["pid"] is not None


def test_full_lifecycle_completes_and_logs_surface(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_run_command",
        _quick_command_factory(PAYLOAD_LINES),
    )

    resp = authed_jobs_client.post(JOBS_PATH, json=make_valid_job_submission())
    job_id = resp.json()["id"]

    final = _await_terminal(authed_jobs_client, job_id)
    assert final["status"] == JobStatus.COMPLETED.value
    assert final["exit_code"] == 0

    log_resp = authed_jobs_client.get(f"{JOBS_PATH}/{job_id}/log")
    assert log_resp.status_code == HTTPStatus.OK
    body = log_resp.text
    for line in PAYLOAD_LINES:
        assert line in body


def test_list_jobs_scopes_to_caller(
    authed_jobs_client: TestClient,
    db_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_run_command",
        _quick_command_factory(["mine"]),
    )
    create_user(db_conn, username=SECONDARY_USERNAME, password=SECONDARY_PASSWORD, role=Role.USER)

    resp = authed_jobs_client.post(JOBS_PATH, json=make_valid_job_submission())
    mine_id = resp.json()["id"]

    listing = authed_jobs_client.get(JOBS_PATH).json()
    assert {row["id"] for row in listing} == {mine_id}


def test_admin_can_list_all_with_query_flag(
    admin_jobs_client: TestClient,
    db_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_run_command",
        _quick_command_factory(["x"]),
    )
    create_user(db_conn, username=TEST_USERNAME, password=TEST_PASSWORD, role=Role.USER)

    admin_resp = admin_jobs_client.post(JOBS_PATH, json=make_valid_job_submission())
    assert admin_resp.status_code == HTTPStatus.CREATED

    full = admin_jobs_client.get(f"{JOBS_PATH}?all=1").json()
    own = admin_jobs_client.get(JOBS_PATH).json()
    assert len(full) >= len(own)


def test_non_admin_all_query_flag_403(
    authed_jobs_client: TestClient,
) -> None:
    resp = authed_jobs_client.get(f"{JOBS_PATH}?all=1")
    assert resp.status_code == HTTPStatus.FORBIDDEN


def test_get_other_users_job_403(
    authed_jobs_client: TestClient,
    db_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_run_command",
        _quick_command_factory(["x"]),
    )
    submit_resp = authed_jobs_client.post(JOBS_PATH, json=make_valid_job_submission())
    job_id = submit_resp.json()["id"]
    _await_terminal(authed_jobs_client, job_id)
    authed_jobs_client.post("/api/auth/logout")

    create_user(db_conn, username=SECONDARY_USERNAME, password=SECONDARY_PASSWORD, role=Role.USER)
    login_resp = authed_jobs_client.post(
        "/api/auth/login",
        json={"username": SECONDARY_USERNAME, "password": SECONDARY_PASSWORD},
    )
    assert login_resp.status_code == HTTPStatus.OK

    fetch_resp = authed_jobs_client.get(f"{JOBS_PATH}/{job_id}")
    assert fetch_resp.status_code == HTTPStatus.FORBIDDEN


def test_get_job_404(authed_jobs_client: TestClient) -> None:
    resp = authed_jobs_client.get(f"{JOBS_PATH}/does-not-exist")
    assert resp.status_code == HTTPStatus.NOT_FOUND


def test_cancel_running_job_transitions_to_cancelled(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_run_command",
        _slow_command_factory(),
    )
    submit = authed_jobs_client.post(JOBS_PATH, json=make_valid_job_submission()).json()
    job_id = submit["id"]
    # The slow command sleeps 60s; cancel should fire SIGTERM and reach
    # CANCELLED before the sleep finishes.
    cancel = authed_jobs_client.delete(f"{JOBS_PATH}/{job_id}")
    assert cancel.status_code == HTTPStatus.OK
    final = _await_terminal(authed_jobs_client, job_id)
    assert final["status"] == JobStatus.CANCELLED.value


def test_cancel_completed_job_returns_409(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_run_command",
        _quick_command_factory(["done"]),
    )
    submit = authed_jobs_client.post(JOBS_PATH, json=make_valid_job_submission()).json()
    job_id = submit["id"]
    _await_terminal(authed_jobs_client, job_id)
    resp = authed_jobs_client.delete(f"{JOBS_PATH}/{job_id}")
    assert resp.status_code == HTTPStatus.CONFLICT


def test_unauthenticated_post_redirects_to_401(
    jobs_client: TestClient,
) -> None:
    resp = jobs_client.post(JOBS_PATH, json=make_valid_job_submission())
    assert resp.status_code == HTTPStatus.UNAUTHORIZED


def test_post_invalid_config_payload_returns_422(authed_jobs_client: TestClient) -> None:
    """D3: validate-on-submit short-circuits before persisting any state."""
    bad_payload = make_valid_experiment_payload()
    del bad_payload["data"]
    resp = authed_jobs_client.post(JOBS_PATH, json={"kind": "run", "config_payload": bad_payload})

    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    body = resp.json()
    assert isinstance(body["detail"], list)
    assert any(item["loc"] == ["data"] for item in body["detail"])
    # Failed submissions don't show up in subsequent listings.
    listing = authed_jobs_client.get(JOBS_PATH).json()
    assert listing == []


def test_get_log_for_unstarted_log_returns_empty(
    authed_jobs_client: TestClient,
    db_conn: sqlite3.Connection,
) -> None:
    """Cover the FileResponse short-circuit when the log file doesn't exist."""
    from webapp.backend.app.infrastructure.job_store import NewJob, insert_job
    from webapp.backend.app.schemas.jobs import JobKind

    user_row = db_conn.execute(
        "SELECT id FROM users WHERE username = ?", (TEST_USERNAME,)
    ).fetchone()
    placeholder = NewJob(
        user_id=int(user_row["id"]),
        kind=JobKind.RUN,
        command=("placeholder",),
        config_path=Path("/tmp/cfg.yaml"),
        log_path=Path("/nonexistent/job.log"),
    )
    job = insert_job(db_conn, placeholder)
    resp = authed_jobs_client.get(f"{JOBS_PATH}/{job.id}/log")
    assert resp.status_code == HTTPStatus.OK
    assert resp.text == ""


def test_post_tune_job_writes_dual_yamls_and_stamps_study_name(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: POST kind=tune triggers build_tune_command (monkeypatched),
    persists experiment_id=study_name, and the spawned subprocess flows through
    the same ProcessManager log/status pipeline."""
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_tune_command",
        _quick_command_factory(["tune-line"]),
    )

    submission = make_valid_tune_submission(study_name="webapp_demo_study")
    resp = authed_jobs_client.post(JOBS_PATH, json=submission)
    assert resp.status_code == HTTPStatus.CREATED, resp.json()
    body = resp.json()
    assert body["kind"] == "tune"
    assert body["experiment_id"] == "webapp_demo_study"

    final = _await_terminal(authed_jobs_client, body["id"])
    assert final["status"] == JobStatus.COMPLETED.value


def test_post_tune_rejects_missing_hpo_payload(authed_jobs_client: TestClient) -> None:
    """The JobSubmission validator rejects kind=tune without an hpo_payload at
    the 422 wire layer (matches pydantic's missing-field semantics)."""
    resp = authed_jobs_client.post(
        JOBS_PATH,
        json={"kind": "tune", "config_payload": make_valid_experiment_payload()},
    )
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_post_tune_rejects_invalid_hpo_payload(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``hpo_payload`` that fails HPOConfig validation surfaces as a
    422 with ``loc`` paths prefixed by ``hpo_payload``."""
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_tune_command",
        _quick_command_factory(["should-not-spawn"]),
    )
    bad = make_valid_tune_submission()
    bad_hpo = cast(dict[str, object], bad["hpo_payload"])
    bad_hpo["study_name"] = "bad/name"

    resp = authed_jobs_client.post(JOBS_PATH, json=bad)
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    detail = resp.json()["detail"]
    assert any("hpo_payload" in err["loc"] for err in detail)


# Compare + holdout API integration ---------------------------------------------------------

_COMPARE_RUN_A = "20260101_120000_AdaptiveBollinger_abc1234_deadbeef"
_COMPARE_RUN_B = "20260201_090000_AdaptiveBollinger_def5678_cafebabe"


def _seed_compare_runs(
    monkeypatch: pytest.MonkeyPatch, run_ids: tuple[str, ...]
) -> Path:
    """Provision a per-test store-root + synthetic runs at ``run_ids``."""
    from webapp.backend.app.core.settings import get_settings

    from ..conftest import make_synthetic_run

    settings = get_settings()
    store_root = settings.store_root
    runs_dir = store_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for run_id in run_ids:
        make_synthetic_run(runs_dir, experiment_id=run_id)
    # Settings cache is already test-scoped via _webapp_test_env.
    return store_root


def test_post_compare_spawns_and_completes(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_compare_command",
        _quick_command_factory(["compare-line"]),
    )
    _seed_compare_runs(monkeypatch, (_COMPARE_RUN_A, _COMPARE_RUN_B))
    submission = {
        "kind": "compare",
        "compare_payload": {
            "run_ids": [_COMPARE_RUN_A, _COMPARE_RUN_B],
            "out_name": "wire_compare",
        },
    }

    resp = authed_jobs_client.post(JOBS_PATH, json=submission)
    assert resp.status_code == HTTPStatus.CREATED, resp.json()
    body = resp.json()
    assert body["kind"] == "compare"
    assert body["experiment_id"] == "wire_compare"
    final = _await_terminal(authed_jobs_client, body["id"])
    assert final["status"] == JobStatus.COMPLETED.value


def test_post_compare_rejects_unknown_run_id(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_compare_runs(monkeypatch, (_COMPARE_RUN_A,))
    resp = authed_jobs_client.post(
        JOBS_PATH,
        json={
            "kind": "compare",
            "compare_payload": {
                "run_ids": [_COMPARE_RUN_A, "ghost_id"],
                "out_name": "bad_compare",
            },
        },
    )
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    detail = resp.json()["detail"]
    assert any("ghost_id" in err["msg"] for err in detail)


def test_post_holdout_from_run_spawns_and_completes(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.core import json_io
    from src.core.persistence import EXPERIMENT_MANIFEST_JSON

    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_holdout_command",
        _quick_command_factory(["holdout-line"]),
    )
    run_id = "20260301_080000_AdaptiveBollinger_aaa0000_bbbb1111"
    store_root = _seed_compare_runs(monkeypatch, (run_id,))
    # Backfill a non-null holdout_start so the source is eligible.
    manifest_path = store_root / "runs" / run_id / EXPERIMENT_MANIFEST_JSON
    manifest = json_io.read_dict(manifest_path)
    manifest["holdout_start"] = "2024-01-01T00:00:00"
    json_io.write(manifest_path, manifest)

    submission = {
        "kind": "holdout",
        "holdout_payload": {
            "source_kind": "run",
            "source_id": run_id,
            "out_name": "wire_holdout",
        },
    }

    resp = authed_jobs_client.post(JOBS_PATH, json=submission)
    assert resp.status_code == HTTPStatus.CREATED, resp.json()
    body = resp.json()
    assert body["kind"] == "holdout"
    assert body["experiment_id"] == "wire_holdout"
    final = _await_terminal(authed_jobs_client, body["id"])
    assert final["status"] == JobStatus.COMPLETED.value


def test_post_holdout_rejects_run_without_holdout_start(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "20260401_080000_AdaptiveBollinger_ccc1111_dddd2222"
    _seed_compare_runs(monkeypatch, (run_id,))
    resp = authed_jobs_client.post(
        JOBS_PATH,
        json={
            "kind": "holdout",
            "holdout_payload": {"source_kind": "run", "source_id": run_id},
        },
    )
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    detail = resp.json()["detail"]
    assert any("no holdout boundary" in err["msg"] for err in detail)
