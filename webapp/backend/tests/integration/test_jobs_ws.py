"""WebSocket streaming behaviour for /api/jobs/{id}/stream."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.db import open_db
from webapp.backend.app.schemas.jobs import JobStatus
from webapp.backend.app.services.user_service import create_user

from ..conftest import SECONDARY_PASSWORD, SECONDARY_USERNAME, make_valid_job_submission

JOBS_PATH = "/api/jobs"

PAYLOAD_LINES = ["ws-payload-1", "ws-payload-2"]
TERMINAL_TIMEOUT_S = 15.0
POLL_INTERVAL_S = 0.05


def _quick_command_factory(lines: list[str]) -> Callable[..., tuple[str, ...]]:
    body = "; ".join(f"print({line!r})" for line in lines)

    def fake_build(**_kwargs: object) -> tuple[str, ...]:
        return (sys.executable, "-c", body)

    return fake_build


def _drain_until_status(ws: WebSocketTestSession) -> tuple[list[str], dict[str, object]]:
    log_lines: list[str] = []
    deadline = time.monotonic() + TERMINAL_TIMEOUT_S
    while time.monotonic() < deadline:
        frame = ws.receive_json()
        if frame["type"] == "log":
            log_lines.append(frame["line"])
        elif frame["type"] == "status":
            return log_lines, frame
    pytest.fail("status frame never arrived")


def _wait_until_terminal(client: TestClient, job_id: str) -> None:
    deadline = time.monotonic() + TERMINAL_TIMEOUT_S
    while time.monotonic() < deadline:
        body = client.get(f"{JOBS_PATH}/{job_id}").json()
        if body["status"] in (
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        ):
            return
        time.sleep(POLL_INTERVAL_S)
    pytest.fail(f"job {job_id} did not reach terminal in {TERMINAL_TIMEOUT_S}s")


def test_stream_emits_logs_and_terminal_status(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_run_command",
        _quick_command_factory(PAYLOAD_LINES),
    )

    submit = authed_jobs_client.post(JOBS_PATH, json=make_valid_job_submission())
    job_id = submit.json()["id"]

    with authed_jobs_client.websocket_connect(f"{JOBS_PATH}/{job_id}/stream") as ws:
        logs, status_frame = _drain_until_status(ws)

    seen = "\n".join(logs)
    for payload in PAYLOAD_LINES:
        assert payload in seen
    assert status_frame["status"] == JobStatus.COMPLETED.value
    assert status_frame["exit_code"] == 0


def test_stream_for_terminal_job_replays_snapshot(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_run_command",
        _quick_command_factory(["already-done"]),
    )
    submit = authed_jobs_client.post(JOBS_PATH, json=make_valid_job_submission()).json()
    job_id = submit["id"]
    _wait_until_terminal(authed_jobs_client, job_id)

    with authed_jobs_client.websocket_connect(f"{JOBS_PATH}/{job_id}/stream") as ws:
        snapshot = ws.receive_json()
    assert snapshot["type"] == "status"
    assert snapshot["status"] == JobStatus.COMPLETED.value


def test_stream_unauthenticated_closes(jobs_client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect):
        with jobs_client.websocket_connect(f"{JOBS_PATH}/anything/stream") as ws:
            ws.receive_json()


def test_stream_other_users_job_closes(
    authed_jobs_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "webapp.backend.app.services.job_service.build_run_command",
        _quick_command_factory(["x"]),
    )
    submit = authed_jobs_client.post(JOBS_PATH, json=make_valid_job_submission()).json()
    job_id = submit["id"]
    authed_jobs_client.post("/api/auth/logout")

    with open_db() as conn:
        create_user(conn, username=SECONDARY_USERNAME, password=SECONDARY_PASSWORD, role=Role.USER)
    assert (
        authed_jobs_client.post(
            "/api/auth/login",
            json={"username": SECONDARY_USERNAME, "password": SECONDARY_PASSWORD},
        ).status_code
        == HTTPStatus.OK
    )

    with pytest.raises(WebSocketDisconnect):
        with authed_jobs_client.websocket_connect(f"{JOBS_PATH}/{job_id}/stream") as ws:
            ws.receive_json()


def test_stream_when_jobs_disabled_closes(authed_client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect):
        with authed_client.websocket_connect(f"{JOBS_PATH}/anything/stream") as ws:
            ws.receive_json()
