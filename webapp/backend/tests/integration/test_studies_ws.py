"""WebSocket streaming behaviour for /api/studies/{name}/stream."""

from __future__ import annotations

import os
import time
from http import HTTPStatus
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.orchestration.study import STUDY_STATE_FILENAME

STREAM_PATH = "/api/studies/main/stream"
STUDY_NAME = "main"
MTIME_BUMP_SECONDS = 2.0


def _bump_mtime(path: Path) -> None:
    """Push the file's mtime forward so the watcher detects a change."""

    now = time.time() + MTIME_BUMP_SECONDS
    os.utime(path, (now, now))


def _state_path(webapp_store: Path) -> Path:
    return webapp_store / "studies" / STUDY_NAME / STUDY_STATE_FILENAME


def test_stream_unauthenticated_closes(client: TestClient, webapp_store: Path) -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(STREAM_PATH) as ws:
            ws.receive_json()


def test_stream_unknown_study_closes(authed_client: TestClient, webapp_store: Path) -> None:
    with pytest.raises(WebSocketDisconnect):
        with authed_client.websocket_connect("/api/studies/missing/stream") as ws:
            ws.receive_json()


def test_stream_emits_initial_snapshot(authed_client: TestClient, webapp_store: Path) -> None:
    """First frame on connect is the current ``StudyDetail`` snapshot."""

    with authed_client.websocket_connect(STREAM_PATH) as ws:
        snapshot = ws.receive_json()
    assert snapshot["name"] == STUDY_NAME
    assert snapshot["total_legs"] == 2
    assert "legs" in snapshot
    assert len(snapshot["legs"]) == 2


def test_stream_emits_frame_after_mtime_bump(authed_client: TestClient, webapp_store: Path) -> None:
    """A change to ``study_state.json`` mtime triggers a fresh snapshot frame."""

    state_path = _state_path(webapp_store)
    assert state_path.exists()
    with authed_client.websocket_connect(STREAM_PATH) as ws:
        first = ws.receive_json()
        _bump_mtime(state_path)
        second = ws.receive_json()
    assert first["name"] == STUDY_NAME
    assert second["name"] == STUDY_NAME


def test_detail_endpoint_still_returns_200(authed_client: TestClient, webapp_store: Path) -> None:
    """Sanity: the HTTP GET endpoint still works alongside the WS surface."""

    response = authed_client.get(f"/api/studies/{STUDY_NAME}")
    assert response.status_code == HTTPStatus.OK
