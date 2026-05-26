"""WebSocket streaming behaviour for /api/hpo/{wire_id}/stream."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from src.optimization.checkpointing import TRIALS_JSONL_NAME
from webapp.backend.app.schemas.hpo import TrialFrame

WIRE_ID = "studies~main~hpo~AdaptiveBollinger__spy_daily_5y"
STREAM_PATH = f"/api/hpo/{WIRE_ID}/stream"
STUDY_NAME = "AdaptiveBollinger__spy_daily_5y"
EXPECTED_TRIALS = 3
LIVE_TRIAL_NUMBER = 99


def _drain_replay(ws: WebSocketTestSession, count: int) -> list[TrialFrame]:
    return [TrialFrame.model_validate(ws.receive_json()) for _ in range(count)]


def _trials_jsonl_path(webapp_store: Path) -> Path:
    return webapp_store / "studies" / "main" / "hpo" / STUDY_NAME / TRIALS_JSONL_NAME


def test_stream_replays_existing_trials(authed_client: TestClient, webapp_store: Path) -> None:
    with authed_client.websocket_connect(STREAM_PATH) as ws:
        frames = _drain_replay(ws, EXPECTED_TRIALS)
    assert [f.trial.number for f in frames] == list(range(EXPECTED_TRIALS))
    assert all(f.type == "trial" for f in frames)


def test_stream_filters_by_after_trial(authed_client: TestClient, webapp_store: Path) -> None:
    with authed_client.websocket_connect(f"{STREAM_PATH}?after_trial=0") as ws:
        frames = _drain_replay(ws, EXPECTED_TRIALS - 1)
    assert [f.trial.number for f in frames] == [1, 2]


def test_stream_unauthenticated_closes(client: TestClient, webapp_store: Path) -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(STREAM_PATH) as ws:
            ws.receive_json()


def test_stream_unknown_study_closes(authed_client: TestClient, webapp_store: Path) -> None:
    with pytest.raises(WebSocketDisconnect):
        with authed_client.websocket_connect("/api/hpo/missing/stream") as ws:
            ws.receive_json()


def test_stream_forwards_appended_trial(authed_client: TestClient, webapp_store: Path) -> None:
    """A trial appended to ``trials.jsonl`` mid-WS must arrive after the replay phase."""
    trials_path = _trials_jsonl_path(webapp_store)
    live_record = {
        "number": LIVE_TRIAL_NUMBER,
        "state": "COMPLETE",
        "value": 1.5,
        "params": {"window": 50, "k": 2.0},
        "datetime_start": None,
        "datetime_complete": None,
        "user_attrs": {},
    }
    with authed_client.websocket_connect(STREAM_PATH) as ws:
        # Drain the replay phase so the next frame is guaranteed live.
        _drain_replay(ws, EXPECTED_TRIALS)
        # File-tailer picks up the appended line on its next poll tick.
        with trials_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(live_record) + "\n")
        frame = TrialFrame.model_validate(ws.receive_json())
    assert frame.type == "trial"
    assert frame.trial.number == LIVE_TRIAL_NUMBER
