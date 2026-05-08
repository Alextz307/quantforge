"""WebSocket streaming behaviour for /api/hpo/{name}/stream."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from webapp.backend.app.schemas.hpo import TrialFrame, TrialRow

STREAM_PATH = "/api/hpo/AdaptiveBollinger__spy_daily_5y/stream"
STUDY_NAME = "AdaptiveBollinger__spy_daily_5y"
EXPECTED_TRIALS = 3
LIVE_TRIAL_NUMBER = 99


def _drain_replay(ws: WebSocketTestSession, count: int) -> list[TrialFrame]:
    return [TrialFrame.model_validate(ws.receive_json()) for _ in range(count)]


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


def test_stream_forwards_live_broker_frame(authed_client: TestClient, webapp_store: Path) -> None:
    """A frame published mid-WS must arrive after the replay phase."""
    broker = authed_client.app.state.hpo_broker  # type: ignore[attr-defined]
    portal = authed_client.portal
    assert portal is not None
    live_trial = TrialRow(
        number=LIVE_TRIAL_NUMBER,
        state="COMPLETE",
        value=1.5,
        params={"window": 50, "k": 2.0},
        datetime_start=None,
        datetime_complete=None,
        experiment_id=None,
    )
    with authed_client.websocket_connect(STREAM_PATH) as ws:
        # Drain the replay phase so the next frame is guaranteed live.
        _drain_replay(ws, EXPECTED_TRIALS)
        portal.call(broker.publish, STUDY_NAME, TrialFrame(trial=live_trial))
        frame = TrialFrame.model_validate(ws.receive_json())
    assert frame.type == "trial"
    assert frame.trial.number == LIVE_TRIAL_NUMBER
