"""HTTP + WebSocket endpoints over persisted HPO studies."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from starlette.websockets import WebSocketState

from webapp.backend.app.api._ws_auth import (
    WS_CLOSE_NOT_FOUND,
    WS_CLOSE_UNAUTHORIZED,
    resolve_ws_user,
)
from webapp.backend.app.core.deps import get_current_user, get_db
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.infrastructure.process_manager import HpoEventBroker
from webapp.backend.app.infrastructure.store import find_hpo_study_dir
from webapp.backend.app.schemas.hpo import (
    HpoDetail,
    HpoSummary,
    ParamImportanceResponse,
    TrialFrame,
    TrialRow,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.hpo_service import (
    HpoStudyNotFoundError,
    find_live_job_for,
    get_hpo_study,
    get_param_importance,
    list_hpo_studies,
    list_trials,
)

# Auth is wired per-endpoint, not router-level: get_current_user reads the
# Request, which doesn't exist on the WS handshake.
router = APIRouter(prefix="/hpo", tags=["hpo"])


@router.get("", response_model=list[HpoSummary])
def get_hpo_studies(_user: UserPublic = Depends(get_current_user)) -> list[HpoSummary]:
    return list_hpo_studies(get_settings().store_root)


@router.get("/{name}", response_model=HpoDetail)
def get_hpo_study_detail(
    name: str,
    _user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> HpoDetail:
    try:
        live_job_id = find_live_job_for(conn, name)
        return get_hpo_study(get_settings().store_root, name, live_job_id=live_job_id)
    except HpoStudyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{name}/trials", response_model=list[TrialRow])
def get_hpo_trials(
    name: str,
    after_trial: int | None = None,
    _user: UserPublic = Depends(get_current_user),
) -> list[TrialRow]:
    try:
        return list_trials(get_settings().store_root, name, after_trial=after_trial)
    except HpoStudyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{name}/param-importance", response_model=ParamImportanceResponse)
def get_hpo_param_importance(
    name: str,
    _user: UserPublic = Depends(get_current_user),
) -> ParamImportanceResponse:
    try:
        return get_param_importance(get_settings().store_root, name)
    except HpoStudyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.websocket("/{name}/stream")
async def stream_hpo(
    websocket: WebSocket,
    name: str,
    after_trial: int | None = None,
) -> None:
    user = resolve_ws_user(websocket)
    if user is None:
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED)
        return
    settings = get_settings()
    try:
        find_hpo_study_dir(settings.store_root, name)
    except HpoStudyNotFoundError:
        await websocket.close(code=WS_CLOSE_NOT_FOUND)
        return

    broker: HpoEventBroker = websocket.app.state.hpo_broker
    # Subscribe BEFORE reading existing trials. The trials.jsonl tailer
    # publishes via the same broker; subscribing first guarantees that any
    # line written between our snapshot read and the live loop lands in the
    # queue instead of being dropped. The replay/live boundary is then a
    # bookkeeping concern (skip rows we already replayed) rather than a race.
    queue = await broker.subscribe(name)
    await websocket.accept()
    try:
        replayed: set[int] = set()
        for row in list_trials(settings.store_root, name, after_trial=after_trial):
            await websocket.send_json(TrialFrame(trial=row).model_dump(mode="json"))
            replayed.add(row.number)
        while True:
            frame = await queue.get()
            if frame is None:
                return
            if frame.trial.number in replayed:
                continue
            if after_trial is not None and frame.trial.number <= after_trial:
                continue
            if websocket.client_state is not WebSocketState.CONNECTED:
                return
            await websocket.send_json(frame.model_dump(mode="json"))
    except WebSocketDisconnect:
        return
    finally:
        await broker.unsubscribe(name, queue)
