"""
HTTP + WebSocket endpoints over persisted HPO studies.
"""

from __future__ import annotations

import asyncio
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from starlette.websockets import WebSocketState

from webapp.backend.app.api._ws_auth import (
    WS_CLOSE_NOT_FOUND,
    WS_CLOSE_UNAUTHORIZED,
    resolve_ws_user,
)
from webapp.backend.app.core.deps import get_current_user, get_db
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.infrastructure.store import find_hpo_study_dir_by_wire_id
from webapp.backend.app.schemas.hpo import (
    HpoDetail,
    HpoSummary,
    ParamImportanceResponse,
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
from webapp.backend.app.services.hpo_stream import tail_hpo_trials

# Auth is wired per-endpoint, not router-level: get_current_user reads the
# Request, which doesn't exist on the WS handshake.
router = APIRouter(prefix="/hpo", tags=["hpo"])


@router.get("", response_model=list[HpoSummary])
def get_hpo_studies(
    all_users: bool = Query(False, alias="all"),
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[HpoSummary]:
    return list_hpo_studies(
        get_settings().store_root, conn=conn, user=user, all_users=all_users
    )


@router.get("/{wire_id}", response_model=HpoDetail)
def get_hpo_study_detail(
    wire_id: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> HpoDetail:
    try:
        live_job_id = find_live_job_for(conn, wire_id)
        return get_hpo_study(
            get_settings().store_root,
            wire_id,
            conn=conn,
            user=user,
            live_job_id=live_job_id,
        )
    except HpoStudyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{wire_id}/trials", response_model=list[TrialRow])
def get_hpo_trials(
    wire_id: str,
    after_trial: int | None = None,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[TrialRow]:
    try:
        return list_trials(
            get_settings().store_root,
            wire_id,
            conn=conn,
            user=user,
            after_trial=after_trial,
        )
    except HpoStudyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{wire_id}/param-importance", response_model=ParamImportanceResponse)
def get_hpo_param_importance(
    wire_id: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> ParamImportanceResponse:
    try:
        return get_param_importance(
            get_settings().store_root, wire_id, conn=conn, user=user
        )
    except HpoStudyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.websocket("/{wire_id}/stream")
async def stream_hpo(
    websocket: WebSocket,
    wire_id: str,
    after_trial: int | None = None,
) -> None:
    """
    Push ``TrialFrame``s as new lines land in ``trials.jsonl``.

    Per-connection file tailer — works whether the producer is a webapp
    tune subprocess, the study orchestrator writing a nested HPO leg, or
    a CLI invocation. ``tail_hpo_trials`` replays existing lines from
    byte 0 then yields live appends through the same loop, so the WS
    handler doesn't have to deduplicate against a replay set.
    """

    user = resolve_ws_user(websocket)
    if user is None:
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED)
        return
    settings = get_settings()
    try:
        study_dir = find_hpo_study_dir_by_wire_id(settings.store_root, wire_id)
    except HpoStudyNotFoundError:
        await websocket.close(code=WS_CLOSE_NOT_FOUND)
        return
    await websocket.accept()
    stop = asyncio.Event()
    try:
        async for frame in tail_hpo_trials(study_dir, stop=stop, after_trial=after_trial):
            if websocket.client_state is not WebSocketState.CONNECTED:
                return
            await websocket.send_json(frame.model_dump(mode="json"))
    except WebSocketDisconnect:
        return
    finally:
        stop.set()
