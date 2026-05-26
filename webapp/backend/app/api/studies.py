"""Read-only HTTP endpoints + live WebSocket stream over persisted studies."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketState

from webapp.backend.app.api._ws_auth import (
    WS_CLOSE_NOT_FOUND,
    WS_CLOSE_UNAUTHORIZED,
    resolve_ws_user,
)
from webapp.backend.app.core.deps import get_current_user
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.infrastructure.store import find_study_dir
from webapp.backend.app.schemas.studies import (
    StudyConsolidatedDTO,
    StudyDetail,
    StudySummary,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.study_service import (
    ConsolidatedReportNotFoundError,
    PlotNotFoundError,
    StudyConsolidationError,
    StudyNotFoundError,
    generate_consolidated,
    get_consolidated,
    get_study,
    list_studies,
    resolve_consolidated_plot,
    resolve_consolidated_table,
)
from webapp.backend.app.services.study_stream import tail_study_state

router = APIRouter(prefix="/studies", tags=["studies"])


@router.get("", response_model=list[StudySummary])
def get_studies(_user: UserPublic = Depends(get_current_user)) -> list[StudySummary]:
    return list_studies(get_settings().store_root)


@router.get("/{name}", response_model=StudyDetail)
def get_study_detail(name: str, _user: UserPublic = Depends(get_current_user)) -> StudyDetail:
    try:
        return get_study(get_settings().store_root, name)
    except StudyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{name}/consolidated", response_model=StudyConsolidatedDTO)
def get_study_consolidated(
    name: str, _user: UserPublic = Depends(get_current_user)
) -> StudyConsolidatedDTO:
    try:
        return get_consolidated(get_settings().store_root, name)
    except (StudyNotFoundError, ConsolidatedReportNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/{name}/consolidated", response_model=StudyConsolidatedDTO)
async def post_study_consolidated(
    name: str, _user: UserPublic = Depends(get_current_user)
) -> StudyConsolidatedDTO:
    """Build (or rebuild) the consolidated report for a study and return it.

    Runs the matplotlib + table-writing work in the threadpool so the event
    loop stays responsive while the (few-seconds) job completes. A study
    that hasn't completed any legs yet returns 422.
    """
    try:
        return await run_in_threadpool(generate_consolidated, get_settings().store_root, name)
    except StudyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except StudyConsolidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.get("/{name}/consolidated/plots/{plot_name}")
def get_study_consolidated_plot(
    name: str, plot_name: str, _user: UserPublic = Depends(get_current_user)
) -> FileResponse:
    try:
        path = resolve_consolidated_plot(get_settings().store_root, name, plot_name)
    except (StudyNotFoundError, PlotNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(path)


@router.get("/{name}/consolidated/tables/{table_name}")
def get_study_consolidated_table(
    name: str, table_name: str, _user: UserPublic = Depends(get_current_user)
) -> FileResponse:
    try:
        path = resolve_consolidated_table(get_settings().store_root, name, table_name)
    except (StudyNotFoundError, PlotNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(path)


@router.websocket("/{name}/stream")
async def stream_study(websocket: WebSocket, name: str) -> None:
    """Push ``StudyDetail`` frames on every ``study_state.json`` mtime bump.

    Per-connection mtime polling (1.0s tick) — no shared broker. Both
    webapp-launched and CLI-launched studies surface; the watcher cares
    about disk state, not whose process is writing.
    """
    user = resolve_ws_user(websocket)
    if user is None:
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED)
        return
    settings = get_settings()
    try:
        study_dir = find_study_dir(settings.store_root, name)
    except StudyNotFoundError:
        await websocket.close(code=WS_CLOSE_NOT_FOUND)
        return
    await websocket.accept()
    stop = asyncio.Event()
    try:
        async for detail in tail_study_state(study_dir, stop=stop):
            if websocket.client_state is not WebSocketState.CONNECTED:
                return
            await websocket.send_json(detail.model_dump(mode="json"))
    except WebSocketDisconnect:
        return
    finally:
        stop.set()
