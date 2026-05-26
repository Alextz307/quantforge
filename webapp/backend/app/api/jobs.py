"""HTTP + WebSocket endpoints for the jobs subsystem."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketState

from webapp.backend.app.api._ws_auth import (
    WS_CLOSE_FORBIDDEN,
    WS_CLOSE_NOT_FOUND,
    WS_CLOSE_UNAUTHORIZED,
    resolve_ws_user,
)
from webapp.backend.app.core.deps import (
    get_current_user,
    get_db,
    get_job_manager,
)
from webapp.backend.app.core.settings import WebappSettings, get_settings
from webapp.backend.app.infrastructure.db import open_db
from webapp.backend.app.infrastructure.job_store import JobNotFoundError
from webapp.backend.app.infrastructure.process_manager import (
    JobEventBroker,
    ProcessManager,
)
from webapp.backend.app.schemas.jobs import (
    TERMINAL_STATUSES,
    JobRow,
    JobStatusFrame,
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
    submit_job,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobRow, status_code=status.HTTP_201_CREATED)
async def post_job(
    submission: JobSubmission,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
    manager: ProcessManager = Depends(get_job_manager),
    settings: WebappSettings = Depends(get_settings),
) -> JobRow:
    try:
        return await submit_job(
            conn=conn,
            manager=manager,
            user=user,
            submission=submission,
            store_root=settings.store_root,
            job_temp_dir=settings.job_temp_dir,
        )
    except JobConfigInvalidError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[err.model_dump() for err in exc.errors],
        ) from exc


@router.get("", response_model=list[JobRow])
def get_jobs(
    all_users: bool = Query(False, alias="all"),
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: WebappSettings = Depends(get_settings),
) -> list[JobRow]:
    try:
        return list_jobs_for(
            conn, user=user, store_root=settings.store_root, all_users=all_users
        )
    except JobNotOwnedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


@router.get("/{job_id}", response_model=JobRow)
def get_job_detail(
    job_id: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> JobRow:
    try:
        return get_job_for(conn, user=user, job_id=job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except JobNotOwnedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


@router.delete("/{job_id}", response_model=JobRow)
async def delete_job(
    job_id: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
    manager: ProcessManager = Depends(get_job_manager),
) -> JobRow:
    try:
        return await cancel_job(conn=conn, manager=manager, user=user, job_id=job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except JobNotOwnedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except JobNotRunningError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/{job_id}/log")
def get_job_log(
    job_id: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> Response:
    try:
        job = get_job_for(conn, user=user, job_id=job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except JobNotOwnedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    log_path = Path(job.log_path)
    if not log_path.is_file():
        return Response(content=b"", media_type="text/plain")
    return FileResponse(log_path, media_type="text/plain", filename=f"{job_id}.log")


@router.websocket("/{job_id}/stream")
async def stream_job(
    websocket: WebSocket,
    job_id: str,
) -> None:
    # FastAPI's WS DI only binds WebSocket-typed params; Depends(...) providers
    # that take Request fail. Resolve user/broker inline.
    user = resolve_ws_user(websocket)
    if user is None:
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED)
        return
    with open_db() as conn:
        try:
            job = get_job_for(conn, user=user, job_id=job_id)
        except JobNotFoundError:
            await websocket.close(code=WS_CLOSE_NOT_FOUND)
            return
        except JobNotOwnedError:
            await websocket.close(code=WS_CLOSE_FORBIDDEN)
            return

    broker: JobEventBroker = websocket.app.state.job_broker
    await websocket.accept()
    queue = await broker.subscribe(job_id)
    try:
        if job.status in TERMINAL_STATUSES:
            # No producer is publishing for a terminal job; emit one snapshot, exit.
            snapshot = JobStatusFrame(
                status=job.status,
                exit_code=job.exit_code,
                experiment_id=job.experiment_id,
            )
            await websocket.send_json(snapshot.model_dump())
            return
        while True:
            frame = await queue.get()
            if frame is None:
                return
            if websocket.client_state is not WebSocketState.CONNECTED:
                return
            await websocket.send_json(frame.model_dump())
    except WebSocketDisconnect:
        return
    finally:
        await broker.unsubscribe(job_id, queue)
