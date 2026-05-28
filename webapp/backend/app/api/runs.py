"""
Read-only HTTP endpoints over persisted runs.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from webapp.backend.app.core.deps import get_current_user, get_db
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.runs import (
    FoldRow,
    RunDetail,
    RunSortBy,
    RunsPage,
    SortOrder,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.run_service import (
    PlotNotFoundError,
    RunNotFoundError,
    get_folds,
    get_run,
    list_runs_page,
    resolve_plot,
)

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("", response_model=RunsPage)
def get_runs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort_by: RunSortBy = Query(RunSortBy.CREATED_AT),
    order: SortOrder = Query(SortOrder.DESC),
    strategy: str | None = Query(None),
    ticker: str | None = Query(None),
    since: datetime | None = Query(None),
    all_users: bool = Query(False, alias="all"),
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> RunsPage:
    return list_runs_page(
        get_settings().store_root,
        conn=conn,
        user=user,
        all_users=all_users,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        order=order,
        strategy=strategy,
        ticker=ticker,
        since=since,
    )


@router.get("/{experiment_id}", response_model=RunDetail)
def get_run_detail(
    experiment_id: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> RunDetail:
    try:
        return get_run(get_settings().store_root, experiment_id, conn=conn, user=user)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{experiment_id}/folds", response_model=list[FoldRow])
def get_run_folds(
    experiment_id: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[FoldRow]:
    try:
        return get_folds(get_settings().store_root, experiment_id, conn=conn, user=user)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{experiment_id}/plots/{plot_name}")
def get_run_plot(
    experiment_id: str,
    plot_name: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> FileResponse:
    try:
        path = resolve_plot(
            get_settings().store_root, experiment_id, plot_name, conn=conn, user=user
        )
    except (RunNotFoundError, PlotNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(path)
