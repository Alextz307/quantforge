"""
Read-only HTTP endpoints over persisted holdout evaluations.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from src.orchestration.holdout_eval import SourceKind
from webapp.backend.app.core.deps import get_current_user, get_db
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.holdout import (
    HoldoutEvalDetail,
    HoldoutEvalsPage,
    HoldoutSortBy,
)
from webapp.backend.app.schemas.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, SortOrder
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.holdout_service import (
    HoldoutEvalNotFoundError,
    PlotNotFoundError,
    get_holdout_eval,
    list_holdout_evals_page,
    resolve_plot,
)

router = APIRouter(prefix="/holdout-evals", tags=["holdout"])


@router.get("", response_model=HoldoutEvalsPage)
def get_holdout_evals(
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
    sort_by: HoldoutSortBy = Query(HoldoutSortBy.CREATED_AT),
    order: SortOrder = Query(SortOrder.DESC),
    source_kind: SourceKind | None = Query(None),
    since: datetime | None = Query(None),
    all_users: bool = Query(False, alias="all"),
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> HoldoutEvalsPage:
    return list_holdout_evals_page(
        get_settings().store_root,
        conn=conn,
        user=user,
        all_users=all_users,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        order=order,
        source_kind=source_kind,
        since=since,
    )


@router.get("/{name}", response_model=HoldoutEvalDetail)
def get_holdout_eval_detail(
    name: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> HoldoutEvalDetail:
    try:
        return get_holdout_eval(get_settings().store_root, name, conn=conn, user=user)
    except HoldoutEvalNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{name}/plots/{plot_name}")
def get_holdout_eval_plot(
    name: str,
    plot_name: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> FileResponse:
    try:
        path = resolve_plot(get_settings().store_root, name, plot_name, conn=conn, user=user)
    except (HoldoutEvalNotFoundError, PlotNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(path)
