"""
Read-only HTTP endpoints over persisted comparisons.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from webapp.backend.app.core.deps import get_current_user, get_db
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.comparisons import ComparisonDetail, ComparisonSummary
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.comparison_service import (
    ComparisonNotFoundError,
    PlotNotFoundError,
    get_comparison,
    list_comparisons,
    resolve_plot,
)

router = APIRouter(prefix="/comparisons", tags=["comparisons"])


@router.get("", response_model=list[ComparisonSummary])
def get_comparisons(
    all_users: bool = Query(False, alias="all"),
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[ComparisonSummary]:
    return list_comparisons(get_settings().store_root, conn=conn, user=user, all_users=all_users)


@router.get("/{name}", response_model=ComparisonDetail)
def get_comparison_detail(
    name: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> ComparisonDetail:
    try:
        return get_comparison(get_settings().store_root, name, conn=conn, user=user)
    except ComparisonNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{name}/plots/{plot_name}")
def get_comparison_plot(
    name: str,
    plot_name: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> FileResponse:
    try:
        path = resolve_plot(get_settings().store_root, name, plot_name, conn=conn, user=user)
    except (ComparisonNotFoundError, PlotNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(path)
