"""
Read-only HTTP endpoints over persisted holdout evaluations.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from webapp.backend.app.core.deps import get_current_user, get_db
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.holdout import HoldoutEvalDetail, HoldoutEvalSummary
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.holdout_service import (
    HoldoutEvalNotFoundError,
    PlotNotFoundError,
    get_holdout_eval,
    list_holdout_evals,
    resolve_plot,
)

router = APIRouter(prefix="/holdout-evals", tags=["holdout"])


@router.get("", response_model=list[HoldoutEvalSummary])
def get_holdout_evals(
    all_users: bool = Query(False, alias="all"),
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[HoldoutEvalSummary]:
    return list_holdout_evals(get_settings().store_root, conn=conn, user=user, all_users=all_users)


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
