"""Read-only HTTP endpoints over persisted holdout evaluations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from webapp.backend.app.core.deps import get_current_user
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.holdout import HoldoutEvalDetail, HoldoutEvalSummary
from webapp.backend.app.services.holdout_service import (
    HoldoutEvalNotFoundError,
    PlotNotFoundError,
    get_holdout_eval,
    list_holdout_evals,
    resolve_plot,
)

router = APIRouter(
    prefix="/holdout-evals", tags=["holdout"], dependencies=[Depends(get_current_user)]
)


@router.get("", response_model=list[HoldoutEvalSummary])
def get_holdout_evals() -> list[HoldoutEvalSummary]:
    return list_holdout_evals(get_settings().store_root)


@router.get("/{name}", response_model=HoldoutEvalDetail)
def get_holdout_eval_detail(name: str) -> HoldoutEvalDetail:
    try:
        return get_holdout_eval(get_settings().store_root, name)
    except HoldoutEvalNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{name}/plots/{plot_name}")
def get_holdout_eval_plot(name: str, plot_name: str) -> FileResponse:
    try:
        path = resolve_plot(get_settings().store_root, name, plot_name)
    except (HoldoutEvalNotFoundError, PlotNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(path)
