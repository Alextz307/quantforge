"""Read-only HTTP endpoints over persisted runs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from webapp.backend.app.core.deps import get_current_user
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.runs import FoldRow, RunDetail, RunSummary
from webapp.backend.app.services.run_service import (
    PlotNotFoundError,
    RunNotFoundError,
    get_folds,
    get_run,
    list_runs,
    resolve_plot,
)

router = APIRouter(prefix="/runs", tags=["runs"], dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[RunSummary])
def get_runs() -> list[RunSummary]:
    return list_runs(get_settings().store_root)


@router.get("/{experiment_id}", response_model=RunDetail)
def get_run_detail(experiment_id: str) -> RunDetail:
    try:
        return get_run(get_settings().store_root, experiment_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{experiment_id}/folds", response_model=list[FoldRow])
def get_run_folds(experiment_id: str) -> list[FoldRow]:
    try:
        return get_folds(get_settings().store_root, experiment_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{experiment_id}/plots/{plot_name}")
def get_run_plot(experiment_id: str, plot_name: str) -> FileResponse:
    try:
        path = resolve_plot(get_settings().store_root, experiment_id, plot_name)
    except (RunNotFoundError, PlotNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(path)
