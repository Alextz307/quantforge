"""Read-only HTTP endpoints over persisted comparisons."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from webapp.backend.app.core.deps import get_current_user
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.comparisons import ComparisonDetail, ComparisonSummary
from webapp.backend.app.services.comparison_service import (
    ComparisonNotFoundError,
    PlotNotFoundError,
    get_comparison,
    list_comparisons,
    resolve_plot,
)

router = APIRouter(
    prefix="/comparisons", tags=["comparisons"], dependencies=[Depends(get_current_user)]
)


@router.get("", response_model=list[ComparisonSummary])
def get_comparisons() -> list[ComparisonSummary]:
    return list_comparisons(get_settings().store_root)


@router.get("/{name}", response_model=ComparisonDetail)
def get_comparison_detail(name: str) -> ComparisonDetail:
    try:
        return get_comparison(get_settings().store_root, name)
    except ComparisonNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{name}/plots/{plot_name}")
def get_comparison_plot(name: str, plot_name: str) -> FileResponse:
    try:
        path = resolve_plot(get_settings().store_root, name, plot_name)
    except (ComparisonNotFoundError, PlotNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(path)
