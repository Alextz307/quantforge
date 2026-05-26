"""Read-only HTTP endpoints over persisted studies."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from webapp.backend.app.core.deps import get_current_user
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.studies import (
    StudyConsolidatedDTO,
    StudyDetail,
    StudySummary,
)
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

router = APIRouter(prefix="/studies", tags=["studies"], dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[StudySummary])
def get_studies() -> list[StudySummary]:
    return list_studies(get_settings().store_root)


@router.get("/{name}", response_model=StudyDetail)
def get_study_detail(name: str) -> StudyDetail:
    try:
        return get_study(get_settings().store_root, name)
    except StudyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{name}/consolidated", response_model=StudyConsolidatedDTO)
def get_study_consolidated(name: str) -> StudyConsolidatedDTO:
    try:
        return get_consolidated(get_settings().store_root, name)
    except (StudyNotFoundError, ConsolidatedReportNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/{name}/consolidated", response_model=StudyConsolidatedDTO)
async def post_study_consolidated(name: str) -> StudyConsolidatedDTO:
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
def get_study_consolidated_plot(name: str, plot_name: str) -> FileResponse:
    try:
        path = resolve_consolidated_plot(get_settings().store_root, name, plot_name)
    except (StudyNotFoundError, PlotNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(path)


@router.get("/{name}/consolidated/tables/{table_name}")
def get_study_consolidated_table(name: str, table_name: str) -> FileResponse:
    try:
        path = resolve_consolidated_table(get_settings().store_root, name, table_name)
    except (StudyNotFoundError, PlotNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(path)
