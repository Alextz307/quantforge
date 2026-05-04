"""Read-only HTTP endpoints over persisted regime reports."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from webapp.backend.app.core.deps import get_current_user
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.regime import RegimeReportDetail, RegimeReportSummary
from webapp.backend.app.services.regime_service import (
    PlotNotFoundError,
    RegimeReportNotFoundError,
    get_regime_report,
    list_regime_reports,
    resolve_plot,
)

router = APIRouter(
    prefix="/regime-reports", tags=["regime"], dependencies=[Depends(get_current_user)]
)


@router.get("", response_model=list[RegimeReportSummary])
def get_regime_reports() -> list[RegimeReportSummary]:
    return list_regime_reports(get_settings().store_root)


@router.get("/{name}", response_model=RegimeReportDetail)
def get_regime_report_detail(name: str) -> RegimeReportDetail:
    try:
        return get_regime_report(get_settings().store_root, name)
    except RegimeReportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{name}/plots/{plot_name}")
def get_regime_report_plot(name: str, plot_name: str) -> FileResponse:
    try:
        path = resolve_plot(get_settings().store_root, name, plot_name)
    except (RegimeReportNotFoundError, PlotNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(path)
