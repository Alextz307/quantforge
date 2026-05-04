"""Read-only HTTP endpoints over persisted studies."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from webapp.backend.app.core.deps import get_current_user
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.studies import StudyDetail, StudySummary
from webapp.backend.app.services.study_service import (
    StudyNotFoundError,
    get_study,
    list_studies,
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
