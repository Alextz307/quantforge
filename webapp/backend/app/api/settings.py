"""Public settings endpoint: feature flags the frontend needs before login."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from webapp.backend.app.core.settings import WebappSettings, get_settings
from webapp.backend.app.schemas.settings import PublicSettings

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/public", response_model=PublicSettings)
def get_public_settings(settings: WebappSettings = Depends(get_settings)) -> PublicSettings:
    return PublicSettings(jobs_enabled=settings.jobs_enabled)
