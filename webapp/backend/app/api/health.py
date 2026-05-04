"""Liveness probe for the webapp backend."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from webapp.backend.app.core.version import APP_VERSION


class HealthResponse(BaseModel):
    status: str
    version: str


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=APP_VERSION)
