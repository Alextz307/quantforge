"""HTTP endpoints for config validation + browsing."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from webapp.backend.app.core.deps import get_current_user
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.schemas.configs import (
    ConfigDetail,
    ConfigEntry,
    ConfigKind,
    ValidateRequest,
    ValidateResponse,
)
from webapp.backend.app.services.config_service import (
    ConfigNotFoundError,
    list_configs,
    read_config,
    validate,
)

router = APIRouter(
    prefix="/configs",
    tags=["configs"],
    dependencies=[Depends(get_current_user)],
)


@router.post("/validate", response_model=ValidateResponse)
def post_validate(request: ValidateRequest) -> ValidateResponse:
    return validate(request.kind, request.payload)


@router.get("/{kind}", response_model=list[ConfigEntry])
def get_configs(kind: ConfigKind) -> list[ConfigEntry]:
    return list_configs(get_settings().config_root, kind)


@router.get("/{kind}/{name}", response_model=ConfigDetail)
def get_config_detail(kind: ConfigKind, name: str) -> ConfigDetail:
    try:
        return read_config(get_settings().config_root, kind, name)
    except ConfigNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
