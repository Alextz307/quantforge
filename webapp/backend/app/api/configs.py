"""HTTP endpoints for config validation, browsing, and user-authored uploads.

Spec-upload routes raise pure service-layer exceptions
(:class:`SpecUploadNotFoundError`, :class:`SpecUploadInvalidError`,
:class:`LibrarySlugCollisionError`, :class:`PermissionError`); the
status-code mapping lives in :mod:`core.upload_handlers`, registered at
app startup. That's why the route bodies have no try/except.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from webapp.backend.app.core.deps import get_current_user, get_db
from webapp.backend.app.core.settings import WebappSettings, get_settings
from webapp.backend.app.schemas.configs import (
    ConfigDetail,
    ConfigEntry,
    ConfigKind,
    ValidateRequest,
    ValidateResponse,
)
from webapp.backend.app.schemas.study_uploads import (
    StudySpecUploadCreate,
    StudySpecUploadDetail,
    StudySpecUploadSummary,
)
from webapp.backend.app.schemas.universe_uploads import (
    UniverseSpecUploadCreate,
    UniverseSpecUploadDetail,
    UniverseSpecUploadSummary,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.config_service import (
    ConfigNotFoundError,
    get_study_spec_schema,
    get_universe_spec_schema,
    list_configs,
    read_config,
    validate,
)
from webapp.backend.app.services.study_spec_uploads import (
    get_upload,
    list_uploads,
    save_upload,
    soft_delete_upload,
    validate_study_spec_text,
)
from webapp.backend.app.services.universe_spec_uploads import (
    get_upload as get_universe_upload,
    list_uploads as list_universe_uploads,
    save_upload as save_universe_upload,
    soft_delete_upload as soft_delete_universe_upload,
    validate_universe_spec_text,
)

router = APIRouter(
    prefix="/configs",
    tags=["configs"],
    dependencies=[Depends(get_current_user)],
)


class StudySpecValidateRequest(BaseModel):
    """``POST /configs/study_spec/validate`` body — raw YAML text.

    A dedicated wire type keeps the existing ValidateRequest (parsed payload)
    uncoupled from the YAML-text path, which has to handle parse errors before
    Pydantic ever sees the body.
    """

    yaml: str


class UniverseSpecValidateRequest(BaseModel):
    """``POST /configs/universe_spec/validate`` body — raw YAML text."""

    yaml: str


@router.post("/validate", response_model=ValidateResponse)
def post_validate(request: ValidateRequest) -> ValidateResponse:
    return validate(request.kind, request.payload)


@router.post("/study_spec/validate", response_model=ValidateResponse)
def post_validate_study_spec(request: StudySpecValidateRequest) -> ValidateResponse:
    return validate_study_spec_text(request.yaml, config_root=get_settings().config_root)


@router.get("/study_spec/schema", response_model=dict[str, object])
def get_study_spec_json_schema() -> dict[str, object]:
    return get_study_spec_schema()


@router.get("/study/uploads", response_model=list[StudySpecUploadSummary])
def get_study_uploads(
    all_users: bool = Query(False, alias="all"),
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[StudySpecUploadSummary]:
    return list_uploads(conn, user=user, all_users=all_users)


@router.post(
    "/study/uploads",
    response_model=StudySpecUploadDetail,
    status_code=status.HTTP_201_CREATED,
)
def post_study_upload(
    body: StudySpecUploadCreate,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: WebappSettings = Depends(get_settings),
) -> StudySpecUploadDetail:
    return save_upload(
        conn,
        user=user,
        slug=body.slug,
        yaml_text=body.yaml,
        uploads_root=settings.study_spec_uploads_dir,
        config_root=settings.config_root,
    )


@router.get("/study/uploads/{slug}", response_model=StudySpecUploadDetail)
def get_study_upload(
    slug: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> StudySpecUploadDetail:
    return get_upload(conn, user=user, slug=slug)


@router.delete("/study/uploads/{slug}", status_code=status.HTTP_204_NO_CONTENT)
def delete_study_upload(
    slug: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: WebappSettings = Depends(get_settings),
) -> None:
    soft_delete_upload(
        conn,
        user=user,
        slug=slug,
        uploads_root=settings.study_spec_uploads_dir,
    )


@router.post("/universe_spec/validate", response_model=ValidateResponse)
def post_validate_universe_spec(request: UniverseSpecValidateRequest) -> ValidateResponse:
    return validate_universe_spec_text(request.yaml)


@router.get("/universe_spec/schema", response_model=dict[str, object])
def get_universe_spec_json_schema() -> dict[str, object]:
    return get_universe_spec_schema()


@router.get("/universe/uploads", response_model=list[UniverseSpecUploadSummary])
def get_universe_uploads(
    all_users: bool = Query(False, alias="all"),
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[UniverseSpecUploadSummary]:
    return list_universe_uploads(conn, user=user, all_users=all_users)


@router.post(
    "/universe/uploads",
    response_model=UniverseSpecUploadDetail,
    status_code=status.HTTP_201_CREATED,
)
def post_universe_upload(
    body: UniverseSpecUploadCreate,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: WebappSettings = Depends(get_settings),
) -> UniverseSpecUploadDetail:
    return save_universe_upload(
        conn,
        user=user,
        slug=body.slug,
        yaml_text=body.yaml,
        uploads_root=settings.universe_spec_uploads_dir,
        config_root=settings.config_root,
    )


@router.get("/universe/uploads/{slug}", response_model=UniverseSpecUploadDetail)
def get_universe_upload_detail(
    slug: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> UniverseSpecUploadDetail:
    return get_universe_upload(conn, user=user, slug=slug)


@router.delete("/universe/uploads/{slug}", status_code=status.HTTP_204_NO_CONTENT)
def delete_universe_upload(
    slug: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: WebappSettings = Depends(get_settings),
) -> None:
    soft_delete_universe_upload(
        conn,
        user=user,
        slug=slug,
        uploads_root=settings.universe_spec_uploads_dir,
    )


@router.get("/{kind}", response_model=list[ConfigEntry])
def get_configs(kind: ConfigKind) -> list[ConfigEntry]:
    return list_configs(get_settings().config_root, kind)


@router.get("/{kind}/{name}", response_model=ConfigDetail)
def get_config_detail(kind: ConfigKind, name: str) -> ConfigDetail:
    try:
        return read_config(get_settings().config_root, kind, name)
    except ConfigNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
