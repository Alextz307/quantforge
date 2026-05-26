"""HTTP endpoints for config validation, browsing, and user-authored uploads."""

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
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.config_service import (
    ConfigNotFoundError,
    get_study_spec_schema,
    list_configs,
    read_config,
    validate,
)
from webapp.backend.app.services.study_spec_uploads import (
    LibrarySlugCollisionError,
    StudySpecUploadInvalidError,
    StudySpecUploadNotFoundError,
    get_upload,
    list_uploads,
    save_upload,
    soft_delete_upload,
    validate_study_spec_text,
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
    try:
        return list_uploads(conn, user=user, all_users=all_users)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


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
    try:
        return save_upload(
            conn,
            user=user,
            slug=body.slug,
            yaml_text=body.yaml,
            uploads_root=settings.study_spec_uploads_dir,
            config_root=settings.config_root,
        )
    except LibrarySlugCollisionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"slug '{exc.slug}' shadows a library spec at "
                f"config/study/{exc.slug}.yaml — pick a different slug"
            ),
        ) from exc
    except StudySpecUploadInvalidError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[err.model_dump() for err in exc.errors],
        ) from exc


@router.get("/study/uploads/{slug}", response_model=StudySpecUploadDetail)
def get_study_upload(
    slug: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> StudySpecUploadDetail:
    try:
        return get_upload(conn, user=user, slug=slug)
    except StudySpecUploadNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"study spec upload not found: {slug}",
        ) from exc


@router.delete("/study/uploads/{slug}", status_code=status.HTTP_204_NO_CONTENT)
def delete_study_upload(
    slug: str,
    user: UserPublic = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
    settings: WebappSettings = Depends(get_settings),
) -> None:
    try:
        soft_delete_upload(
            conn,
            user=user,
            slug=slug,
            uploads_root=settings.study_spec_uploads_dir,
        )
    except StudySpecUploadNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"study spec upload not found: {slug}",
        ) from exc


@router.get("/{kind}", response_model=list[ConfigEntry])
def get_configs(kind: ConfigKind) -> list[ConfigEntry]:
    return list_configs(get_settings().config_root, kind)


@router.get("/{kind}/{name}", response_model=ConfigDetail)
def get_config_detail(kind: ConfigKind, name: str) -> ConfigDetail:
    try:
        return read_config(get_settings().config_root, kind, name)
    except ConfigNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
