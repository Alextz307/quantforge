"""
User-authored universe-spec uploads: validate, persist, soft-delete.

The CRUD shape is owned by :class:`SpecUploadStore` in
:mod:`spec_upload_store`; this module supplies the universe-specific
validator (a Pydantic check against :class:`UniverseProfile` - universe
specs are self-contained so there are no referenced-file checks) and
the public free-function API.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.core.config import UniverseProfile
from webapp.backend.app.schemas.configs import ValidateResponse, ValidationErrorItem
from webapp.backend.app.schemas.universe_uploads import (
    UniverseSpecUploadDetail,
    UniverseSpecUploadSummary,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.spec_upload_store import (
    SpecUploadInvalidError,
    SpecUploadNotFoundError,
    SpecUploadStore,
    find_upload_path,
    parse_yaml_mapping,
    validate_against_pydantic,
)


class UniverseSpecUploadNotFoundError(SpecUploadNotFoundError):
    """
    Raised when a slug is absent (or soft-deleted) for the caller.
    """

    kind_label = "universe"


class UniverseSpecUploadInvalidError(SpecUploadInvalidError):
    def __init__(self, errors: list[ValidationErrorItem]) -> None:
        super().__init__("universe", errors)


def validate_universe_spec_text(yaml_text: str) -> ValidateResponse:
    """
    YAML parse + UniverseProfile pydantic validation.

    Errors carry the ``loc`` shape from pydantic ``ValidationError.errors()``
    so the editor can mark the offending line. Empty / non-mapping YAML is
    rejected at the root.
    """

    parsed, parse_errors = parse_yaml_mapping(yaml_text)
    if parsed is None:
        return ValidateResponse(valid=False, errors=parse_errors)
    schema_errors = validate_against_pydantic(parsed, UniverseProfile)
    return ValidateResponse(valid=not schema_errors, errors=schema_errors)


def _store_validator(yaml_text: str, _config_root: Path) -> ValidateResponse:
    return validate_universe_spec_text(yaml_text)


_store: SpecUploadStore[UniverseSpecUploadSummary, UniverseSpecUploadDetail] = SpecUploadStore(
    table_name="universe_spec_uploads",
    library_subdir="universes",
    summary_cls=UniverseSpecUploadSummary,
    detail_cls=UniverseSpecUploadDetail,
    not_found_error=UniverseSpecUploadNotFoundError,
    invalid_error=UniverseSpecUploadInvalidError,
    validator=_store_validator,
)


def list_uploads(
    conn: sqlite3.Connection, *, user: UserPublic, all_users: bool = False
) -> list[UniverseSpecUploadSummary]:
    return _store.list_uploads(conn, user=user, all_users=all_users)


def get_upload(
    conn: sqlite3.Connection, *, user: UserPublic, slug: str
) -> UniverseSpecUploadDetail:
    return _store.get_upload(conn, user=user, slug=slug)


def save_upload(
    conn: sqlite3.Connection,
    *,
    user: UserPublic,
    slug: str,
    yaml_text: str,
    uploads_root: Path,
    config_root: Path,
) -> UniverseSpecUploadDetail:
    return _store.save_upload(
        conn,
        user=user,
        slug=slug,
        yaml_text=yaml_text,
        uploads_root=uploads_root,
        config_root=config_root,
    )


def soft_delete_upload(
    conn: sqlite3.Connection,
    *,
    user: UserPublic,
    slug: str,
    uploads_root: Path,
) -> None:
    _store.soft_delete_upload(conn, user=user, slug=slug, uploads_root=uploads_root)


__all__ = [
    "UniverseSpecUploadInvalidError",
    "UniverseSpecUploadNotFoundError",
    "find_upload_path",
    "get_upload",
    "list_uploads",
    "save_upload",
    "soft_delete_upload",
    "validate_universe_spec_text",
]
