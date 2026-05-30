"""
User-authored study-spec uploads: validate, persist, soft-delete.

The CRUD shape is owned by :class:`SpecUploadStore` in
:mod:`spec_upload_store`; this module supplies the study-specific
validator (Pydantic schema + referenced-file existence) and the public
free-function API that callers use.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.core.config import StudySpec
from webapp.backend.app.schemas.configs import ValidateResponse, ValidationErrorItem
from webapp.backend.app.schemas.study_uploads import (
    StudySpecUploadDetail,
    StudySpecUploadSummary,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.spec_upload_store import (
    LibrarySlugCollisionError,
    SpecUploadInvalidError,
    SpecUploadNotFoundError,
    SpecUploadStore,
    find_upload_path,
    parse_yaml_mapping,
    validate_against_pydantic,
)


class StudySpecUploadNotFoundError(SpecUploadNotFoundError):
    """
    Raised when a slug is not present (or is soft-deleted) for the caller.
    """

    kind_label = "study"


class StudySpecUploadInvalidError(SpecUploadInvalidError):
    def __init__(self, errors: list[ValidationErrorItem]) -> None:
        super().__init__("study", errors)


def validate_study_spec_text(yaml_text: str, *, config_root: Path) -> ValidateResponse:
    """
    YAML parse + StudySpec pydantic validation + referenced-file existence.

    Path-existence errors carry the path-shaped ``loc`` ``["legs", i,
    "<field>"]`` so the editor can mark the offending line. Empty /
    non-mapping YAML is rejected at the root.
    """

    parsed, parse_errors = parse_yaml_mapping(yaml_text)
    if parsed is None:
        return ValidateResponse(valid=False, errors=parse_errors)
    schema_errors = validate_against_pydantic(parsed, StudySpec)
    if schema_errors:
        return ValidateResponse(valid=False, errors=schema_errors)
    spec = StudySpec.model_validate(parsed)
    path_errors = _check_referenced_paths(spec, config_root)
    return ValidateResponse(valid=not path_errors, errors=path_errors)


def _check_referenced_paths(spec: StudySpec, config_root: Path) -> list[ValidationErrorItem]:
    """
    For every leg, verify ``strategy_config``, ``hpo_config``, and each
    universe slug resolve to a file under ``config_root``.

    Universe slugs are stored as bare names; the canonical layout is
    ``config/universes/<slug>.yaml``.
    """

    errors: list[ValidationErrorItem] = []
    universes_dir = config_root / "universes"
    for leg_idx, leg in enumerate(spec.legs):
        if not Path(leg.strategy_config).is_file():
            errors.append(
                ValidationErrorItem(
                    loc=["legs", str(leg_idx), "strategy_config"],
                    msg=f"file not found: {leg.strategy_config}",
                    type="value_error",
                )
            )
        if not Path(leg.hpo_config).is_file():
            errors.append(
                ValidationErrorItem(
                    loc=["legs", str(leg_idx), "hpo_config"],
                    msg=f"file not found: {leg.hpo_config}",
                    type="value_error",
                )
            )
        for u_idx, universe in enumerate(leg.universes):
            if not (universes_dir / f"{universe}.yaml").is_file():
                errors.append(
                    ValidationErrorItem(
                        loc=["legs", str(leg_idx), "universes", str(u_idx)],
                        msg=f"unknown universe: {universe}",
                        type="value_error",
                    )
                )
    return errors


def _store_validator(yaml_text: str, config_root: Path) -> ValidateResponse:
    return validate_study_spec_text(yaml_text, config_root=config_root)


_store: SpecUploadStore[StudySpecUploadSummary, StudySpecUploadDetail] = SpecUploadStore(
    table_name="study_spec_uploads",
    library_subdir="study",
    summary_cls=StudySpecUploadSummary,
    detail_cls=StudySpecUploadDetail,
    not_found_error=StudySpecUploadNotFoundError,
    invalid_error=StudySpecUploadInvalidError,
    validator=_store_validator,
)


def list_uploads(
    conn: sqlite3.Connection, *, user: UserPublic, all_users: bool = False
) -> list[StudySpecUploadSummary]:
    return _store.list_uploads(conn, user=user, all_users=all_users)


def get_upload(conn: sqlite3.Connection, *, user: UserPublic, slug: str) -> StudySpecUploadDetail:
    return _store.get_upload(conn, user=user, slug=slug)


def save_upload(
    conn: sqlite3.Connection,
    *,
    user: UserPublic,
    slug: str,
    yaml_text: str,
    uploads_root: Path,
    config_root: Path,
) -> StudySpecUploadDetail:
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
    "LibrarySlugCollisionError",
    "StudySpecUploadInvalidError",
    "StudySpecUploadNotFoundError",
    "find_upload_path",
    "get_upload",
    "list_uploads",
    "save_upload",
    "soft_delete_upload",
    "validate_study_spec_text",
]
