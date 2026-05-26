"""User-authored study-spec uploads: validate, persist, soft-delete.

The DB row is authoritative for listing + soft-delete; the on-disk YAML at
``<uploads_root>/<user_id>/<slug>.yaml`` is the subprocess interface — the
study CLI consumes a path. Saves write disk then commit; deletes commit then
unlink. Either order leaves the worst-case state recoverable on the next
save.

A user-uploaded slug must not collide with a library spec at
``config/study/<slug>.yaml``. The collision check is part of the save path,
not just an editor convenience, so the `_StudyHandler` resolver can rely on
upload lookups never shadowing library entries.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.core.config import StudySpec
from src.core.fs import ensure_parent_dir
from webapp.backend.app.core.types import Role
from webapp.backend.app.schemas.configs import ValidateResponse, ValidationErrorItem
from webapp.backend.app.schemas.study_uploads import (
    StudySpecUploadDetail,
    StudySpecUploadSummary,
)
from webapp.backend.app.schemas.users import UserPublic

_STUDY_LIBRARY_SUBDIR = "study"


class StudySpecUploadNotFoundError(LookupError):
    """Raised when a slug is not present (or is soft-deleted) for the caller."""


class StudySpecUploadInvalidError(ValueError):
    """Raised when an upload payload fails YAML parse / schema / path checks.

    Carries the same ``ValidationErrorItem`` shape the jobs API uses so the
    router can surface inline errors verbatim under the editor.
    """

    def __init__(self, errors: list[ValidationErrorItem]) -> None:
        super().__init__(f"invalid study spec upload ({len(errors)} error(s))")
        self.errors = errors


class LibrarySlugCollisionError(ValueError):
    """Raised when a save would shadow an existing ``config/study/<slug>.yaml``."""

    def __init__(self, slug: str) -> None:
        super().__init__(slug)
        self.slug = slug


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _upload_path(uploads_root: Path, user_id: int, slug: str) -> Path:
    return uploads_root / str(user_id) / f"{slug}.yaml"


def find_upload_path(uploads_root: Path, user_id: int, slug: str) -> Path | None:
    """Return the on-disk YAML path for an upload if the file exists, else None.

    Pure filesystem lookup — does not touch the DB. Used by ``_StudyHandler``
    on the spawn path where the row has already been verified live by a
    prior validate() check.
    """
    path = _upload_path(uploads_root, user_id, slug)
    return path if path.is_file() else None


def validate_study_spec_text(
    yaml_text: str, *, config_root: Path
) -> ValidateResponse:
    """YAML parse + StudySpec pydantic validation + referenced-file existence.

    Path-existence errors carry the path-shaped ``loc`` ``["legs", i, "<field>"]``
    so the editor can mark the offending line. Empty / non-mapping YAML is
    rejected at the root.
    """
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return ValidateResponse(
            valid=False,
            errors=[
                ValidationErrorItem(
                    loc=["yaml"], msg=f"YAML parse error: {exc}", type="value_error"
                )
            ],
        )
    if parsed is None:
        return ValidateResponse(
            valid=False,
            errors=[
                ValidationErrorItem(loc=["yaml"], msg="empty YAML", type="value_error")
            ],
        )
    if not isinstance(parsed, dict):
        return ValidateResponse(
            valid=False,
            errors=[
                ValidationErrorItem(
                    loc=["yaml"],
                    msg=f"top-level YAML must be a mapping, got {type(parsed).__name__}",
                    type="type_error",
                )
            ],
        )
    try:
        spec = StudySpec.model_validate(parsed)
    except ValidationError as exc:
        return ValidateResponse(
            valid=False,
            errors=[
                ValidationErrorItem(
                    loc=[str(p) for p in err["loc"]], msg=err["msg"], type=err["type"]
                )
                for err in exc.errors()
            ],
        )
    errors = _check_referenced_paths(spec, config_root)
    return ValidateResponse(valid=not errors, errors=errors)


def _check_referenced_paths(
    spec: StudySpec, config_root: Path
) -> list[ValidationErrorItem]:
    """For every leg, verify ``strategy_config``, ``hpo_config``, and each
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


def _row_to_summary(row: sqlite3.Row) -> StudySpecUploadSummary:
    return StudySpecUploadSummary(
        slug=str(row["slug"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        owner_user_id=int(row["user_id"]),
        owner_username=str(row["username"]),
    )


def _row_to_detail(row: sqlite3.Row) -> StudySpecUploadDetail:
    return StudySpecUploadDetail(
        slug=str(row["slug"]),
        yaml=str(row["yaml_text"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        owner_user_id=int(row["user_id"]),
        owner_username=str(row["username"]),
    )


_LIST_SELECT = """
    SELECT
        u.id, u.slug, u.created_at, u.updated_at, u.user_id, u.yaml_text,
        users.username AS username
    FROM study_spec_uploads u
    JOIN users ON users.id = u.user_id
    WHERE u.deleted_at IS NULL
"""


def list_uploads(
    conn: sqlite3.Connection, *, user: UserPublic, all_users: bool = False
) -> list[StudySpecUploadSummary]:
    """List non-deleted uploads.

    Default scope: caller's uploads. ``all_users=True`` requires admin role
    and lists every active upload (used by the "shared library" admin view
    once Sharpening A lands).
    """
    if all_users and user.role is not Role.ADMIN:
        raise PermissionError("only admins can list all users' uploads")
    if all_users:
        rows = conn.execute(_LIST_SELECT + " ORDER BY u.updated_at DESC").fetchall()
    else:
        rows = conn.execute(
            _LIST_SELECT + " AND u.user_id = ? ORDER BY u.updated_at DESC",
            (user.id,),
        ).fetchall()
    return [_row_to_summary(r) for r in rows]


def get_upload(
    conn: sqlite3.Connection, *, user: UserPublic, slug: str
) -> StudySpecUploadDetail:
    """Read one upload by slug. Admins can read any user's; users only their own."""
    if user.role is Role.ADMIN:
        row = conn.execute(
            _LIST_SELECT + " AND u.slug = ? LIMIT 1", (slug,)
        ).fetchone()
    else:
        row = conn.execute(
            _LIST_SELECT + " AND u.user_id = ? AND u.slug = ? LIMIT 1",
            (user.id, slug),
        ).fetchone()
    if row is None:
        raise StudySpecUploadNotFoundError(slug)
    return _row_to_detail(row)


def save_upload(
    conn: sqlite3.Connection,
    *,
    user: UserPublic,
    slug: str,
    yaml_text: str,
    uploads_root: Path,
    config_root: Path,
) -> StudySpecUploadDetail:
    """Validate-and-persist an upload (create or update for the user/slug).

    Side effects: writes ``<uploads_root>/<user_id>/<slug>.yaml`` and commits
    the DB row. On a previously-soft-deleted row with the same (user_id,
    slug), reactivates by clearing ``deleted_at`` — the
    [[soft-delete-schema-pattern]] tombstone-reuse pattern.
    """
    if _library_spec_exists(config_root, slug):
        raise LibrarySlugCollisionError(slug)
    result = validate_study_spec_text(yaml_text, config_root=config_root)
    if not result.valid:
        raise StudySpecUploadInvalidError(result.errors)

    path = _upload_path(uploads_root, user.id, slug)
    ensure_parent_dir(path).write_text(yaml_text, encoding="utf-8")

    now = _now_iso()
    existing = conn.execute(
        "SELECT id, created_at, deleted_at FROM study_spec_uploads "
        "WHERE user_id = ? AND slug = ? LIMIT 1",
        (user.id, slug),
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO study_spec_uploads "
            "(user_id, slug, yaml_text, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user.id, slug, yaml_text, now, now),
        )
        created_at_iso = now
    else:
        # Single UPDATE handles both "edit active row" and "reactivate
        # tombstoned row" — clearing deleted_at is harmless on an active row.
        conn.execute(
            "UPDATE study_spec_uploads "
            "SET yaml_text = ?, updated_at = ?, deleted_at = NULL "
            "WHERE id = ?",
            (yaml_text, now, int(existing["id"])),
        )
        created_at_iso = str(existing["created_at"])
    conn.commit()
    return StudySpecUploadDetail(
        slug=slug,
        yaml=yaml_text,
        created_at=datetime.fromisoformat(created_at_iso),
        updated_at=datetime.fromisoformat(now),
        owner_user_id=user.id,
        owner_username=user.username,
    )


def soft_delete_upload(
    conn: sqlite3.Connection,
    *,
    user: UserPublic,
    slug: str,
    uploads_root: Path,
) -> None:
    """Mark the upload deleted and unlink the on-disk YAML.

    Admins can delete any user's; users only their own. Idempotent on the
    filesystem side — a missing file is not an error.
    """
    if user.role is Role.ADMIN:
        row = conn.execute(
            "SELECT id, user_id FROM study_spec_uploads "
            "WHERE slug = ? AND deleted_at IS NULL LIMIT 1",
            (slug,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, user_id FROM study_spec_uploads "
            "WHERE user_id = ? AND slug = ? AND deleted_at IS NULL LIMIT 1",
            (user.id, slug),
        ).fetchone()
    if row is None:
        raise StudySpecUploadNotFoundError(slug)
    conn.execute(
        "UPDATE study_spec_uploads SET deleted_at = ? WHERE id = ?",
        (_now_iso(), int(row["id"])),
    )
    conn.commit()
    path = _upload_path(uploads_root, int(row["user_id"]), slug)
    path.unlink(missing_ok=True)


def _library_spec_exists(config_root: Path, slug: str) -> bool:
    return (config_root / _STUDY_LIBRARY_SUBDIR / f"{slug}.yaml").is_file()


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
