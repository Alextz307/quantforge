"""Generic CRUD over ``*_spec_uploads`` tables.

A ``SpecUploadStore`` instance binds a single upload kind (e.g. study,
universe) to its table name, library subdir, DTO classes, exception
classes, and YAML validator. Both ``study_spec_uploads`` and
``universe_spec_uploads`` reduce to a thin spec-specific validator plus
a store binding — the duplicated row-mapping, list/get/save/soft-delete
SQL, on-disk YAML write, and tombstone-reuse logic all live here.

The CRUD shape:

* **List** — non-deleted rows, scoped to the caller (admins can pass
  ``all_users=True`` to see every active row).
* **Get** — one row by slug; non-admins can only read their own.
* **Save** — validate, write disk, then commit DB. On an existing row
  (active or tombstoned) the same UPDATE clears ``deleted_at`` and bumps
  ``updated_at``; the partial unique index treats tombstones as absent
  so a single UPDATE handles both edit-active and reactivate-tombstoned.
* **Soft delete** — set ``deleted_at`` and ``unlink(missing_ok=True)``.

A user-uploaded slug must not shadow a library spec under
``<config_root>/<library_subdir>/<slug>.yaml`` — the collision check
fires before validation so the resolver can rely on uploads never
shadowing library entries. ``LibrarySlugCollisionError`` carries the
resolved library path so routers don't hardcode subdir strings.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, Protocol, TypeAlias, TypeVar, cast

import yaml
from pydantic import BaseModel, ValidationError

from src.core.fs import ensure_parent_dir
from webapp.backend.app.core.types import Role
from webapp.backend.app.schemas.configs import ValidateResponse, ValidationErrorItem
from webapp.backend.app.schemas.users import UserPublic


class LibrarySlugCollisionError(ValueError):
    """Raised when a save would shadow an existing library spec.

    Carries ``library_path`` so the router can format the 409 detail
    message without re-deriving the subdir.
    """

    def __init__(self, slug: str, library_path: Path) -> None:
        super().__init__(slug)
        self.slug = slug
        self.library_path = library_path


class SpecUploadNotFoundError(LookupError):
    """Raised when a slug is absent (or soft-deleted) for the caller.

    Per-kind subclasses set ``kind_label`` so the app-level 404 handler
    can format ``{label} spec upload not found: {slug}`` without
    discriminating on the concrete class.
    """

    kind_label: str = "spec"


class SpecUploadInvalidError(ValueError):
    """Raised when an upload payload fails YAML parse / schema / path checks.

    Carries the same ``ValidationErrorItem`` shape the jobs API uses so
    routers can surface inline errors verbatim under the editor.

    Per-kind subclasses fix the prose ``kind`` in their own ``__init__`` so
    the store can construct them with just the errors list (see
    :class:`InvalidErrorFactory`).
    """

    def __init__(self, kind: str, errors: list[ValidationErrorItem]) -> None:
        super().__init__(f"invalid {kind} spec upload ({len(errors)} error(s))")
        self.errors = errors


class InvalidErrorFactory(Protocol):
    """Callable shape the store uses to construct kind-specific invalid errors.

    Per-kind subclasses of :class:`SpecUploadInvalidError` accept just
    the errors list (and inject the kind string themselves); this Protocol
    captures that narrower call signature so the store's call site
    typechecks against the subclass shape rather than the base.
    """

    def __call__(self, errors: list[ValidationErrorItem]) -> SpecUploadInvalidError: ...


SummaryT = TypeVar("SummaryT", bound=BaseModel)
DetailT = TypeVar("DetailT", bound=BaseModel)

ValidatorFn: TypeAlias = Callable[[str, Path], ValidateResponse]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _upload_path(uploads_root: Path, user_id: int, slug: str) -> Path:
    return uploads_root / str(user_id) / f"{slug}.yaml"


def find_upload_path(uploads_root: Path, user_id: int, slug: str) -> Path | None:
    """Return the on-disk YAML path for an upload if the file exists, else None.

    Pure filesystem lookup — does not touch the DB. Used by job handlers
    on the spawn path where the row has already been verified by a prior
    validate() call.
    """
    path = _upload_path(uploads_root, user_id, slug)
    return path if path.is_file() else None


def parse_yaml_mapping(
    yaml_text: str,
) -> tuple[dict[str, object] | None, list[ValidationErrorItem]]:
    """Parse YAML text and confirm the top level is a mapping.

    Returns ``(parsed, [])`` on success or ``(None, [errors])`` on parse
    error, empty body, or non-mapping top level. Caller chains its own
    schema validation onto the returned mapping.
    """
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return None, [
            ValidationErrorItem(
                loc=["yaml"], msg=f"YAML parse error: {exc}", type="value_error"
            )
        ]
    if parsed is None:
        return None, [
            ValidationErrorItem(loc=["yaml"], msg="empty YAML", type="value_error")
        ]
    if not isinstance(parsed, dict):
        return None, [
            ValidationErrorItem(
                loc=["yaml"],
                msg=f"top-level YAML must be a mapping, got {type(parsed).__name__}",
                type="type_error",
            )
        ]
    return cast(dict[str, object], parsed), []


def validate_against_pydantic(
    parsed: dict[str, object], model_cls: type[BaseModel]
) -> list[ValidationErrorItem]:
    """Run ``model_cls.model_validate(parsed)`` and surface Pydantic errors.

    Returns ``[]`` on success or a list of editor-friendly
    ``ValidationErrorItem`` rows on failure.
    """
    try:
        model_cls.model_validate(parsed)
    except ValidationError as exc:
        return [
            ValidationErrorItem(
                loc=[str(p) for p in err["loc"]], msg=err["msg"], type=err["type"]
            )
            for err in exc.errors()
        ]
    return []


@dataclass(frozen=True)
class SpecUploadStore(Generic[SummaryT, DetailT]):
    """Per-kind CRUD over a ``*_spec_uploads`` table.

    Bound once per upload kind (study, universe, ...) at the module level
    of the kind-specific service module. The store owns the SQL + on-disk
    flows; the kind module owns the spec-specific validator and exposes
    the public free-function API.
    """

    table_name: str
    library_subdir: str
    summary_cls: type[SummaryT]
    detail_cls: type[DetailT]
    not_found_error: type[SpecUploadNotFoundError]
    invalid_error: InvalidErrorFactory
    validator: ValidatorFn

    def library_path(self, config_root: Path, slug: str) -> Path:
        return config_root / self.library_subdir / f"{slug}.yaml"

    def _summary_select(self) -> str:
        return (
            f"SELECT u.id, u.slug, u.created_at, u.updated_at, u.user_id, "
            f"users.username AS username "
            f"FROM {self.table_name} u "  # noqa: S608 - table name is a frozen-dataclass constant
            f"JOIN users ON users.id = u.user_id "
            f"WHERE u.deleted_at IS NULL"
        )

    def _detail_select(self) -> str:
        return (
            f"SELECT u.id, u.slug, u.created_at, u.updated_at, u.user_id, u.yaml_text, "
            f"users.username AS username "
            f"FROM {self.table_name} u "  # noqa: S608 - table name is a frozen-dataclass constant
            f"JOIN users ON users.id = u.user_id "
            f"WHERE u.deleted_at IS NULL"
        )

    def _row_to_summary(self, row: sqlite3.Row) -> SummaryT:
        return self.summary_cls(
            slug=str(row["slug"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            owner_user_id=int(row["user_id"]),
            owner_username=str(row["username"]),
        )

    def _row_to_detail(self, row: sqlite3.Row) -> DetailT:
        return self.detail_cls(
            slug=str(row["slug"]),
            yaml=str(row["yaml_text"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            owner_user_id=int(row["user_id"]),
            owner_username=str(row["username"]),
        )

    def list_uploads(
        self,
        conn: sqlite3.Connection,
        *,
        user: UserPublic,
        all_users: bool = False,
    ) -> list[SummaryT]:
        if all_users and user.role is not Role.ADMIN:
            raise PermissionError("only admins can list all users' uploads")
        if all_users:
            rows = conn.execute(
                self._summary_select() + " ORDER BY u.updated_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                self._summary_select()
                + " AND u.user_id = ? ORDER BY u.updated_at DESC",
                (user.id,),
            ).fetchall()
        return [self._row_to_summary(r) for r in rows]

    def get_upload(
        self, conn: sqlite3.Connection, *, user: UserPublic, slug: str
    ) -> DetailT:
        if user.role is Role.ADMIN:
            row = conn.execute(
                self._detail_select() + " AND u.slug = ? LIMIT 1", (slug,)
            ).fetchone()
        else:
            row = conn.execute(
                self._detail_select() + " AND u.user_id = ? AND u.slug = ? LIMIT 1",
                (user.id, slug),
            ).fetchone()
        if row is None:
            raise self.not_found_error(slug)
        return self._row_to_detail(row)

    def save_upload(
        self,
        conn: sqlite3.Connection,
        *,
        user: UserPublic,
        slug: str,
        yaml_text: str,
        uploads_root: Path,
        config_root: Path,
    ) -> DetailT:
        library = self.library_path(config_root, slug)
        if library.is_file():
            raise LibrarySlugCollisionError(slug, library)
        result = self.validator(yaml_text, config_root)
        if not result.valid:
            raise self.invalid_error(result.errors)

        path = _upload_path(uploads_root, user.id, slug)
        ensure_parent_dir(path).write_text(yaml_text, encoding="utf-8")

        now = _now_iso()
        existing = conn.execute(
            f"SELECT id, created_at, deleted_at FROM {self.table_name} "  # noqa: S608
            f"WHERE user_id = ? AND slug = ? LIMIT 1",
            (user.id, slug),
        ).fetchone()
        if existing is None:
            conn.execute(
                f"INSERT INTO {self.table_name} "  # noqa: S608
                f"(user_id, slug, yaml_text, created_at, updated_at) "
                f"VALUES (?, ?, ?, ?, ?)",
                (user.id, slug, yaml_text, now, now),
            )
            created_at_iso = now
        else:
            conn.execute(
                f"UPDATE {self.table_name} "  # noqa: S608
                f"SET yaml_text = ?, updated_at = ?, deleted_at = NULL "
                f"WHERE id = ?",
                (yaml_text, now, int(existing["id"])),
            )
            created_at_iso = str(existing["created_at"])
        conn.commit()
        return self.detail_cls(
            slug=slug,
            yaml=yaml_text,
            created_at=datetime.fromisoformat(created_at_iso),
            updated_at=datetime.fromisoformat(now),
            owner_user_id=user.id,
            owner_username=user.username,
        )

    def soft_delete_upload(
        self,
        conn: sqlite3.Connection,
        *,
        user: UserPublic,
        slug: str,
        uploads_root: Path,
    ) -> None:
        if user.role is Role.ADMIN:
            row = conn.execute(
                f"SELECT id, user_id FROM {self.table_name} "  # noqa: S608
                f"WHERE slug = ? AND deleted_at IS NULL LIMIT 1",
                (slug,),
            ).fetchone()
        else:
            row = conn.execute(
                f"SELECT id, user_id FROM {self.table_name} "  # noqa: S608
                f"WHERE user_id = ? AND slug = ? AND deleted_at IS NULL LIMIT 1",
                (user.id, slug),
            ).fetchone()
        if row is None:
            raise self.not_found_error(slug)
        conn.execute(
            f"UPDATE {self.table_name} SET deleted_at = ? WHERE id = ?",  # noqa: S608
            (_now_iso(), int(row["id"])),
        )
        conn.commit()
        path = _upload_path(uploads_root, int(row["user_id"]), slug)
        path.unlink(missing_ok=True)


__all__ = [
    "InvalidErrorFactory",
    "LibrarySlugCollisionError",
    "SpecUploadInvalidError",
    "SpecUploadNotFoundError",
    "SpecUploadStore",
    "ValidatorFn",
    "find_upload_path",
    "parse_yaml_mapping",
    "validate_against_pydantic",
]
