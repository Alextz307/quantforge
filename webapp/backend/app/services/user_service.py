"""User CRUD — never surfaces password hashes outside this module."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from webapp.backend.app.core.types import Role
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.auth_service import hash_password


class UsernameAlreadyExistsError(ValueError):
    pass


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_public(row: sqlite3.Row) -> UserPublic:
    return UserPublic(id=int(row["id"]), username=str(row["username"]), role=Role(str(row["role"])))


def _insert_user(conn: sqlite3.Connection, username: str, password_hash: str, role: Role) -> int:
    cursor = conn.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
        (username, password_hash, role.value, _now_iso()),
    )
    user_id = cursor.lastrowid
    if user_id is None:
        raise RuntimeError("INSERT returned no lastrowid")
    return int(user_id)


def create_user(
    conn: sqlite3.Connection, *, username: str, password: str, role: Role
) -> UserPublic:
    # Reactivate a soft-deleted row with the same username instead of raising:
    # the column-level UNIQUE constraint covers tombstones too, so a fresh INSERT
    # would trip on a previously-deleted user that the admin can no longer see.
    existing = conn.execute(
        "SELECT id, deleted_at FROM users WHERE username = ?", (username,)
    ).fetchone()
    password_hash = hash_password(password)
    if existing is not None:
        if existing["deleted_at"] is None:
            raise UsernameAlreadyExistsError(f"username '{username}' already exists")
        user_id = int(existing["id"])
        conn.execute(
            "UPDATE users SET password_hash = ?, role = ?, deleted_at = NULL WHERE id = ?",
            (password_hash, role.value, user_id),
        )
    else:
        user_id = _insert_user(conn, username, password_hash, role)
    conn.commit()
    return UserPublic(id=user_id, username=username, role=role)


def upsert_user(
    conn: sqlite3.Connection, *, username: str, password: str, role: Role
) -> UserPublic:
    """Create-or-overwrite — used by the bootstrap CLI."""
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    password_hash = hash_password(password)
    if existing is None:
        user_id = _insert_user(conn, username, password_hash, role)
    else:
        user_id = int(existing["id"])
        conn.execute(
            "UPDATE users SET password_hash = ?, role = ?, deleted_at = NULL WHERE id = ?",
            (password_hash, role.value, user_id),
        )
    conn.commit()
    return UserPublic(id=user_id, username=username, role=role)


def list_users(conn: sqlite3.Connection) -> list[UserPublic]:
    rows = conn.execute(
        "SELECT id, username, role FROM users WHERE deleted_at IS NULL ORDER BY id"
    ).fetchall()
    return [_row_to_public(row) for row in rows]


def get_user(conn: sqlite3.Connection, user_id: int) -> UserPublic | None:
    row = conn.execute(
        "SELECT id, username, role FROM users WHERE id = ? AND deleted_at IS NULL",
        (user_id,),
    ).fetchone()
    return _row_to_public(row) if row else None


def soft_delete_user(conn: sqlite3.Connection, user_id: int) -> bool:
    cursor = conn.execute(
        "UPDATE users SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
        (_now_iso(), user_id),
    )
    conn.commit()
    return cursor.rowcount > 0
