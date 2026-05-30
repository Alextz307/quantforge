"""
User CRUD - never surfaces password hashes outside this module.
"""

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
    auto_created_at = row["auto_created_at"]
    return UserPublic(
        id=int(row["id"]),
        username=str(row["username"]),
        role=Role(str(row["role"])),
        auto_created_at=str(auto_created_at) if auto_created_at is not None else None,
    )


def _insert_user(
    conn: sqlite3.Connection,
    username: str,
    password_hash: str,
    role: Role,
    *,
    created_at: str,
    auto_created_at: str | None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO users (username, password_hash, role, created_at, auto_created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (username, password_hash, role.value, created_at, auto_created_at),
    )
    user_id = cursor.lastrowid
    if user_id is None:
        raise RuntimeError("INSERT returned no lastrowid")
    return int(user_id)


def create_user(
    conn: sqlite3.Connection,
    *,
    username: str,
    password: str,
    role: Role,
    auto_created: bool = False,
) -> UserPublic:
    """
    Insert a new account or reactivate a soft-deleted one with the same username.

    ``auto_created=True`` stamps ``users.auto_created_at`` so an admin
    cleanup view can distinguish CLI ``--user`` auto-creates from deliberate
    ``scripts.create_user`` runs. On reactivation, the flag is honoured the
    same way: a previously-explicit account becomes "auto-created" if a CLI
    user re-creates it, which keeps the cleanup view honest about
    provenance.
    """

    # Reactivate a soft-deleted row with the same username instead of raising:
    # the column-level UNIQUE constraint covers tombstones too, so a fresh INSERT
    # would trip on a previously-deleted user that the admin can no longer see.
    existing = conn.execute(
        "SELECT id, deleted_at FROM users WHERE username = ?", (username,)
    ).fetchone()
    password_hash = hash_password(password)
    now = _now_iso()
    auto_created_at = now if auto_created else None
    if existing is not None:
        if existing["deleted_at"] is None:
            raise UsernameAlreadyExistsError(f"username '{username}' already exists")
        user_id = int(existing["id"])
        conn.execute(
            "UPDATE users SET password_hash = ?, role = ?, deleted_at = NULL, "
            "auto_created_at = ? WHERE id = ?",
            (password_hash, role.value, auto_created_at, user_id),
        )
    else:
        user_id = _insert_user(
            conn,
            username,
            password_hash,
            role,
            created_at=now,
            auto_created_at=auto_created_at,
        )
    conn.commit()
    return UserPublic(id=user_id, username=username, role=role, auto_created_at=auto_created_at)


def upsert_user(
    conn: sqlite3.Connection, *, username: str, password: str, role: Role
) -> UserPublic:
    """
    Create-or-overwrite - used by the bootstrap CLI.
    """

    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    password_hash = hash_password(password)
    if existing is None:
        user_id = _insert_user(
            conn,
            username,
            password_hash,
            role,
            created_at=_now_iso(),
            auto_created_at=None,
        )
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
        "SELECT id, username, role, auto_created_at FROM users WHERE deleted_at IS NULL ORDER BY id"
    ).fetchall()
    return [_row_to_public(row) for row in rows]


def get_user(conn: sqlite3.Connection, user_id: int) -> UserPublic | None:
    row = conn.execute(
        "SELECT id, username, role, auto_created_at FROM users WHERE id = ? AND deleted_at IS NULL",
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
