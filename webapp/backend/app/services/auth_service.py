"""Password hashing + credential verification against the users table."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import bcrypt

from webapp.backend.app.core.types import Role


@dataclass(frozen=True)
class AuthenticatedUser:
    id: int
    username: str
    role: Role


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def authenticate(
    conn: sqlite3.Connection, username: str, password: str
) -> AuthenticatedUser | None:
    row = conn.execute(
        "SELECT id, username, password_hash, role FROM users "
        "WHERE username = ? AND deleted_at IS NULL",
        (username,),
    ).fetchone()
    if row is None:
        return None
    if not verify_password(password, str(row["password_hash"])):
        return None
    return AuthenticatedUser(
        id=int(row["id"]),
        username=str(row["username"]),
        role=Role(str(row["role"])),
    )


def has_any_active_user(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT 1 FROM users WHERE deleted_at IS NULL LIMIT 1").fetchone()
    return row is not None
