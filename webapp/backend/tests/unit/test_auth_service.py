"""
Verify password hashing + authenticate behaviour against the users table.
"""

from __future__ import annotations

import sqlite3

from webapp.backend.app.core.types import Role
from webapp.backend.app.services.auth_service import (
    authenticate,
    has_any_active_user,
    hash_password,
    verify_password,
)
from webapp.backend.app.services.user_service import (
    create_user,
    soft_delete_user,
)

ALEX = "alex"
PASSWORD = "password123"
WRONG_PASSWORD = "wrong456"


def test_hash_password_round_trips() -> None:
    hashed = hash_password(PASSWORD)

    assert verify_password(PASSWORD, hashed)
    assert not verify_password(WRONG_PASSWORD, hashed)


def test_hash_password_uses_unique_salt() -> None:
    h1 = hash_password(PASSWORD)
    h2 = hash_password(PASSWORD)

    assert h1 != h2  # bcrypt salts each call
    assert verify_password(PASSWORD, h1)
    assert verify_password(PASSWORD, h2)


def test_authenticate_returns_user_on_valid_credentials(db_conn: sqlite3.Connection) -> None:
    user = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.ADMIN)

    authed = authenticate(db_conn, ALEX, PASSWORD)

    assert authed is not None
    assert authed.id == user.id
    assert authed.role is Role.ADMIN


def test_authenticate_returns_none_on_wrong_password(db_conn: sqlite3.Connection) -> None:
    create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.USER)

    assert authenticate(db_conn, ALEX, WRONG_PASSWORD) is None


def test_authenticate_returns_none_for_unknown_user(db_conn: sqlite3.Connection) -> None:
    assert authenticate(db_conn, ALEX, PASSWORD) is None


def test_authenticate_excludes_soft_deleted(db_conn: sqlite3.Connection) -> None:
    user = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.USER)

    soft_delete_user(db_conn, user.id)

    assert authenticate(db_conn, ALEX, PASSWORD) is None


def test_has_any_active_user_reports_state(db_conn: sqlite3.Connection) -> None:
    assert not has_any_active_user(db_conn)

    user = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.USER)
    assert has_any_active_user(db_conn)

    soft_delete_user(db_conn, user.id)
    assert not has_any_active_user(db_conn)
