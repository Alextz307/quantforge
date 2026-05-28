"""
Verify user_service CRUD: create, list, get, soft-delete, upsert.
"""

from __future__ import annotations

import sqlite3

import pytest

from webapp.backend.app.core.types import Role
from webapp.backend.app.services.auth_service import verify_password
from webapp.backend.app.services.user_service import (
    UsernameAlreadyExistsError,
    create_user,
    get_user,
    list_users,
    soft_delete_user,
    upsert_user,
)

ALEX = "alex"
GUEST = "guest"
PASSWORD = "password123"
NEW_PASSWORD = "different456"


def _password_hash_for(conn: sqlite3.Connection, username: str) -> str:
    row = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
    assert row is not None
    return str(row["password_hash"])


def test_create_user_inserts_active_row(db_conn: sqlite3.Connection) -> None:
    user = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.ADMIN)

    assert user.username == ALEX
    assert user.role is Role.ADMIN
    assert user.id > 0
    assert verify_password(PASSWORD, _password_hash_for(db_conn, ALEX))


def test_create_user_with_duplicate_username_raises(db_conn: sqlite3.Connection) -> None:
    create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.USER)

    with pytest.raises(UsernameAlreadyExistsError):
        create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.USER)


def test_create_user_reactivates_soft_deleted_username(db_conn: sqlite3.Connection) -> None:
    original = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.USER)
    soft_delete_user(db_conn, original.id)

    revived = create_user(db_conn, username=ALEX, password=NEW_PASSWORD, role=Role.ADMIN)

    assert revived.id == original.id
    assert revived.role is Role.ADMIN
    assert verify_password(NEW_PASSWORD, _password_hash_for(db_conn, ALEX))
    fetched = get_user(db_conn, revived.id)
    assert fetched is not None and fetched.role is Role.ADMIN


def test_reactivate_preserves_artifact_ownership(db_conn: sqlite3.Connection) -> None:
    """
    Reactivating a soft-deleted username keeps the FK link from jobs alive.

    Load-bearing invariant the reactivate-on-collision pattern guards: any
    artifact attributed to ``user_id=X`` before the soft-delete must still
    resolve to the (now-revived) ``user_id=X`` after a same-username
    ``create_user``. If a future refactor switches to plain INSERT, the
    revived user gets a fresh id, the jobs row dangles, and every artifact
    that user ever launched silently becomes ownerless. This test fails the
    moment that happens.
    """

    original = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.USER)
    db_conn.execute(
        "INSERT INTO jobs (id, user_id, kind, command, config_path, log_path, "
        "status, experiment_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("job-1", original.id, "run", "exp run", "/cfg", "/log", "succeeded", "exp-abc"),
    )
    db_conn.commit()

    soft_delete_user(db_conn, original.id)
    revived = create_user(db_conn, username=ALEX, password=NEW_PASSWORD, role=Role.USER)

    assert revived.id == original.id
    owner_row = db_conn.execute(
        "SELECT user_id FROM jobs WHERE experiment_id = ?", ("exp-abc",)
    ).fetchone()
    assert owner_row is not None
    assert int(owner_row["user_id"]) == revived.id


def test_list_users_excludes_soft_deleted(db_conn: sqlite3.Connection) -> None:
    alex = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.ADMIN)
    guest = create_user(db_conn, username=GUEST, password=PASSWORD, role=Role.USER)

    soft_delete_user(db_conn, alex.id)

    listed = list_users(db_conn)
    assert [u.id for u in listed] == [guest.id]


def test_list_users_orders_by_id(db_conn: sqlite3.Connection) -> None:
    a = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.ADMIN)
    g = create_user(db_conn, username=GUEST, password=PASSWORD, role=Role.USER)

    listed = list_users(db_conn)

    assert [u.id for u in listed] == [a.id, g.id]


def test_get_user_returns_user(db_conn: sqlite3.Connection) -> None:
    user = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.ADMIN)

    fetched = get_user(db_conn, user.id)

    assert fetched is not None
    assert fetched.username == ALEX
    assert fetched.role is Role.ADMIN


def test_get_user_returns_none_for_missing(db_conn: sqlite3.Connection) -> None:
    assert get_user(db_conn, 9999) is None


def test_get_user_returns_none_for_soft_deleted(db_conn: sqlite3.Connection) -> None:
    user = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.ADMIN)

    soft_delete_user(db_conn, user.id)

    assert get_user(db_conn, user.id) is None


def test_soft_delete_returns_false_for_unknown(db_conn: sqlite3.Connection) -> None:
    assert soft_delete_user(db_conn, 9999) is False


def test_soft_delete_is_idempotent(db_conn: sqlite3.Connection) -> None:
    user = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.USER)

    assert soft_delete_user(db_conn, user.id) is True
    assert soft_delete_user(db_conn, user.id) is False


def test_upsert_creates_new_user(db_conn: sqlite3.Connection) -> None:
    user = upsert_user(db_conn, username=ALEX, password=PASSWORD, role=Role.ADMIN)

    fetched = get_user(db_conn, user.id)
    assert fetched is not None
    assert fetched.role is Role.ADMIN


def test_upsert_overwrites_existing_password_and_role(db_conn: sqlite3.Connection) -> None:
    original = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.USER)

    updated = upsert_user(db_conn, username=ALEX, password=NEW_PASSWORD, role=Role.ADMIN)

    assert updated.id == original.id
    assert updated.role is Role.ADMIN
    assert verify_password(NEW_PASSWORD, _password_hash_for(db_conn, ALEX))
    assert not verify_password(PASSWORD, _password_hash_for(db_conn, ALEX))


def test_upsert_revives_soft_deleted_user(db_conn: sqlite3.Connection) -> None:
    user = create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.USER)
    soft_delete_user(db_conn, user.id)
    assert get_user(db_conn, user.id) is None

    upsert_user(db_conn, username=ALEX, password=PASSWORD, role=Role.USER)

    assert get_user(db_conn, user.id) is not None
