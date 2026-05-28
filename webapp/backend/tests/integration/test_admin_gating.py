"""
Verify /api/users gating: admin-only access + CRUD round-trip.
"""

from __future__ import annotations

import sqlite3
from http import HTTPStatus

from fastapi.testclient import TestClient

from webapp.backend.app.core.types import Role
from webapp.backend.app.services.user_service import create_user

ADMIN_NAME = "admin"
USER_NAME = "guest"
PASSWORD = "password123"
LOGIN_PATH = "/api/auth/login"
USERS_PATH = "/api/users"


def _seed(db_conn: sqlite3.Connection, *, name: str, role: Role) -> int:
    return create_user(db_conn, username=name, password=PASSWORD, role=role).id


def _login(client: TestClient, username: str) -> None:
    response = client.post(LOGIN_PATH, json={"username": username, "password": PASSWORD})
    assert response.status_code == HTTPStatus.OK, response.text


def test_unauthenticated_request_returns_401(client: TestClient) -> None:
    response = client.get(USERS_PATH)

    assert response.status_code == HTTPStatus.UNAUTHORIZED


def test_non_admin_request_returns_403(client: TestClient, db_conn: sqlite3.Connection) -> None:
    _seed(db_conn, name=USER_NAME, role=Role.USER)
    _login(client, USER_NAME)

    response = client.get(USERS_PATH)

    assert response.status_code == HTTPStatus.FORBIDDEN


def test_admin_can_list_users(client: TestClient, db_conn: sqlite3.Connection) -> None:
    _seed(db_conn, name=ADMIN_NAME, role=Role.ADMIN)
    _login(client, ADMIN_NAME)

    response = client.get(USERS_PATH)

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert any(u["username"] == ADMIN_NAME for u in payload)


def test_admin_can_create_user(client: TestClient, db_conn: sqlite3.Connection) -> None:
    _seed(db_conn, name=ADMIN_NAME, role=Role.ADMIN)
    _login(client, ADMIN_NAME)

    response = client.post(
        USERS_PATH,
        json={"username": "newbie", "password": "newpass123", "role": "user"},
    )

    assert response.status_code == HTTPStatus.CREATED
    payload = response.json()
    assert payload["username"] == "newbie"
    assert payload["role"] == Role.USER.value
    assert "password" not in payload
    assert "password_hash" not in payload


def test_admin_create_with_duplicate_username_returns_409(
    client: TestClient, db_conn: sqlite3.Connection
) -> None:
    _seed(db_conn, name=ADMIN_NAME, role=Role.ADMIN)
    _login(client, ADMIN_NAME)

    payload = {"username": "duplicate", "password": "newpass123", "role": "user"}
    client.post(USERS_PATH, json=payload)

    response = client.post(USERS_PATH, json=payload)

    assert response.status_code == HTTPStatus.CONFLICT


def test_admin_can_soft_delete_user(client: TestClient, db_conn: sqlite3.Connection) -> None:
    _seed(db_conn, name=ADMIN_NAME, role=Role.ADMIN)
    target_id = _seed(db_conn, name=USER_NAME, role=Role.USER)
    _login(client, ADMIN_NAME)

    response = client.delete(f"{USERS_PATH}/{target_id}")

    assert response.status_code == HTTPStatus.NO_CONTENT
    listed = client.get(USERS_PATH).json()
    assert all(u["id"] != target_id for u in listed)


def test_admin_delete_unknown_user_returns_404(
    client: TestClient, db_conn: sqlite3.Connection
) -> None:
    _seed(db_conn, name=ADMIN_NAME, role=Role.ADMIN)
    _login(client, ADMIN_NAME)

    response = client.delete(f"{USERS_PATH}/9999")

    assert response.status_code == HTTPStatus.NOT_FOUND
