"""
End-to-end login -> /me -> logout flow with cookie verification.
"""

from __future__ import annotations

import sqlite3
from http import HTTPStatus

from fastapi.testclient import TestClient

from webapp.backend.app.core.security import SESSION_COOKIE_NAME
from webapp.backend.app.core.types import Role
from webapp.backend.app.services.user_service import create_user

ALEX = "alex"
PASSWORD = "password123"
LOGIN_PATH = "/api/auth/login"
LOGOUT_PATH = "/api/auth/logout"
ME_PATH = "/api/auth/me"


def _seed(db_conn: sqlite3.Connection, *, role: Role = Role.USER) -> int:
    return create_user(db_conn, username=ALEX, password=PASSWORD, role=role).id


def test_login_with_valid_credentials_returns_user_and_sets_cookie(
    client: TestClient, db_conn: sqlite3.Connection
) -> None:
    _seed(db_conn)

    response = client.post(LOGIN_PATH, json={"username": ALEX, "password": PASSWORD})

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["username"] == ALEX
    assert payload["role"] == Role.USER.value
    assert SESSION_COOKIE_NAME in response.cookies


def test_login_with_invalid_credentials_returns_401(
    client: TestClient, db_conn: sqlite3.Connection
) -> None:
    _seed(db_conn)

    response = client.post(LOGIN_PATH, json={"username": ALEX, "password": "wrong"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert SESSION_COOKIE_NAME not in response.cookies


def test_login_for_unknown_user_returns_401(client: TestClient) -> None:
    response = client.post(LOGIN_PATH, json={"username": "ghost", "password": PASSWORD})

    assert response.status_code == HTTPStatus.UNAUTHORIZED


def test_me_without_cookie_returns_null(client: TestClient) -> None:
    response = client.get(ME_PATH)

    assert response.status_code == HTTPStatus.OK
    assert response.json() is None


def test_me_after_login_returns_current_user(
    client: TestClient, db_conn: sqlite3.Connection
) -> None:
    _seed(db_conn, role=Role.ADMIN)
    client.post(LOGIN_PATH, json={"username": ALEX, "password": PASSWORD})

    response = client.get(ME_PATH)

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["username"] == ALEX
    assert payload["role"] == Role.ADMIN.value


def test_me_with_tampered_cookie_returns_null(client: TestClient) -> None:
    client.cookies.set(SESSION_COOKIE_NAME, "tampered.value.here")

    response = client.get(ME_PATH)

    assert response.status_code == HTTPStatus.OK
    assert response.json() is None


def test_logout_clears_cookie(client: TestClient, db_conn: sqlite3.Connection) -> None:
    _seed(db_conn)
    client.post(LOGIN_PATH, json={"username": ALEX, "password": PASSWORD})

    response = client.post(LOGOUT_PATH)

    assert response.status_code == HTTPStatus.OK
    me_response = client.get(ME_PATH)
    assert me_response.status_code == HTTPStatus.OK
    assert me_response.json() is None
