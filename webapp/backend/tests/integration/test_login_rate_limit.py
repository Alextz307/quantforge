"""
Verify slowapi rate-limit on /api/auth/login: 6th attempt within window 429s.
"""

from __future__ import annotations

import sqlite3
from http import HTTPStatus

from fastapi.testclient import TestClient

from webapp.backend.app.core.rate_limit import LOGIN_RATE_LIMIT
from webapp.backend.app.core.types import Role
from webapp.backend.app.services.user_service import create_user

ALEX = "alex"
PASSWORD = "password123"
LOGIN_PATH = "/api/auth/login"

ALLOWED_PER_WINDOW = 5
TOTAL_ATTEMPTS = ALLOWED_PER_WINDOW + 1


def test_login_rate_limit_caps_attempts_per_window_per_ip(
    client: TestClient, db_conn: sqlite3.Connection
) -> None:
    create_user(db_conn, username=ALEX, password=PASSWORD, role=Role.USER)
    payload = {"username": ALEX, "password": "wrong"}

    statuses = [client.post(LOGIN_PATH, json=payload).status_code for _ in range(TOTAL_ATTEMPTS)]

    assert statuses[:ALLOWED_PER_WINDOW] == [HTTPStatus.UNAUTHORIZED] * ALLOWED_PER_WINDOW
    assert statuses[-1] == HTTPStatus.TOO_MANY_REQUESTS


def test_login_rate_limit_constant_matches_documented_window() -> None:
    assert LOGIN_RATE_LIMIT == "5 per 15 minutes"
