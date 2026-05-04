"""Shared pytest fixtures: per-test temp DB, test secret, fresh app + DB."""

from __future__ import annotations

import secrets
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webapp.backend.app.core.rate_limit import login_limiter
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.infrastructure.db import bootstrap_schema, open_db
from webapp.backend.app.main import create_app

TEST_SECRET_KEY = secrets.token_urlsafe(48)


@pytest.fixture(autouse=True)
def _webapp_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("WEBAPP_SECRET_KEY", TEST_SECRET_KEY)
    monkeypatch.setenv("WEBAPP_DB_PATH", str(tmp_path / "webapp.sqlite"))
    get_settings.cache_clear()
    login_limiter.reset()
    yield
    get_settings.cache_clear()
    login_limiter.reset()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def db_conn() -> Iterator[sqlite3.Connection]:
    with open_db() as conn:
        bootstrap_schema(conn)
        yield conn
