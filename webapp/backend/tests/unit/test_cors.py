"""Verify CORS middleware is gated on WEBAPP_ENV=development."""

from __future__ import annotations

from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient

from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.main import DEV_FRONTEND_ORIGIN, create_app

HEALTH_PATH = "/api/health"
ALLOW_ORIGIN_HEADER = "access-control-allow-origin"
ALLOW_CREDENTIALS_HEADER = "access-control-allow-credentials"


def _preflight_headers() -> dict[str, str]:
    return {
        "Origin": DEV_FRONTEND_ORIGIN,
        "Access-Control-Request-Method": "GET",
    }


def test_cors_enabled_in_development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBAPP_ENV", "development")
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        response = client.options(HEALTH_PATH, headers=_preflight_headers())

    assert response.status_code == HTTPStatus.OK
    assert response.headers[ALLOW_ORIGIN_HEADER] == DEV_FRONTEND_ORIGIN
    assert response.headers[ALLOW_CREDENTIALS_HEADER] == "true"


def test_cors_disabled_in_local_mode() -> None:
    with TestClient(create_app()) as client:
        response = client.options(HEALTH_PATH, headers=_preflight_headers())

    assert ALLOW_ORIGIN_HEADER not in response.headers
