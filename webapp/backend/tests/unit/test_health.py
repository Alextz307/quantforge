"""
Verify the liveness endpoint and OpenAPI schema surface.
"""

from __future__ import annotations

from http import HTTPStatus

from fastapi.testclient import TestClient

from webapp.backend.app.core.version import APP_VERSION

HEALTH_PATH = "/api/health"
OPENAPI_PATH = "/openapi.json"


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get(HEALTH_PATH)

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"status": "ok", "version": APP_VERSION}


def test_openapi_schema_lists_health_route(client: TestClient) -> None:
    response = client.get(OPENAPI_PATH)

    assert response.status_code == HTTPStatus.OK
    schema = response.json()
    assert HEALTH_PATH in schema["paths"]
