"""Public-settings endpoint exposes feature flags the SPA needs before login."""

from __future__ import annotations

from http import HTTPStatus

from fastapi.testclient import TestClient

PUBLIC_SETTINGS_PATH = "/api/settings/public"


def test_public_settings_jobs_disabled_by_default(client: TestClient) -> None:
    response = client.get(PUBLIC_SETTINGS_PATH)

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"jobs_enabled": False}


def test_public_settings_jobs_enabled_when_flag_set(jobs_client: TestClient) -> None:
    response = jobs_client.get(PUBLIC_SETTINGS_PATH)

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"jobs_enabled": True}


def test_public_settings_does_not_require_auth(client: TestClient) -> None:
    # No cookie set; endpoint must still respond 200.
    response = client.get(PUBLIC_SETTINGS_PATH)
    assert response.status_code == HTTPStatus.OK
