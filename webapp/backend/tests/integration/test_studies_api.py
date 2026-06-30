"""
Integration tests for /api/studies (auth-gated).
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

from fastapi.testclient import TestClient

LIST_PATH = "/api/studies"
EXPECTED_NAME = "main"
EXPECTED_SPEC_NAME = "demo_spec"
EXPECTED_TOTAL_LEGS = 2
EXPECTED_COMPLETED_LEGS = 1
EXPECTED_STUDY_COUNT = 1
OFFSET_BEYOND_RANGE = 100
LIMIT_ABOVE_CAP = 501
FUTURE_SINCE = "2099-01-01T00:00:00Z"


def test_list_requires_auth(client: TestClient, webapp_store: Path) -> None:
    assert client.get(LIST_PATH).status_code == HTTPStatus.UNAUTHORIZED


def test_list_returns_study_summary(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(LIST_PATH)

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    items = payload["items"]
    assert len(items) == EXPECTED_STUDY_COUNT
    assert payload["total"] == EXPECTED_STUDY_COUNT
    assert payload["specs"] == [EXPECTED_SPEC_NAME]
    assert items[0]["name"] == EXPECTED_NAME
    assert items[0]["spec_name"] == EXPECTED_SPEC_NAME
    assert items[0]["total_legs"] == EXPECTED_TOTAL_LEGS
    assert items[0]["completed_legs"] == EXPECTED_COMPLETED_LEGS


def test_list_offset_beyond_range_returns_empty(
    authed_client: TestClient, webapp_store: Path
) -> None:
    response = authed_client.get(LIST_PATH, params={"offset": OFFSET_BEYOND_RANGE})

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["items"] == []
    assert payload["total"] == EXPECTED_STUDY_COUNT


def test_list_rejects_limit_above_cap(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(LIST_PATH, params={"limit": LIMIT_ABOVE_CAP})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_list_future_since_excludes_all(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(LIST_PATH, params={"since": FUTURE_SINCE})

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["items"] == []
    assert payload["total"] == 0


def test_detail_returns_full_payload(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(f"{LIST_PATH}/{EXPECTED_NAME}")

    assert response.status_code == HTTPStatus.OK
    detail = response.json()
    assert detail["spec_name"] == EXPECTED_SPEC_NAME
    assert len(detail["legs"]) == EXPECTED_TOTAL_LEGS
    assert any(leg["is_complete"] for leg in detail["legs"])


def test_detail_404_for_unknown_name(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(f"{LIST_PATH}/missing")

    assert response.status_code == HTTPStatus.NOT_FOUND
