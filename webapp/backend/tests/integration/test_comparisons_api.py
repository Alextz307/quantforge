"""
Integration tests for /api/comparisons (auth-gated).
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

from fastapi.testclient import TestClient

from src.core.persistence import COMPARISONS_SUBDIR
from webapp.backend.tests.conftest import make_synthetic_comparison

LIST_PATH = "/api/comparisons"
EXPECTED_NAME = "flat_compare"
EXPECTED_STRATEGIES = {"AdaptiveBollinger", "PairsTrading"}
EXPECTED_RUN_COUNT = 1
PLOT_NAME = "equity.png"
OFFSET_BEYOND_RANGE = 100
LIMIT_ABOVE_CAP = 501
PAGE_LIMIT_ONE = 1
SEEDED_TOTAL = 2
FUTURE_SINCE = "2099-01-01T00:00:00Z"


def test_list_requires_auth(client: TestClient, webapp_store: Path) -> None:
    assert client.get(LIST_PATH).status_code == HTTPStatus.UNAUTHORIZED


def test_list_returns_comparison_summary(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(LIST_PATH)

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    items = payload["items"]
    assert len(items) == EXPECTED_RUN_COUNT
    assert payload["total"] == EXPECTED_RUN_COUNT
    assert set(payload["strategies"]) == EXPECTED_STRATEGIES
    assert items[0]["name"] == EXPECTED_NAME
    assert set(items[0]["strategies"]) == EXPECTED_STRATEGIES


def test_list_offset_beyond_range_returns_empty(
    authed_client: TestClient, webapp_store: Path
) -> None:
    response = authed_client.get(LIST_PATH, params={"offset": OFFSET_BEYOND_RANGE})

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["items"] == []
    assert payload["total"] == EXPECTED_RUN_COUNT


def test_list_rejects_limit_above_cap(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(LIST_PATH, params={"limit": LIMIT_ABOVE_CAP})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_list_truncates_page_to_limit(authed_client: TestClient, webapp_store: Path) -> None:
    # Seed a second comparison so the total exceeds a limit of one; the page must
    # clip to `limit` while `total` still reports the full count. A limit/offset
    # arg swapped into paginate() surfaces here - the single-row fixture can't.
    make_synthetic_comparison(
        webapp_store / "flat_store" / COMPARISONS_SUBDIR, name="second_compare"
    )

    response = authed_client.get(LIST_PATH, params={"limit": PAGE_LIMIT_ONE})

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert len(payload["items"]) == PAGE_LIMIT_ONE
    assert payload["total"] == SEEDED_TOTAL


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
    assert {row["strategy"] for row in detail["per_strategy_stats"]} == EXPECTED_STRATEGIES


def test_detail_404_for_unknown_name(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(f"{LIST_PATH}/missing")

    assert response.status_code == HTTPStatus.NOT_FOUND


def test_plot_returns_file(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(f"{LIST_PATH}/{EXPECTED_NAME}/plots/{PLOT_NAME}")

    assert response.status_code == HTTPStatus.OK
    assert response.content.startswith(b"\x89PNG")


def test_plot_404_for_traversal(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(f"{LIST_PATH}/{EXPECTED_NAME}/plots/missing.png")

    assert response.status_code == HTTPStatus.NOT_FOUND
