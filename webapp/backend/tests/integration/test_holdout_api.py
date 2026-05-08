"""Integration tests for /api/holdout-evals (auth-gated)."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

from fastapi.testclient import TestClient

LIST_PATH = "/api/holdout-evals"
EXPECTED_NAME = "study_holdout"
EXPECTED_SOURCE_KIND = "run"
EXPECTED_STORE = "studies/main/holdout_evals"
EXPECTED_SHARPE = 0.6
EXPECTED_SLIPPAGE = "normal"
EXPECTED_EQUITY_CURVE = [10000.0, 10100.0, 10500.0]
EXPECTED_RUN_COUNT = 1
PLOT_NAME = "equity.png"


def test_list_requires_auth(client: TestClient, webapp_store: Path) -> None:
    assert client.get(LIST_PATH).status_code == HTTPStatus.UNAUTHORIZED


def test_list_returns_holdout_summary(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(LIST_PATH)

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert len(payload) == EXPECTED_RUN_COUNT
    assert payload[0]["name"] == EXPECTED_NAME
    assert payload[0]["source_kind"] == EXPECTED_SOURCE_KIND
    assert payload[0]["store"] == EXPECTED_STORE


def test_detail_returns_full_payload(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(f"{LIST_PATH}/{EXPECTED_NAME}")

    assert response.status_code == HTTPStatus.OK
    detail = response.json()
    assert detail["sharpe_ratio"] == EXPECTED_SHARPE
    assert detail["slippage_scenario"] == EXPECTED_SLIPPAGE
    assert detail["equity_curve"] == EXPECTED_EQUITY_CURVE


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
