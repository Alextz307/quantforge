"""Integration tests for /api/hpo (auth-gated)."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

from fastapi.testclient import TestClient

LIST_PATH = "/api/hpo"
EXPECTED_NAME = "AdaptiveBollinger__spy_daily_5y"
EXPECTED_STORE = "studies/main"
EXPECTED_HPO_COUNT = 1
EXPECTED_N_TRIALS = 3
AFTER_TRIAL_FILTER = 0


def test_list_requires_auth(client: TestClient, webapp_store: Path) -> None:
    assert client.get(LIST_PATH).status_code == HTTPStatus.UNAUTHORIZED


def test_list_returns_hpo_summary(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(LIST_PATH)

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert len(payload) == EXPECTED_HPO_COUNT
    assert payload[0]["name"] == EXPECTED_NAME
    assert payload[0]["store"] == EXPECTED_STORE
    assert payload[0]["n_trials"] == EXPECTED_N_TRIALS


def test_detail_returns_best_config(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(f"{LIST_PATH}/{EXPECTED_NAME}")

    assert response.status_code == HTTPStatus.OK
    detail = response.json()
    assert detail["best_config"]["strategy"]["name"] == "AdaptiveBollinger"


def test_detail_404_for_unknown_name(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(f"{LIST_PATH}/missing")

    assert response.status_code == HTTPStatus.NOT_FOUND


def test_trials_returns_all_by_default(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(f"{LIST_PATH}/{EXPECTED_NAME}/trials")

    assert response.status_code == HTTPStatus.OK
    rows = response.json()
    assert [row["number"] for row in rows] == list(range(EXPECTED_N_TRIALS))


def test_trials_filters_by_after_trial(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(
        f"{LIST_PATH}/{EXPECTED_NAME}/trials", params={"after_trial": AFTER_TRIAL_FILTER}
    )

    assert response.status_code == HTTPStatus.OK
    rows = response.json()
    assert [row["number"] for row in rows] == list(range(AFTER_TRIAL_FILTER + 1, EXPECTED_N_TRIALS))


def test_trials_404_for_unknown_name(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(f"{LIST_PATH}/missing/trials")

    assert response.status_code == HTTPStatus.NOT_FOUND
