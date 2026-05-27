"""Integration tests for /api/runs endpoints (auth-gated)."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

from fastapi.testclient import TestClient

from webapp.backend.tests.conftest import PLOT_BYTES, PLOT_FILENAME

FLAT_ID = "20260101_120000_AdaptiveBollinger_abc1234_deadbeef"
STUDY_ID = "20260201_090000_PairsTrading_def5678_cafebabe"
EXPECTED_RUN_COUNT = 2
DEFAULT_FOLD_COUNT = 3

RUNS_PATH = "/api/runs"


def test_list_runs_requires_auth(webapp_store: Path, client: TestClient) -> None:
    response = client.get(RUNS_PATH)

    assert response.status_code == HTTPStatus.UNAUTHORIZED


def test_list_runs_returns_both_layouts(webapp_store: Path, authed_client: TestClient) -> None:
    response = authed_client.get(RUNS_PATH)

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    ids = {row["experiment_id"] for row in payload["items"]}
    assert ids == {FLAT_ID, STUDY_ID}
    assert payload["total"] == EXPECTED_RUN_COUNT
    assert len(payload["items"]) == EXPECTED_RUN_COUNT


def test_list_runs_sorted_newest_first(webapp_store: Path, authed_client: TestClient) -> None:
    payload = authed_client.get(RUNS_PATH).json()
    items = payload["items"]

    assert items[0]["experiment_id"] == STUDY_ID
    assert items[1]["experiment_id"] == FLAT_ID


def test_list_runs_pagination_clips_to_limit(webapp_store: Path, authed_client: TestClient) -> None:
    payload = authed_client.get(f"{RUNS_PATH}?limit=1&offset=0").json()
    assert payload["total"] == EXPECTED_RUN_COUNT
    assert len(payload["items"]) == 1
    assert payload["limit"] == 1
    assert payload["offset"] == 0


def test_list_runs_sort_by_sharpe_asc(webapp_store: Path, authed_client: TestClient) -> None:
    payload = authed_client.get(f"{RUNS_PATH}?sort_by=sharpe_mean&order=asc").json()
    sharpes = [row["sharpe_mean"] for row in payload["items"] if row["sharpe_mean"] is not None]
    assert sharpes == sorted(sharpes)


def test_list_runs_filter_by_strategy(webapp_store: Path, authed_client: TestClient) -> None:
    payload = authed_client.get(f"{RUNS_PATH}?strategy=AdaptiveBollinger").json()
    assert payload["total"] == 1
    assert payload["items"][0]["experiment_id"] == FLAT_ID


def test_get_run_detail_returns_manifest_and_metrics(
    webapp_store: Path, authed_client: TestClient
) -> None:
    response = authed_client.get(f"{RUNS_PATH}/{FLAT_ID}")

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["experiment_id"] == FLAT_ID
    assert payload["strategy"] == "AdaptiveBollinger"
    assert payload["tickers"] == ["SPY"]
    assert payload["store"] == "thesis_demo/runs"
    assert "sharpe_mean" in payload["metrics"]
    assert PLOT_FILENAME in payload["plots"]


def test_get_run_detail_404_for_unknown(webapp_store: Path, authed_client: TestClient) -> None:
    response = authed_client.get(f"{RUNS_PATH}/missing_id")

    assert response.status_code == HTTPStatus.NOT_FOUND


def test_get_run_folds_returns_n_folds(webapp_store: Path, authed_client: TestClient) -> None:
    response = authed_client.get(f"{RUNS_PATH}/{FLAT_ID}/folds")

    assert response.status_code == HTTPStatus.OK
    folds = response.json()
    assert len(folds) == DEFAULT_FOLD_COUNT
    assert folds[0]["fold_index"] == 0
    assert folds[0]["equity_curve"]


def test_get_run_plot_returns_bytes(webapp_store: Path, authed_client: TestClient) -> None:
    response = authed_client.get(f"{RUNS_PATH}/{FLAT_ID}/plots/{PLOT_FILENAME}")

    assert response.status_code == HTTPStatus.OK
    assert response.content == PLOT_BYTES


def test_get_run_plot_404_for_missing_plot(webapp_store: Path, authed_client: TestClient) -> None:
    response = authed_client.get(f"{RUNS_PATH}/{FLAT_ID}/plots/does_not_exist.png")

    assert response.status_code == HTTPStatus.NOT_FOUND
