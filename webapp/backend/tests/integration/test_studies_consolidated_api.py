"""Integration tests for /api/studies/{name}/consolidated and its file downloads."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

from fastapi.testclient import TestClient

from webapp.backend.app.services.plots import PLOTS_DIRNAME, TABLES_DIRNAME
from webapp.backend.tests.conftest import (
    CONSOLIDATED_PLOT_FILENAME,
    CONSOLIDATED_TABLE_FILENAME,
    PLOT_BYTES,
    make_synthetic_consolidated_report,
)

CONSOLIDATED_PATH_TPL = "/api/studies/{name}/consolidated"
CONSOLIDATED_PLOT_PATH_TPL = "/api/studies/{name}/consolidated/plots/{plot}"
CONSOLIDATED_TABLE_PATH_TPL = "/api/studies/{name}/consolidated/tables/{table}"
EXPECTED_STUDY_NAME = "main"
EXPECTED_PUBLISH_LABEL = "demo_publish"
EXPECTED_STRATEGIES = ("AdaptiveBollinger", "PairsTrading")
EXPECTED_UNIVERSES = ("spy_daily_5y", "spy_daily_10y")
EXPECTED_N_LEGS_HOLDOUT = 2
EXPECTED_N_PAIRWISE = 1


def _populate(webapp_store: Path) -> None:
    make_synthetic_consolidated_report(
        webapp_store / "studies" / EXPECTED_STUDY_NAME,
        study_name=EXPECTED_STUDY_NAME,
        publish_label=EXPECTED_PUBLISH_LABEL,
        strategies=EXPECTED_STRATEGIES,
        universes=EXPECTED_UNIVERSES,
        n_legs_with_holdout=EXPECTED_N_LEGS_HOLDOUT,
        n_universes_with_pairwise=EXPECTED_N_PAIRWISE,
    )


def test_consolidated_requires_auth(client: TestClient, webapp_store: Path) -> None:
    _populate(webapp_store)
    assert (
        client.get(CONSOLIDATED_PATH_TPL.format(name=EXPECTED_STUDY_NAME)).status_code
        == HTTPStatus.UNAUTHORIZED
    )


def test_consolidated_returns_dto(authed_client: TestClient, webapp_store: Path) -> None:
    _populate(webapp_store)

    response = authed_client.get(CONSOLIDATED_PATH_TPL.format(name=EXPECTED_STUDY_NAME))

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["study_name"] == EXPECTED_STUDY_NAME
    assert body["publish_label"] == EXPECTED_PUBLISH_LABEL
    assert body["strategies"] == list(EXPECTED_STRATEGIES)
    assert body["universes"] == list(EXPECTED_UNIVERSES)
    assert body["n_legs_with_holdout"] == EXPECTED_N_LEGS_HOLDOUT
    assert body["n_universes_with_pairwise"] == EXPECTED_N_PAIRWISE
    assert body["tables"] == [CONSOLIDATED_TABLE_FILENAME]
    assert body["plots"] == [CONSOLIDATED_PLOT_FILENAME]


def test_consolidated_404_when_manifest_missing(
    authed_client: TestClient, webapp_store: Path
) -> None:
    response = authed_client.get(CONSOLIDATED_PATH_TPL.format(name=EXPECTED_STUDY_NAME))

    assert response.status_code == HTTPStatus.NOT_FOUND


def test_consolidated_404_when_study_missing(authed_client: TestClient, webapp_store: Path) -> None:
    response = authed_client.get(CONSOLIDATED_PATH_TPL.format(name="does-not-exist"))

    assert response.status_code == HTTPStatus.NOT_FOUND


def test_consolidated_plot_download(authed_client: TestClient, webapp_store: Path) -> None:
    _populate(webapp_store)

    response = authed_client.get(
        CONSOLIDATED_PLOT_PATH_TPL.format(name=EXPECTED_STUDY_NAME, plot=CONSOLIDATED_PLOT_FILENAME)
    )

    assert response.status_code == HTTPStatus.OK
    assert response.content == PLOT_BYTES


def test_consolidated_table_download(authed_client: TestClient, webapp_store: Path) -> None:
    _populate(webapp_store)

    response = authed_client.get(
        CONSOLIDATED_TABLE_PATH_TPL.format(
            name=EXPECTED_STUDY_NAME, table=CONSOLIDATED_TABLE_FILENAME
        )
    )

    assert response.status_code == HTTPStatus.OK
    assert "latex" in response.text


def test_consolidated_plot_blocks_traversal(authed_client: TestClient, webapp_store: Path) -> None:
    _populate(webapp_store)

    response = authed_client.get(
        CONSOLIDATED_PLOT_PATH_TPL.format(name=EXPECTED_STUDY_NAME, plot="../manifest.json")
    )

    assert response.status_code == HTTPStatus.NOT_FOUND


def test_consolidated_subdirs_match_constants(authed_client: TestClient) -> None:
    """Pin the well-known subdir constants the router URLs rely on."""
    assert PLOTS_DIRNAME == "plots"
    assert TABLES_DIRNAME == "tables"
