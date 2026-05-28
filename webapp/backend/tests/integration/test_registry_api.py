"""
Integration tests for /api/strategies and /api/models (auth-gated).
"""

from __future__ import annotations

from http import HTTPStatus

from fastapi.testclient import TestClient

STRATEGIES_PATH = "/api/strategies"
MODELS_PATH = "/api/models"


def test_strategies_requires_auth(client: TestClient) -> None:
    assert client.get(STRATEGIES_PATH).status_code == HTTPStatus.UNAUTHORIZED


def test_models_requires_auth(client: TestClient) -> None:
    assert client.get(MODELS_PATH).status_code == HTTPStatus.UNAUTHORIZED


def test_strategies_returns_known_entries(authed_client: TestClient) -> None:
    response = authed_client.get(STRATEGIES_PATH)

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    names = {row["name"] for row in payload}
    assert "AdaptiveBollinger" in names
    assert "PairsTrading" in names
    for row in payload:
        assert row["qualname"].startswith("src.strategies.")


def test_models_includes_predictors_and_classifiers(authed_client: TestClient) -> None:
    response = authed_client.get(MODELS_PATH)

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    kinds = {row["kind"] for row in payload}
    names = {row["name"] for row in payload}
    assert kinds == {"predictor", "classifier"}
    assert "garch" in names
    assert "xgboost_directional" in names
