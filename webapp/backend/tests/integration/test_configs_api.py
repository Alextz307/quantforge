"""Integration tests for /api/configs/* and /api/strategies/{name}/schema."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from webapp.backend.app.core.settings import get_settings
from webapp.backend.tests.conftest import make_valid_experiment_payload

VALIDATE_PATH = "/api/configs/validate"
CONFIGS_PATH = "/api/configs"
STRATEGY_SCHEMA_PATH = "/api/strategies/{name}/schema"


@pytest.fixture
def webapp_config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "config"
    (root / "universes").mkdir(parents=True)
    (root / "strategies").mkdir()
    (root / "universes" / "spy.yaml").write_text(
        yaml.safe_dump(
            {
                "data": {
                    "source": "yfinance",
                    "tickers": ["SPY"],
                    "start": "2020-01-01",
                    "end": "2024-12-31",
                    "interval": "daily",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WEBAPP_CONFIG_ROOT", str(root))
    get_settings.cache_clear()
    return root


def test_configs_requires_auth(client: TestClient) -> None:
    assert client.post(VALIDATE_PATH, json={}).status_code == HTTPStatus.UNAUTHORIZED
    assert client.get(f"{CONFIGS_PATH}/universe").status_code == HTTPStatus.UNAUTHORIZED
    assert client.get(f"{CONFIGS_PATH}/universe/spy").status_code == HTTPStatus.UNAUTHORIZED


def test_validate_happy_path(authed_client: TestClient) -> None:
    response = authed_client.post(
        VALIDATE_PATH,
        json={"kind": "experiment", "payload": make_valid_experiment_payload()},
    )

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body == {"valid": True, "errors": []}


def test_validate_returns_structured_errors(authed_client: TestClient) -> None:
    bad = make_valid_experiment_payload()
    del bad["data"]

    response = authed_client.post(VALIDATE_PATH, json={"kind": "experiment", "payload": bad})

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["valid"] is False
    assert any(err["loc"] == ["data"] for err in body["errors"])


def test_validate_rejects_unknown_kind(authed_client: TestClient) -> None:
    response = authed_client.post(
        VALIDATE_PATH,
        json={"kind": "nonsense", "payload": {}},
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_list_configs_universe(authed_client: TestClient, webapp_config_root: Path) -> None:
    response = authed_client.get(f"{CONFIGS_PATH}/universe")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == [{"name": "spy"}]


def test_get_config_detail(authed_client: TestClient, webapp_config_root: Path) -> None:
    response = authed_client.get(f"{CONFIGS_PATH}/universe/spy")

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["name"] == "spy"
    assert body["parsed"] is not None
    assert body["parse_error"] is None
    assert "tickers" in body["raw"]


def test_get_config_detail_missing_returns_404(
    authed_client: TestClient, webapp_config_root: Path
) -> None:
    response = authed_client.get(f"{CONFIGS_PATH}/universe/nope")

    assert response.status_code == HTTPStatus.NOT_FOUND


def test_strategy_schema_known(authed_client: TestClient) -> None:
    response = authed_client.get(STRATEGY_SCHEMA_PATH.format(name="AdaptiveBollinger"))

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["name"] == "AdaptiveBollinger"
    param_names = {p["name"] for p in body["params"]}
    assert "window" in param_names
    # ``interval`` is framework-managed and must never appear as a form
    # field (see strategy_service._HIDDEN_PARAMS).
    assert "interval" not in param_names


def test_strategy_schema_unknown_returns_404(authed_client: TestClient) -> None:
    response = authed_client.get(STRATEGY_SCHEMA_PATH.format(name="NoSuchStrategy"))

    assert response.status_code == HTTPStatus.NOT_FOUND
