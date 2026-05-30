"""
Integration tests for /api/configs/study/uploads + study_spec/{schema,validate}.
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from webapp.backend.app.core.settings import get_settings

UPLOADS_PATH = "/api/configs/study/uploads"
VALIDATE_PATH = "/api/configs/study_spec/validate"
SCHEMA_PATH = "/api/configs/study_spec/schema"


@pytest.fixture
def study_config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Plant the strategy/hpo/universe files an upload validates against.
    """

    root = tmp_path / "config"
    (root / "strategies").mkdir(parents=True)
    (root / "hpo").mkdir()
    (root / "universes").mkdir()
    (root / "study").mkdir()
    (root / "strategies" / "adaptive_bollinger.yaml").write_text("body\n")
    (root / "hpo" / "adaptive_bollinger.yaml").write_text("body\n")
    (root / "universes" / "spy_daily_5y.yaml").write_text("body\n")
    (root / "study" / "library_only.yaml").write_text("body\n")
    monkeypatch.setenv("WEBAPP_CONFIG_ROOT", str(root))
    get_settings.cache_clear()
    return root


def _valid_yaml(config_root: Path) -> str:
    return yaml.safe_dump(
        {
            "name": "my_study",
            "description": "Toy 1-leg study.",
            "seed": 42,
            "output_dir": "studies/my_study",
            "legs": [
                {
                    "strategy": "AdaptiveBollinger",
                    "strategy_config": str(config_root / "strategies" / "adaptive_bollinger.yaml"),
                    "hpo_config": str(config_root / "hpo" / "adaptive_bollinger.yaml"),
                    "universes": ["spy_daily_5y"],
                }
            ],
        }
    )


def test_uploads_require_auth(client: TestClient) -> None:
    assert client.get(UPLOADS_PATH).status_code == HTTPStatus.UNAUTHORIZED
    assert (
        client.post(UPLOADS_PATH, json={"slug": "x", "yaml": "x"}).status_code
        == HTTPStatus.UNAUTHORIZED
    )
    assert client.post(VALIDATE_PATH, json={"yaml": "x"}).status_code == HTTPStatus.UNAUTHORIZED


def test_schema_carries_descriptions(authed_client: TestClient) -> None:
    response = authed_client.get(SCHEMA_PATH)
    assert response.status_code == HTTPStatus.OK
    schema = response.json()
    top = schema["properties"]
    assert "description" in top["name"]
    assert "study" in top["name"]["description"].lower()
    defs = schema.get("$defs") or schema.get("definitions") or {}
    leg = defs["StudyLeg"]["properties"]
    assert "description" in leg["strategy"]
    assert "registered" in leg["strategy"]["description"].lower()


def test_validate_happy_path(authed_client: TestClient, study_config_root: Path) -> None:
    response = authed_client.post(VALIDATE_PATH, json={"yaml": _valid_yaml(study_config_root)})
    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body == {"valid": True, "errors": []}


def test_validate_yaml_parse_error(authed_client: TestClient) -> None:
    response = authed_client.post(VALIDATE_PATH, json={"yaml": "name: [unterminated"})
    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["valid"] is False
    assert body["errors"][0]["loc"] == ["yaml"]


def test_validate_unknown_universe(authed_client: TestClient, study_config_root: Path) -> None:
    yaml_text = _valid_yaml(study_config_root).replace("spy_daily_5y", "ghost_universe")
    response = authed_client.post(VALIDATE_PATH, json={"yaml": yaml_text})
    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["valid"] is False
    assert any(err["loc"] == ["legs", "0", "universes", "0"] for err in body["errors"])


def test_create_list_get_delete_roundtrip(
    authed_jobs_client: TestClient, study_config_root: Path
) -> None:
    """
    authed_jobs_client provides the WEBAPP_STUDY_SPEC_UPLOADS_DIR env.
    """

    yaml_text = _valid_yaml(study_config_root)
    create = authed_jobs_client.post(UPLOADS_PATH, json={"slug": "my_study", "yaml": yaml_text})
    assert create.status_code == HTTPStatus.CREATED
    body = create.json()
    assert body["slug"] == "my_study"
    assert "my_study" in body["yaml"]

    listing = authed_jobs_client.get(UPLOADS_PATH)
    assert listing.status_code == HTTPStatus.OK
    assert [u["slug"] for u in listing.json()] == ["my_study"]

    detail = authed_jobs_client.get(f"{UPLOADS_PATH}/my_study")
    assert detail.status_code == HTTPStatus.OK
    assert "my_study" in detail.json()["yaml"]

    delete = authed_jobs_client.delete(f"{UPLOADS_PATH}/my_study")
    assert delete.status_code == HTTPStatus.NO_CONTENT
    assert authed_jobs_client.get(UPLOADS_PATH).json() == []


def test_create_rejects_library_collision(
    authed_jobs_client: TestClient, study_config_root: Path
) -> None:
    response = authed_jobs_client.post(
        UPLOADS_PATH,
        json={"slug": "library_only", "yaml": _valid_yaml(study_config_root)},
    )
    assert response.status_code == HTTPStatus.CONFLICT
    assert "library" in response.json()["detail"].lower()


def test_create_returns_422_on_invalid_yaml(
    authed_jobs_client: TestClient, study_config_root: Path
) -> None:
    response = authed_jobs_client.post(UPLOADS_PATH, json={"slug": "broken", "yaml": "name: [bad"})
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert detail[0]["loc"] == ["yaml"]


def test_get_missing_returns_404(authed_jobs_client: TestClient) -> None:
    response = authed_jobs_client.get(f"{UPLOADS_PATH}/ghost")
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_delete_missing_returns_404(authed_jobs_client: TestClient) -> None:
    response = authed_jobs_client.delete(f"{UPLOADS_PATH}/ghost")
    assert response.status_code == HTTPStatus.NOT_FOUND
