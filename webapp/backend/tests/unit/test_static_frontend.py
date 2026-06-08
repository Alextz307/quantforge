"""
Tests for serving the built single-page application from the backend.

The backend exposes the API under ``/api`` and, when a built frontend bundle is
present, mounts it at ``/`` so a single ``uvicorn`` process serves both. These
tests opt in to a temporary bundle; the autouse environment fixture points
every other test at an absent bundle so the API-only path stays the default.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.main import create_app

INDEX_MARKER = "QuantForge SPA shell"
INDEX_HTML = f"<!doctype html><title>{INDEX_MARKER}</title><div id=root></div>"
ASSET_JS = "console.log('app');"
DEEP_LINK = "/runs/some-experiment-id"


@pytest.fixture
def built_frontend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX_HTML)
    (dist / "assets" / "app.js").write_text(ASSET_JS)
    monkeypatch.setenv("WEBAPP_FRONTEND_DIST", str(dist))
    get_settings.cache_clear()
    return dist


def test_serves_index_at_root(built_frontend: Path) -> None:
    with TestClient(create_app()) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert INDEX_MARKER in resp.text


def test_serves_static_asset(built_frontend: Path) -> None:
    with TestClient(create_app()) as client:
        resp = client.get("/assets/app.js")
    assert resp.status_code == 200
    assert ASSET_JS in resp.text


def test_spa_fallback_for_client_route(built_frontend: Path) -> None:
    with TestClient(create_app()) as client:
        resp = client.get(DEEP_LINK)
    assert resp.status_code == 200
    assert INDEX_MARKER in resp.text


def test_api_still_served_under_api_prefix(built_frontend: Path) -> None:
    with TestClient(create_app()) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_unknown_api_path_is_not_spa_fallback(built_frontend: Path) -> None:
    with TestClient(create_app()) as client:
        resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404
    # A real API 404, not the SPA shell served with a 404 status.
    assert INDEX_MARKER not in resp.text


def test_non_404_error_is_not_spa_fallback(built_frontend: Path) -> None:
    # A disallowed method on a client-route path raises 405 from the static
    # layer; the fallback only rewrites 404s, so this must surface as 405.
    with TestClient(create_app()) as client:
        resp = client.post(DEEP_LINK)
    assert resp.status_code == 405
    assert INDEX_MARKER not in resp.text


def test_no_mount_when_bundle_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBAPP_FRONTEND_DIST", str(tmp_path / "absent"))
    get_settings.cache_clear()
    app = create_app()
    # Prove the bundle is genuinely unmounted -- a status-only check cannot tell
    # "no mount" apart from "mounted but missing index.html", both of which 404.
    assert all(getattr(route, "name", None) != "frontend" for route in app.routes)
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.status_code == 404
