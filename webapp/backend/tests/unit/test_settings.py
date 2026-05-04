"""Verify settings defaults, WEBAPP_* env-prefix resolution, and frozen behaviour."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from webapp.backend.app.core.settings import WebappEnv, WebappSettings, get_settings


@pytest.fixture(autouse=True)
def _isolate_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_defaults() -> None:
    settings = get_settings()

    assert isinstance(settings, WebappSettings)
    assert settings.env is WebappEnv.LOCAL
    assert settings.store_root == Path("experiment_results")


def test_settings_resolve_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBAPP_ENV", "development")

    settings = get_settings()

    assert settings.env is WebappEnv.DEVELOPMENT


def test_settings_are_frozen() -> None:
    settings = get_settings()

    with pytest.raises(ValidationError):
        settings.store_root = Path("/tmp")
