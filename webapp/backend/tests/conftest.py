"""Shared pytest fixtures for webapp/backend tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from webapp.backend.app.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Yield a TestClient bound to a fresh app instance with lifespan exercised."""
    with TestClient(create_app()) as test_client:
        yield test_client
