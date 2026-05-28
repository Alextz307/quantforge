"""
Tests for the OpenAPI snapshot drift guard.
"""

from __future__ import annotations

import pytest

from src.core import json_io
from tests.conftest import REPO_ROOT, load_script_module

GUARD_SCRIPT = REPO_ROOT / "scripts" / "check_openapi_snapshot.py"
guard = load_script_module(GUARD_SCRIPT, "check_openapi_snapshot")


class TestRepoStateIsClean:
    """
    End-to-end: the committed snapshot must agree with the live FastAPI app.

    Skipped automatically if fastapi isn't installed (e.g. in the python-test job
    that runs on the research-framework deps only).
    """

    def test_committed_snapshot_matches_live_app(self) -> None:
        try:
            spec = guard.build_openapi_spec()
        except ModuleNotFoundError:
            pytest.skip("fastapi/webapp deps not installed in this environment")
        snapshot = REPO_ROOT / "webapp" / "frontend" / "openapi.snapshot.json"
        assert (
            json_io.diff_against_snapshot(
                spec,
                snapshot,
                label="OpenAPI snapshot",
                fix_command="make webapp-openapi-snapshot",
            )
            == []
        )
