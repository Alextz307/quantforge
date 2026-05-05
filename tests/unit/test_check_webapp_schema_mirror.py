"""Tests for the Pydantic ↔ zod schema-mirror drift guard."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from src.core import json_io
from tests.conftest import REPO_ROOT, load_script_module

GUARD_SCRIPT = REPO_ROOT / "scripts" / "check_webapp_schema_mirror.py"
guard = load_script_module(GUARD_SCRIPT, "check_webapp_schema_mirror")


class _SampleLogin(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=256)


class TestExtractFieldShape:
    def test_extracts_type_and_min_max(self) -> None:
        shape = guard.extract_field_shape(_SampleLogin)
        assert shape == {
            "username": {"type": "string", "min": 1, "max": 64},
            "password": {"type": "string", "min": 8, "max": 256},
        }

    def test_rejects_non_pydantic_model(self) -> None:
        with pytest.raises(TypeError, match="pydantic v2"):
            guard.extract_field_shape(object)


class TestRepoStateIsClean:
    """End-to-end: the committed mirror snapshot must agree with the live models.

    Skipped automatically if fastapi/webapp deps aren't installed.
    """

    def test_committed_snapshot_matches_live_models(self) -> None:
        try:
            shape = guard._build_mirror_shape()
        except ModuleNotFoundError:
            pytest.skip("fastapi/webapp deps not installed in this environment")
        snapshot = REPO_ROOT / "webapp" / "frontend" / "schema-mirror.snapshot.json"
        assert (
            json_io.diff_against_snapshot(
                shape,
                snapshot,
                label="Schema-mirror snapshot",
                fix_command="python -m scripts.check_webapp_schema_mirror --write",
            )
            == []
        )
