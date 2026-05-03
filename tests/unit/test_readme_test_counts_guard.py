"""Tests for the README test-count drift guard."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_SCRIPT = REPO_ROOT / "scripts" / "check_readme_test_counts.py"


def _load_guard_module() -> ModuleType:
    """Import scripts/check_readme_test_counts.py by path."""
    spec = importlib.util.spec_from_file_location("check_readme_test_counts", GUARD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard_module()

_VALID_PHRASE = (
    "CI is green on Linux and macOS with **1149 Python tests** "
    "(+18 opt-in skips), **222 C++ tests**, ruff clean, mypy clean."
)


class TestParseReadmeCounts:
    def test_parses_well_formed_phrase(self) -> None:
        assert guard.parse_readme_counts(_VALID_PHRASE) == (1149, 18, 222)

    def test_handles_phrase_anywhere_in_text(self) -> None:
        long_text = "lorem ipsum\n" + _VALID_PHRASE + "\nmore prose"
        assert guard.parse_readme_counts(long_text) == (1149, 18, 222)

    def test_raises_when_phrase_missing(self) -> None:
        with pytest.raises(ValueError, match="expected test-count phrase"):
            guard.parse_readme_counts("no test counts mentioned here")

    def test_raises_when_python_bold_dropped(self) -> None:
        broken = "with 1149 Python tests (+18 opt-in skips), **222 C++ tests**"
        with pytest.raises(ValueError, match="expected test-count phrase"):
            guard.parse_readme_counts(broken)


@pytest.mark.skipif(
    os.environ.get("RUN_README_DRIFT") != "1",
    reason="set RUN_README_DRIFT=1 to run the readme drift integration test "
    "(~3s; CI lint job runs the script directly so this is local-only)",
)
class TestRepoStateIsClean:
    """End-to-end: invoke the script on the real repo, must exit 0."""

    def test_real_repo_readme_in_sync(self) -> None:
        result = subprocess.run(
            [sys.executable, str(GUARD_SCRIPT)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"check_readme_test_counts.py failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
