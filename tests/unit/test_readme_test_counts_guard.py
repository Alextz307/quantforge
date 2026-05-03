"""Tests for the README test-count drift guard."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.conftest import REPO_ROOT, load_script_module

GUARD_SCRIPT = REPO_ROOT / "scripts" / "check_readme_test_counts.py"
guard = load_script_module(GUARD_SCRIPT, "check_readme_test_counts")

_FIXTURE_PY = 7
_FIXTURE_SKIPS = 1
_FIXTURE_CPP = 3
_VALID_PHRASE = (
    f"CI is green on Linux and macOS with **{_FIXTURE_PY} Python tests** "
    f"(+{_FIXTURE_SKIPS} opt-in skips), **{_FIXTURE_CPP} C++ tests**, ruff clean, mypy clean."
)


class TestParseReadmeCounts:
    def test_parses_well_formed_phrase(self) -> None:
        assert guard.parse_readme_counts(_VALID_PHRASE) == (
            _FIXTURE_PY,
            _FIXTURE_SKIPS,
            _FIXTURE_CPP,
        )

    def test_handles_phrase_anywhere_in_text(self) -> None:
        long_text = "lorem ipsum\n" + _VALID_PHRASE + "\nmore prose"
        assert guard.parse_readme_counts(long_text) == (
            _FIXTURE_PY,
            _FIXTURE_SKIPS,
            _FIXTURE_CPP,
        )

    def test_raises_when_phrase_missing(self) -> None:
        with pytest.raises(ValueError, match="expected test-count phrase"):
            guard.parse_readme_counts("no test counts mentioned here")

    def test_raises_when_python_bold_dropped(self) -> None:
        broken = (
            f"with {_FIXTURE_PY} Python tests (+{_FIXTURE_SKIPS} opt-in skips), "
            f"**{_FIXTURE_CPP} C++ tests**"
        )
        with pytest.raises(ValueError, match="expected test-count phrase"):
            guard.parse_readme_counts(broken)


class TestRewriteReadmeCounts:
    """Round-trip test for the ``--fix`` rewrite path."""

    def test_rewrite_substitutes_both_bold_spans(self) -> None:
        rewritten = guard.rewrite_readme_counts(_VALID_PHRASE, py_passed=42, py_skipped=2, cpp=99)
        assert guard.parse_readme_counts(rewritten) == (42, 2, 99)

    def test_rewrite_leaves_cpp_untouched_when_cpp_is_none(self) -> None:
        rewritten = guard.rewrite_readme_counts(_VALID_PHRASE, py_passed=42, py_skipped=2, cpp=None)
        assert guard.parse_readme_counts(rewritten) == (42, 2, _FIXTURE_CPP)


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
