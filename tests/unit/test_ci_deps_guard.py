"""Tests for the CI/pyproject dependency drift guard."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_SCRIPT = REPO_ROOT / "scripts" / "check_ci_deps.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_YAML = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load_guard_module() -> ModuleType:
    """Import scripts/check_ci_deps.py by path (it's not under a package root)."""
    spec = importlib.util.spec_from_file_location("check_ci_deps", GUARD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard_module()

# Minimal synthetic YAML exercising the same `python-test:` → `run: pip install …`
# structure as the real workflow. Kept here to avoid depending on the repo's
# evolving runtime dep list.
_FAKE_CI_YAML = """\
name: CI
jobs:
  lint:
    steps:
      - run: pip install ruff
  python-test:
    steps:
      - name: Install
        run: pip install pytest pandas numpy
  other:
    steps:
      - run: echo hi
"""

_FAKE_PYPROJECT_ALL_PRESENT = """\
[project]
name = "x"
dependencies = ["pandas>=2.0", "numpy>=1.0"]
"""

_FAKE_PYPROJECT_WITH_MISSING = """\
[project]
name = "x"
dependencies = ["pandas>=2.0", "numpy>=1.0", "pmdarima>=2.0", "xgboost"]
"""


class TestFindMissingDeps:
    def test_no_drift_when_all_deps_present(self) -> None:
        assert guard.find_missing_deps(_FAKE_PYPROJECT_ALL_PRESENT, _FAKE_CI_YAML) == []

    def test_reports_missing_deps_sorted(self) -> None:
        missing = guard.find_missing_deps(_FAKE_PYPROJECT_WITH_MISSING, _FAKE_CI_YAML)
        assert missing == ["pmdarima", "xgboost"]

    def test_raises_when_python_test_job_absent(self) -> None:
        ci_without_job = _FAKE_CI_YAML.replace("python-test:", "renamed-job:")
        with pytest.raises(ValueError, match="python-test"):
            guard.find_missing_deps(_FAKE_PYPROJECT_ALL_PRESENT, ci_without_job)

    def test_version_specifiers_are_stripped(self) -> None:
        pyproject = '[project]\nname = "x"\ndependencies = ["pandas>=2.2.0", "numpy~=1.26"]\n'
        ci = "jobs:\n  python-test:\n    steps:\n      - run: pip install pandas numpy\n"
        assert guard.find_missing_deps(pyproject, ci) == []

    def test_extras_are_stripped(self) -> None:
        pyproject = '[project]\nname = "x"\ndependencies = ["pydantic[email]>=2.0"]\n'
        ci = "jobs:\n  python-test:\n    steps:\n      - run: pip install pydantic\n"
        assert guard.find_missing_deps(pyproject, ci) == []


class TestRepoStateIsClean:
    """End-to-end: the real repo's pyproject/ci.yml must pass the guard."""

    def test_real_repo_has_no_drift(self) -> None:
        missing = guard.find_missing_deps(PYPROJECT.read_text(), CI_YAML.read_text())
        assert missing == []
