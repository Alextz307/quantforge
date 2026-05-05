"""Tests for the CI/pyproject dependency drift guard."""
# Long YAML/TOML fixture lines mirror real CI shapes verbatim — wrapping them
# breaks the regex anchoring the guard relies on.
# ruff: noqa: E501

from __future__ import annotations

import pytest

from tests.conftest import REPO_ROOT, load_script_module

GUARD_SCRIPT = REPO_ROOT / "scripts" / "check_ci_deps.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_YAML = REPO_ROOT / ".github" / "workflows" / "ci.yml"

guard = load_script_module(GUARD_SCRIPT, "check_ci_deps")

# Minimal synthetic YAML exercising the same `python-test:` → `run: pip install …`
# structure as the real workflow. Kept here to avoid depending on the repo's
# evolving runtime dep list.
_FAKE_CI_YAML = """\
name: CI
jobs:
  lint-and-typecheck:
    steps:
      - run: pip install ruff mypy pandas-stubs types-PyYAML
  python-test:
    steps:
      - name: Install
        run: pip install pytest pandas numpy
  webapp:
    steps:
      - name: Install
        run: pip install fastapi 'uvicorn[standard]' bcrypt itsdangerous slowapi httpx pytest pytest-cov mypy ruff
  webapp-frontend:
    steps:
      - name: Install
        run: pip install fastapi 'uvicorn[standard]' bcrypt itsdangerous slowapi httpx
  other:
    steps:
      - run: echo hi
"""

_FAKE_PYPROJECT_WITH_WEBAPP = """\
[project]
name = "x"
dependencies = ["pandas>=2.0", "numpy>=1.0"]

[project.optional-dependencies]
dev = ["mypy>=1.8", "pandas-stubs", "types-PyYAML", "ruff"]
webapp = ["fastapi>=0.115", "uvicorn[standard]>=0.32", "bcrypt", "itsdangerous", "slowapi", "httpx"]
"""

_FAKE_PYPROJECT_WITH_MISSING_WEBAPP = """\
[project]
name = "x"
dependencies = ["pandas>=2.0", "numpy>=1.0"]

[project.optional-dependencies]
dev = ["mypy>=1.8", "pandas-stubs", "types-PyYAML", "ruff"]
webapp = ["fastapi>=0.115", "uvicorn[standard]>=0.32", "bcrypt", "itsdangerous", "slowapi", "httpx", "python-multipart"]
"""

_FAKE_PYPROJECT_ALL_PRESENT = """\
[project]
name = "x"
dependencies = ["pandas>=2.0", "numpy>=1.0"]

[project.optional-dependencies]
dev = ["mypy>=1.8", "pandas-stubs", "types-PyYAML", "ruff"]
"""

_FAKE_PYPROJECT_WITH_MISSING = """\
[project]
name = "x"
dependencies = ["pandas>=2.0", "numpy>=1.0", "pmdarima>=2.0", "xgboost"]

[project.optional-dependencies]
dev = ["mypy>=1.8", "pandas-stubs", "types-PyYAML", "ruff"]
"""

_FAKE_PYPROJECT_WITH_MISSING_STUB = """\
[project]
name = "x"
dependencies = ["pandas>=2.0", "numpy>=1.0"]

[project.optional-dependencies]
dev = ["mypy>=1.8", "pandas-stubs", "types-PyYAML", "types-psutil", "ruff"]
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


class TestFindMissingTypeStubs:
    def test_no_drift_when_all_stubs_present(self) -> None:
        assert guard.find_missing_type_stubs(_FAKE_PYPROJECT_ALL_PRESENT, _FAKE_CI_YAML) == []

    def test_reports_missing_stubs_sorted(self) -> None:
        missing = guard.find_missing_type_stubs(_FAKE_PYPROJECT_WITH_MISSING_STUB, _FAKE_CI_YAML)
        assert missing == ["types-psutil"]

    def test_raises_when_lint_job_absent(self) -> None:
        ci_without_job = _FAKE_CI_YAML.replace("lint-and-typecheck:", "renamed-job:")
        with pytest.raises(ValueError, match="lint-and-typecheck"):
            guard.find_missing_type_stubs(_FAKE_PYPROJECT_ALL_PRESENT, ci_without_job)

    def test_non_stub_dev_deps_are_ignored(self) -> None:
        # `mypy` and `ruff` are dev deps but NOT type stubs — should not be flagged
        # even though they aren't in the CI lint pip install line. This test
        # exists because the check_ci_deps script only validates ``types-*`` and
        # ``*-stubs`` packages; tools like mypy / ruff are installed elsewhere.
        pyproject = (
            '[project]\nname = "x"\ndependencies = []\n'
            '[project.optional-dependencies]\ndev = ["mypy", "ruff"]\n'
        )
        ci = "jobs:\n  lint-and-typecheck:\n    steps:\n      - run: pip install ruff mypy\n"
        assert guard.find_missing_type_stubs(pyproject, ci) == []


class TestFindMissingWebappDeps:
    def test_no_drift_when_all_webapp_deps_present(self) -> None:
        assert guard.find_missing_webapp_deps(_FAKE_PYPROJECT_WITH_WEBAPP, _FAKE_CI_YAML) == []

    def test_reports_missing_webapp_extra_dep(self) -> None:
        missing = guard.find_missing_webapp_deps(_FAKE_PYPROJECT_WITH_MISSING_WEBAPP, _FAKE_CI_YAML)
        assert missing == ["python-multipart"]

    def test_reports_missing_dev_tooling(self) -> None:
        ci_without_cov = _FAKE_CI_YAML.replace(" pytest-cov", "")
        missing = guard.find_missing_webapp_deps(_FAKE_PYPROJECT_WITH_WEBAPP, ci_without_cov)
        assert missing == ["pytest-cov"]

    def test_raises_when_webapp_job_absent(self) -> None:
        ci_without_job = _FAKE_CI_YAML.replace("webapp:", "renamed-job:")
        with pytest.raises(ValueError, match="webapp"):
            guard.find_missing_webapp_deps(_FAKE_PYPROJECT_WITH_WEBAPP, ci_without_job)


class TestFindMissingWebappFrontendDeps:
    def test_no_drift_when_all_webapp_extras_present(self) -> None:
        assert (
            guard.find_missing_webapp_frontend_deps(_FAKE_PYPROJECT_WITH_WEBAPP, _FAKE_CI_YAML)
            == []
        )

    def test_reports_missing_webapp_extra_dep(self) -> None:
        missing = guard.find_missing_webapp_frontend_deps(
            _FAKE_PYPROJECT_WITH_MISSING_WEBAPP, _FAKE_CI_YAML
        )
        assert missing == ["python-multipart"]

    def test_dev_tooling_not_required_in_frontend_job(self) -> None:
        # The webapp-frontend job only boots FastAPI for the OpenAPI dump; it
        # doesn't run pytest/mypy/ruff. The frontend check should ignore the
        # absence of pytest/mypy/ruff even though the webapp check would flag it.
        pyproject = _FAKE_PYPROJECT_WITH_WEBAPP
        assert guard.find_missing_webapp_frontend_deps(pyproject, _FAKE_CI_YAML) == []
        ci_without_test_tools = _FAKE_CI_YAML.replace("httpx pytest pytest-cov mypy ruff", "httpx")
        assert sorted(guard.find_missing_webapp_deps(pyproject, ci_without_test_tools)) == [
            "mypy",
            "pytest",
            "pytest-cov",
            "ruff",
        ]


class TestRepoStateIsClean:
    """End-to-end: the real repo's pyproject/ci.yml must pass the guard."""

    def test_real_repo_runtime_deps_clean(self) -> None:
        missing = guard.find_missing_deps(PYPROJECT.read_text(), CI_YAML.read_text())
        assert missing == []

    def test_real_repo_type_stubs_clean(self) -> None:
        missing = guard.find_missing_type_stubs(PYPROJECT.read_text(), CI_YAML.read_text())
        assert missing == []

    def test_real_repo_webapp_deps_clean(self) -> None:
        missing = guard.find_missing_webapp_deps(PYPROJECT.read_text(), CI_YAML.read_text())
        assert missing == []

    def test_real_repo_webapp_frontend_deps_clean(self) -> None:
        missing = guard.find_missing_webapp_frontend_deps(
            PYPROJECT.read_text(), CI_YAML.read_text()
        )
        assert missing == []
