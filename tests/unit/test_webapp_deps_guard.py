"""
Tests for the webapp dependency drift guard.

Long YAML/TOML fixture lines mirror real CI shapes verbatim - wrapping them
breaks the regex anchoring the guard relies on.
"""
# ruff: noqa: E501

from __future__ import annotations

import pytest

from tests.conftest import REPO_ROOT, load_script_module

GUARD_SCRIPT = REPO_ROOT / "scripts" / "check_webapp_deps.py"

guard = load_script_module(GUARD_SCRIPT, "check_webapp_deps")

_FAKE_CI_YAML = """\
name: CI
jobs:
  lint-and-typecheck:
    steps:
      - run: pip install ruff mypy
  webapp:
    steps:
      - name: Install project + dev + webapp dependencies
        run: pip install -e ".[dev,webapp]"
  webapp-frontend:
    steps:
      - run: pip install -e ".[webapp]"
"""

_FAKE_PYPROJECT = """\
[project]
name = "x"
dependencies = ["pandas>=2.2", "pyyaml>=6.0"]

[project.optional-dependencies]
webapp = ["fastapi==0.136.1", "starlette==1.0.0", "pydantic-settings==2.14.0"]
"""


class TestImportRootsInSource:
    def test_collects_plain_and_from_imports(self) -> None:
        source = "import os.path\nimport bcrypt\nfrom fastapi import APIRouter\n"
        assert guard.import_roots_in_source(source) == {"os", "bcrypt", "fastapi"}

    def test_relative_imports_excluded(self) -> None:
        source = "from . import sibling\nfrom .pkg import thing\nimport yaml\n"
        assert guard.import_roots_in_source(source) == {"yaml"}


class TestRootsToDistributions:
    def test_maps_aliases_and_drops_stdlib_and_first_party(self) -> None:
        roots = {
            "yaml",
            "pydantic_settings",
            "bcrypt",
            "os",
            "sys",
            "src",
            "webapp",
            "quant_engine",
        }
        assert guard.roots_to_distributions(roots) == {"pyyaml", "pydantic-settings", "bcrypt"}


class TestFindUncoveredImports:
    def test_no_drift_when_all_imports_declared(self) -> None:
        roots = {"fastapi", "pandas", "yaml", "os", "src"}
        assert guard.find_uncovered_imports(roots, _FAKE_PYPROJECT) == []

    def test_reports_uncovered_sorted(self) -> None:
        roots = {"fastapi", "requests", "redis"}
        assert guard.find_uncovered_imports(roots, _FAKE_PYPROJECT) == ["redis", "requests"]

    def test_base_dependency_covers_backend_import(self) -> None:
        # ``pandas`` lives in base [project] deps, not the [webapp] extra.
        assert guard.find_uncovered_imports({"pandas"}, _FAKE_PYPROJECT) == []


class TestWebappJobInstallsExtra:
    def test_true_when_extra_present(self) -> None:
        assert guard.webapp_job_installs_extra(_FAKE_CI_YAML) is True

    def test_false_when_no_extra(self) -> None:
        ci = _FAKE_CI_YAML.replace('".[dev,webapp]"', '"."')
        assert guard.webapp_job_installs_extra(ci) is False

    def test_raises_when_webapp_job_absent(self) -> None:
        ci = _FAKE_CI_YAML.replace("webapp:", "renamed-job:")
        with pytest.raises(ValueError, match="webapp"):
            guard.webapp_job_installs_extra(ci)

    def test_does_not_leak_onto_sibling_job_install(self) -> None:
        # The webapp job loses its own pip install; the only `[webapp]` install
        # left belongs to the adjacent webapp-frontend job. The check must fail
        # loudly rather than leak across the boundary and report a false PASS.
        ci = _FAKE_CI_YAML.replace(
            "      - name: Install project + dev + webapp dependencies\n"
            '        run: pip install -e ".[dev,webapp]"\n',
            '      - run: echo "no install in this job"\n',
        )
        with pytest.raises(ValueError, match="webapp"):
            guard.webapp_job_installs_extra(ci)


class TestRepoStateIsClean:
    """
    End-to-end: the real backend imports and CI workflow must pass the guard.
    """

    def test_real_backend_imports_are_declared(self) -> None:
        roots = guard.scan_backend_import_roots(guard.BACKEND_DIR)
        assert guard.find_uncovered_imports(roots, guard.PYPROJECT.read_text()) == []

    def test_real_webapp_job_installs_extra(self) -> None:
        assert guard.webapp_job_installs_extra(guard.CI_YAML.read_text()) is True
