"""Guard against CI/pyproject dependency drift.

Four checks:

1. Every runtime dependency declared in ``pyproject.toml`` appears in the
   ``python-test`` job's ``pip install`` line of ``.github/workflows/ci.yml``.
2. Every type-stub dev dependency (``types-*`` or ``*-stubs``) declared in
   ``pyproject.toml`` ``[project.optional-dependencies] dev`` appears in the
   ``lint-and-typecheck`` job's ``pip install`` line — otherwise mypy strict
   fails in CI with "Library stubs not installed for ...".
3. Every ``[webapp]`` extra dep plus the dev tooling the webapp tests/lint need
   (pytest, pytest-cov, mypy, ruff) appears in the ``webapp`` job's pip install.
4. Every ``[webapp]`` extra dep appears in the ``webapp-frontend`` job's pip
   install (it boots the FastAPI app to dump the OpenAPI snapshot).

Run locally with ``python scripts/check_ci_deps.py``; also wired into CI so a
forgotten dep update fails the same PR that introduced it.

Stdlib-only — intentionally avoids PyYAML so it runs in the lint job before the
project's own deps are installed.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_YAML = REPO_ROOT / ".github" / "workflows" / "ci.yml"

_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+")
# Greedy from the `python-test:` heading to the first `run: pip install` line
# that follows it — the job has exactly one such step.
_PYTHON_TEST_INSTALL_RE = re.compile(
    r"^\s*python-test:.*?run: pip install (?P<pkgs>[^\n]+)$",
    re.DOTALL | re.MULTILINE,
)
# Same shape for the lint-and-typecheck job's pip install line. It's the only
# install step in that job, so a greedy match from the heading works.
_LINT_INSTALL_RE = re.compile(
    r"^\s*lint-and-typecheck:.*?run: pip install (?P<pkgs>[^\n]+)$",
    re.DOTALL | re.MULTILINE,
)
_WEBAPP_INSTALL_RE = re.compile(
    r"^\s*webapp:.*?run: pip install (?P<pkgs>[^\n]+)$",
    re.DOTALL | re.MULTILINE,
)
_WEBAPP_FRONTEND_INSTALL_RE = re.compile(
    r"^\s*webapp-frontend:.*?run: pip install (?P<pkgs>[^\n]+)$",
    re.DOTALL | re.MULTILINE,
)

# The webapp job runs pytest/mypy/ruff against webapp/backend; these dev tools
# are not in [webapp] extras (they're shared dev deps) but must be present.
_WEBAPP_DEV_TOOLING = ("pytest", "pytest-cov", "mypy", "ruff")


def _extract_name(spec: str) -> str:
    """Canonicalize a PEP 508 / pip-CLI package spec to its bare name.

    Strips surrounding single/double quotes (CI YAML wraps extras like
    ``'uvicorn[standard]'`` so the shell doesn't glob them).
    """
    stripped = spec.strip("\"'")
    match = _PACKAGE_NAME_RE.match(stripped)
    if not match:
        raise ValueError(f"Unrecognized dep spec: {spec!r}")
    return match.group(0).lower().replace("_", "-")


def _is_type_stub(name: str) -> bool:
    """Heuristic for PEP 561 stub packages: ``types-*`` or ``*-stubs``."""
    return name.startswith("types-") or name.endswith("-stubs")


def _runtime_dep_names(pyproject_text: str) -> list[str]:
    """Canonicalized runtime dep names from a ``pyproject.toml`` text."""
    deps = tomllib.loads(pyproject_text)["project"]["dependencies"]
    return [_extract_name(d) for d in deps]


def _dev_type_stub_names(pyproject_text: str) -> list[str]:
    """Canonicalized type-stub names from the ``dev`` optional-deps group."""
    parsed = tomllib.loads(pyproject_text)
    dev_deps = parsed.get("project", {}).get("optional-dependencies", {}).get("dev", [])
    return [name for d in dev_deps if _is_type_stub(name := _extract_name(d))]


def _webapp_extra_names(pyproject_text: str) -> list[str]:
    """Canonicalized dep names from the ``webapp`` optional-deps group."""
    parsed = tomllib.loads(pyproject_text)
    webapp_deps = parsed.get("project", {}).get("optional-dependencies", {}).get("webapp", [])
    return [_extract_name(d) for d in webapp_deps]


def _ci_install_packages(ci_yaml_text: str, regex: re.Pattern[str], job_label: str) -> set[str]:
    match = regex.search(ci_yaml_text)
    if match is None:
        raise ValueError(f"could not find '{job_label}' pip install line in CI YAML")
    return {_extract_name(t) for t in match.group("pkgs").split()}


def find_missing_deps(pyproject_text: str, ci_yaml_text: str) -> list[str]:
    """Return sorted list of pyproject runtime deps absent from the CI pip install line.

    Raises:
        ValueError: if the ``python-test`` pip install line cannot be located.
    """
    runtime_deps = set(_runtime_dep_names(pyproject_text))
    ci_packages = _ci_install_packages(ci_yaml_text, _PYTHON_TEST_INSTALL_RE, "python-test")
    return sorted(runtime_deps - ci_packages)


def find_missing_type_stubs(pyproject_text: str, ci_yaml_text: str) -> list[str]:
    """Return sorted list of dev type-stub deps absent from the lint pip install line.

    Raises:
        ValueError: if the ``lint-and-typecheck`` pip install line cannot be located.
    """
    stubs = set(_dev_type_stub_names(pyproject_text))
    ci_packages = _ci_install_packages(ci_yaml_text, _LINT_INSTALL_RE, "lint-and-typecheck")
    return sorted(stubs - ci_packages)


def find_missing_webapp_deps(pyproject_text: str, ci_yaml_text: str) -> list[str]:
    """Return sorted list of [webapp] extras + dev tooling absent from the webapp job.

    Raises:
        ValueError: if the ``webapp`` pip install line cannot be located.
    """
    required = set(_webapp_extra_names(pyproject_text)) | set(_WEBAPP_DEV_TOOLING)
    ci_packages = _ci_install_packages(ci_yaml_text, _WEBAPP_INSTALL_RE, "webapp")
    return sorted(required - ci_packages)


def find_missing_webapp_frontend_deps(pyproject_text: str, ci_yaml_text: str) -> list[str]:
    """Return sorted list of [webapp] extras absent from the webapp-frontend job.

    Raises:
        ValueError: if the ``webapp-frontend`` pip install line cannot be located.
    """
    required = set(_webapp_extra_names(pyproject_text))
    ci_packages = _ci_install_packages(ci_yaml_text, _WEBAPP_FRONTEND_INSTALL_RE, "webapp-frontend")
    return sorted(required - ci_packages)


def main() -> int:
    pyproject_text = PYPROJECT.read_text()
    ci_text = CI_YAML.read_text()
    try:
        missing_runtime = find_missing_deps(pyproject_text, ci_text)
        missing_stubs = find_missing_type_stubs(pyproject_text, ci_text)
        missing_webapp = find_missing_webapp_deps(pyproject_text, ci_text)
        missing_webapp_frontend = find_missing_webapp_frontend_deps(pyproject_text, ci_text)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    failed = False
    if missing_runtime:
        print(
            "CI dep drift: pyproject.toml runtime deps missing from python-test pip install:",
            file=sys.stderr,
        )
        for name in missing_runtime:
            print(f"  - {name}", file=sys.stderr)
        failed = True
    if missing_stubs:
        print(
            "CI dep drift: pyproject.toml [dev] type-stubs missing from "
            "lint-and-typecheck pip install:",
            file=sys.stderr,
        )
        for name in missing_stubs:
            print(f"  - {name}", file=sys.stderr)
        failed = True
    if missing_webapp:
        print(
            "CI dep drift: pyproject.toml [webapp] extras / dev tooling missing from "
            "webapp pip install:",
            file=sys.stderr,
        )
        for name in missing_webapp:
            print(f"  - {name}", file=sys.stderr)
        failed = True
    if missing_webapp_frontend:
        print(
            "CI dep drift: pyproject.toml [webapp] extras missing from "
            "webapp-frontend pip install:",
            file=sys.stderr,
        )
        for name in missing_webapp_frontend:
            print(f"  - {name}", file=sys.stderr)
        failed = True

    if failed:
        return 1

    total_runtime = len(_runtime_dep_names(pyproject_text))
    total_stubs = len(_dev_type_stub_names(pyproject_text))
    total_webapp = len(_webapp_extra_names(pyproject_text))
    print(
        f"OK: all {total_runtime} runtime deps present in python-test CI step; "
        f"all {total_stubs} type-stubs present in lint-and-typecheck CI step; "
        f"all {total_webapp} [webapp] extras present in webapp + webapp-frontend CI steps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
