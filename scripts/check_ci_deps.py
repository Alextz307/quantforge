"""Guard against CI/pyproject dependency drift.

Asserts that every runtime dependency declared in ``pyproject.toml`` appears in
the ``python-test`` job's ``pip install`` line of ``.github/workflows/ci.yml``.

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


def _extract_name(spec: str) -> str:
    """Canonicalize a PEP 508 / pip-CLI package spec to its bare name."""
    match = _PACKAGE_NAME_RE.match(spec)
    if not match:
        raise ValueError(f"Unrecognized dep spec: {spec!r}")
    return match.group(0).lower().replace("_", "-")


def _runtime_dep_names(pyproject_text: str) -> list[str]:
    """Canonicalized runtime dep names from a ``pyproject.toml`` text."""
    deps = tomllib.loads(pyproject_text)["project"]["dependencies"]
    return [_extract_name(d) for d in deps]


def find_missing_deps(pyproject_text: str, ci_yaml_text: str) -> list[str]:
    """Return sorted list of pyproject runtime deps absent from the CI pip install line.

    Raises:
        ValueError: if the ``python-test`` pip install line cannot be located.
    """
    runtime_deps = set(_runtime_dep_names(pyproject_text))

    match = _PYTHON_TEST_INSTALL_RE.search(ci_yaml_text)
    if match is None:
        raise ValueError("could not find 'python-test' pip install line in CI YAML")
    ci_packages = {_extract_name(t) for t in match.group("pkgs").split()}

    return sorted(runtime_deps - ci_packages)


def main() -> int:
    pyproject_text = PYPROJECT.read_text()
    try:
        missing = find_missing_deps(pyproject_text, CI_YAML.read_text())
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if missing:
        print(
            "CI dep drift: pyproject.toml runtime deps missing from python-test pip install:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        return 1
    total = len(_runtime_dep_names(pyproject_text))
    print(f"OK: all {total} runtime deps present in python-test CI step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
