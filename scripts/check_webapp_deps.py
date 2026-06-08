"""
Guard against webapp dependency drift.

Two checks, analogous to ``check_ci_deps.py`` but for the web-application tier:

1. Every third-party package the backend (``webapp/backend/app``) imports is
   declared in either the ``[webapp]`` optional-dependency extra or the base
   ``[project] dependencies`` of ``pyproject.toml`` -- so a backend module can
   never import a package that only happens to be present transitively.
2. The continuous-integration ``webapp`` job installs the project with the
   ``[webapp]`` extra, so CI resolves the same set the check validates.

Run locally with ``python scripts/check_webapp_deps.py``; also wired into the
lint job so a forgotten dependency fails the same PR that introduced it.

Stdlib-only (``ast``, ``tomllib``, ``sys.stdlib_module_names``) -- it runs in
the lint job before the project's own dependencies are installed.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_YAML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
BACKEND_DIR = REPO_ROOT / "webapp" / "backend" / "app"

# Import roots that resolve to a different distribution name than the module.
# Pure ``_``->``-`` casing (e.g. ``pydantic_settings`` -> ``pydantic-settings``)
# is handled generically; only genuine renames need an entry here.
_IMPORT_TO_DISTRIBUTION = {
    "yaml": "pyyaml",
}

# First-party roots that live in this repository, not on PyPI.
_FIRST_PARTY_ROOTS = frozenset({"src", "webapp", "quant_engine"})

_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+")
# Isolate the ``webapp:`` job's own block -- from its heading to the next sibling
# job at the same indent (or end of file) -- so the pip-install search below can
# never leak across a job boundary onto, say, the adjacent ``webapp-frontend:``
# job's byte-identical ``[webapp]`` install line and mask a real regression.
# ``webapp-frontend:`` does not match the heading because the colon must follow
# ``webapp`` directly.
_WEBAPP_JOB_RE = re.compile(
    r"^(?P<indent>[ \t]*)webapp:[ \t]*\n(?P<body>.*?)(?=^(?P=indent)\S|\Z)",
    re.DOTALL | re.MULTILINE,
)
_PIP_INSTALL_RE = re.compile(r"run: pip install (?P<spec>[^\n]+)")
_WEBAPP_EXTRA_RE = re.compile(r"\[[^\]]*webapp[^\]]*\]")


def _canonical_distribution(name: str) -> str:
    """
    Canonicalize a distribution name: lowercase with ``_`` mapped to ``-``.
    """

    return name.lower().replace("_", "-")


def _extract_name(spec: str) -> str:
    """
    Canonicalize a PEP 508 dependency spec to its bare distribution name,
    stripping version specifiers, extras, and surrounding quotes.
    """

    stripped = spec.strip("\"'")
    match = _PACKAGE_NAME_RE.match(stripped)
    if not match:
        raise ValueError(f"Unrecognized dep spec: {spec!r}")
    return _canonical_distribution(match.group(0))


def import_roots_in_source(source: str) -> set[str]:
    """
    Root module names imported by a Python source string. Relative imports
    (``from . import x``) are first-party and excluded.
    """

    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _is_third_party(root: str) -> bool:
    return root not in sys.stdlib_module_names and root not in _FIRST_PARTY_ROOTS


def roots_to_distributions(roots: set[str]) -> set[str]:
    """
    Map third-party import roots to their canonical distribution names.
    """

    dists: set[str] = set()
    for root in roots:
        if not _is_third_party(root):
            continue
        dists.add(_canonical_distribution(_IMPORT_TO_DISTRIBUTION.get(root, root)))
    return dists


def covered_distributions(pyproject_text: str) -> set[str]:
    """
    Distributions available to the backend: the base runtime dependencies plus
    the ``[webapp]`` optional-dependency extra.
    """

    parsed = tomllib.loads(pyproject_text)
    project = parsed.get("project", {})
    base = project.get("dependencies", [])
    webapp = project.get("optional-dependencies", {}).get("webapp", [])
    return {_extract_name(d) for d in [*base, *webapp]}


def scan_backend_import_roots(backend_dir: Path) -> set[str]:
    """
    Union of import roots across every Python module under ``backend_dir``.
    """

    roots: set[str] = set()
    for path in sorted(backend_dir.rglob("*.py")):
        roots |= import_roots_in_source(path.read_text())
    return roots


def find_uncovered_imports(import_roots: set[str], pyproject_text: str) -> list[str]:
    """
    Sorted distribution names the backend imports but neither the ``[webapp]``
    extra nor the base dependencies declare.
    """

    needed = roots_to_distributions(import_roots)
    return sorted(needed - covered_distributions(pyproject_text))


def webapp_job_installs_extra(ci_yaml_text: str) -> bool:
    """
    Whether the CI ``webapp`` job installs the project with a ``[...webapp...]``
    extra. The pip-install search is scoped to the ``webapp`` job's own block so
    it cannot latch onto a sibling job's install line. Raises if the job, or its
    pip-install line, cannot be located.
    """

    job = _WEBAPP_JOB_RE.search(ci_yaml_text)
    if job is None:
        raise ValueError("could not find 'webapp' job in CI YAML")
    install = _PIP_INSTALL_RE.search(job.group("body"))
    if install is None:
        raise ValueError("could not find 'webapp' job pip install line in CI YAML")
    return _WEBAPP_EXTRA_RE.search(install.group("spec")) is not None


def main() -> int:
    pyproject_text = PYPROJECT.read_text()
    ci_text = CI_YAML.read_text()
    import_roots = scan_backend_import_roots(BACKEND_DIR)

    failed = False
    uncovered = find_uncovered_imports(import_roots, pyproject_text)
    if uncovered:
        print(
            "Webapp dep drift: backend imports not declared in [webapp] extra "
            "or base dependencies of pyproject.toml:",
            file=sys.stderr,
        )
        for name in uncovered:
            print(f"  - {name}", file=sys.stderr)
        failed = True

    try:
        installs_extra = webapp_job_installs_extra(ci_text)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if not installs_extra:
        print(
            "Webapp dep drift: CI 'webapp' job does not install the [webapp] extra.",
            file=sys.stderr,
        )
        failed = True

    if failed:
        return 1

    covered = len(roots_to_distributions(import_roots))
    print(
        f"OK: all {covered} third-party backend imports are declared in "
        "pyproject; CI installs the [webapp] extra"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
