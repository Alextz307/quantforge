"""Guard against ``README.md`` test-count drift.

Compares the test counts referenced in ``README.md`` against the live values
reported by the test runners:

* Python — ``pytest --collect-only -q tests/``. The README quotes both the
  passing total and the opt-in skip total, so the source-of-truth comparison
  is ``passing + skipped == collected``.
* C++ — ``ctest --test-dir cpp/build -N``. Skipped (downgraded to a notice,
  not a failure) when ``cpp/build/`` doesn't exist, since the lint CI job
  doesn't build C++. The ``cpp-build-and-test`` matrix job verifies the C++
  pipeline from source on every push.

The README format this script understands:

    "**1144 Python tests** (+17 opt-in skips), **222 C++ tests**"

Stdlib-only — runs in the lint job after the project's own deps are
installed (so ``pytest --collect-only`` can import ``src/``).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
CPP_BUILD = REPO_ROOT / "cpp" / "build"

# Bold around the Python count, parenthesized opt-in skip count, then bold around
# the C++ count. The character class between groups tolerates intervening prose
# (commas, "+", whitespace) but never another `**` so we don't span unrelated
# bold spans.
_README_RE = re.compile(
    r"\*\*(?P<py>\d+)\s+Python tests\*\*"
    r"\s*\(\+(?P<skips>\d+)\s+opt-in skips\)"
    r"[^*]*\*\*(?P<cpp>\d+)\s+C\+\+ tests\*\*"
)
_PYTEST_COLLECT_RE = re.compile(r"(?P<n>\d+)\s+tests?\s+collected\s+in\s+")
_CTEST_TOTAL_RE = re.compile(r"^Total Tests:\s*(?P<n>\d+)", re.MULTILINE)


def parse_readme_counts(readme_text: str) -> tuple[int, int, int]:
    """Return ``(python_passing, python_skipped, cpp)`` from README prose.

    Raises:
        ValueError: README does not contain the expected phrase.
    """
    match = _README_RE.search(readme_text)
    if match is None:
        raise ValueError(
            "README does not contain the expected test-count phrase "
            "'**N Python tests** (+M opt-in skips), **K C++ tests**'; "
            "if the wording changed, update this script's regex."
        )
    return int(match.group("py")), int(match.group("skips")), int(match.group("cpp"))


def collect_pytest_count() -> int:
    """Return the total number of tests collected by ``pytest --collect-only``."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    match = _PYTEST_COLLECT_RE.search(result.stdout)
    if match is None:
        raise RuntimeError(
            f"pytest --collect-only output did not match expected format; got:\n{result.stdout}"
        )
    return int(match.group("n"))


def collect_ctest_count() -> int | None:
    """Return ``Total Tests:`` from ``ctest -N`` or ``None`` if no build dir."""
    if not CPP_BUILD.is_dir():
        return None
    result = subprocess.run(
        ["ctest", "--test-dir", str(CPP_BUILD), "-N"],
        capture_output=True,
        text=True,
        check=True,
    )
    match = _CTEST_TOTAL_RE.search(result.stdout)
    if match is None:
        raise RuntimeError(f"ctest -N output did not match expected format; got:\n{result.stdout}")
    return int(match.group("n"))


def main() -> int:
    py_passing, py_skipped, cpp_readme = parse_readme_counts(README.read_text())

    expected_collected = py_passing + py_skipped
    actual_collected = collect_pytest_count()

    failed = False
    if actual_collected != expected_collected:
        print(
            f"README test-count drift: README says "
            f"{py_passing} passing + {py_skipped} skipped = {expected_collected} "
            f"Python tests, pytest --collect-only reports {actual_collected}. "
            f"Update README.md (or fix the test fixture mismatch).",
            file=sys.stderr,
        )
        failed = True

    actual_cpp = collect_ctest_count()
    if actual_cpp is None:
        print(
            f"NOTE: cpp/build/ not present; skipping C++ test-count check "
            f"(README claims {cpp_readme}). The cpp-build-and-test CI job "
            f"verifies the C++ pipeline from source."
        )
    elif actual_cpp != cpp_readme:
        print(
            f"README test-count drift: README says {cpp_readme} C++ tests, "
            f"ctest -N reports {actual_cpp}. Update README.md.",
            file=sys.stderr,
        )
        failed = True

    if failed:
        return 1

    cpp_msg = f"{actual_cpp} C++" if actual_cpp is not None else f"{cpp_readme} C++ (unverified)"
    print(
        f"OK: README test counts match — "
        f"{py_passing} passing + {py_skipped} skipped Python, {cpp_msg}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
