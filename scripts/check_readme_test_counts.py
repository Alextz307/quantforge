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

    "**<N> Python tests** (+<M> opt-in skips), **<K> C++ tests**"

Pass ``--fix`` to rewrite the README counts in place from the live numbers
instead of just reporting drift. The C++ count is only rewritten when
``cpp/build/`` is present.

Stdlib-only — runs in the lint job after the project's own deps are
installed (so ``pytest --collect-only`` can import ``src/``).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
CPP_BUILD = REPO_ROOT / "cpp" / "build"

# Each bold half declared once. The combined `_README_RE` is used by the
# read path; `_PY_PHRASE_RE` / `_CPP_PHRASE_RE` are used by the `--fix`
# rewrite path. The `[^*]*` glue between halves tolerates intervening
# prose ("CI is green ... with **N Python tests** (+M opt-in skips), **K
# C++ tests**, ruff clean") but never another `**`.
_PY_PHRASE_PATTERN = r"\*\*(?P<py>\d+)\s+Python tests\*\*\s*\(\+(?P<skips>\d+)\s+opt-in skips\)"
_CPP_PHRASE_PATTERN = r"\*\*(?P<cpp>\d+)\s+C\+\+ tests\*\*"
_README_RE = re.compile(_PY_PHRASE_PATTERN + r"[^*]*" + _CPP_PHRASE_PATTERN)
_PY_PHRASE_RE = re.compile(_PY_PHRASE_PATTERN)
_CPP_PHRASE_RE = re.compile(_CPP_PHRASE_PATTERN)
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


_PYTEST_SUMMARY_RE = re.compile(r"(?P<passed>\d+) passed(?:, (?P<skipped>\d+) skipped)?")


def run_pytest_passed_skipped() -> tuple[int, int]:
    """Return ``(passed, skipped)`` by actually running the suite.

    Needed by ``--fix`` because ``--collect-only`` reports the combined
    total but cannot split the bold "(+M opt-in skips)" half — the
    skipif predicates that gate opt-in tests are only evaluated at run
    time.

    Refuses to return on a failing suite: if the README rewrite
    proceeded after a partial run, we'd quietly bury the failures
    behind freshly-correct numbers. The drift guard's contract is
    "README equals truth", and silent partial truth is worse than
    visible drift.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pytest exited {result.returncode}; refusing to rewrite README from a "
            f"failing run.\n--- pytest tail ---\n{result.stdout[-1000:]}"
        )
    match = _PYTEST_SUMMARY_RE.search(result.stdout)
    if match is None:
        raise RuntimeError(
            f"pytest summary did not match expected format; got:\n{result.stdout[-500:]}"
        )
    return int(match.group("passed")), int(match.group("skipped") or 0)


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


def rewrite_readme_counts(text: str, py_passed: int, py_skipped: int, cpp: int | None) -> str:
    """Return ``text`` with the bold count phrases substituted in place."""
    new = _PY_PHRASE_RE.sub(f"**{py_passed} Python tests** (+{py_skipped} opt-in skips)", text)
    if cpp is not None:
        new = _CPP_PHRASE_RE.sub(f"**{cpp} C++ tests**", new)
    return new


def _check(readme_text: str) -> int:
    py_passing, py_skipped, cpp_readme = parse_readme_counts(readme_text)
    expected_collected = py_passing + py_skipped
    actual_collected = collect_pytest_count()

    failed = False
    if actual_collected != expected_collected:
        print(
            f"README test-count drift: README says "
            f"{py_passing} passing + {py_skipped} skipped = {expected_collected} "
            f"Python tests, pytest --collect-only reports {actual_collected}. "
            f"Run `python scripts/check_readme_test_counts.py --fix` to update the README.",
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
            f"ctest -N reports {actual_cpp}. "
            f"Run `python scripts/check_readme_test_counts.py --fix` to update the README.",
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


def _fix(readme_text: str) -> int:
    py_readme, skips_readme, cpp_readme = parse_readme_counts(readme_text)
    cpp = collect_ctest_count()

    # Cheap check first: only run the full suite when the Python totals
    # actually differ. Re-running a 30-40s pytest pass to confirm a clean
    # README is the wrong default for a `--fix` developer keeps reaching for.
    python_in_sync = collect_pytest_count() == py_readme + skips_readme
    if python_in_sync:
        py_passed, py_skipped = py_readme, skips_readme
    else:
        py_passed, py_skipped = run_pytest_passed_skipped()

    new_text = rewrite_readme_counts(readme_text, py_passed, py_skipped, cpp)
    if new_text == readme_text:
        print("OK: README already in sync, no rewrite needed.")
        return 0
    README.write_text(new_text)
    cpp_msg = f"{cpp} C++" if cpp is not None else "C++ count untouched (cpp/build/ absent)"
    print(f"Updated README: {py_passed} passing + {py_skipped} skipped Python, {cpp_msg}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Rewrite README counts in place from the live numbers (runs the full "
            "pytest suite to split passed vs skipped). Default mode is read-only "
            "drift detection."
        ),
    )
    args = parser.parse_args(argv)
    text = README.read_text()
    return _fix(text) if args.fix else _check(text)


if __name__ == "__main__":
    raise SystemExit(main())
