"""Guard against Python / C++ numeric-constant drift.

Several scalar constants are mirrored between ``src/core/constants.py`` and
``cpp/include/quant/core/types.hpp`` (trading-calendar counts, position
limits). They have to agree numerically — a silent divergence between the
Python annualization factor and the C++ engine's would produce wrong Sharpes
or volatility numbers in every report without ever raising.

This script parses both files (text-only, stdlib-only) and flags any pair
where the values differ or one side is missing. Run locally with
``python scripts/check_constants_sync.py``; wired into CI's lint job so a
forgotten mirror update fails the same PR.

When adding a new constant that exists on both sides of the bridge, append
its name pair to ``MIRROR_PAIRS`` below.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY_CONSTANTS = REPO_ROOT / "src" / "core" / "constants.py"
CPP_TYPES = REPO_ROOT / "cpp" / "include" / "quant" / "core" / "types.hpp"

MIRROR_PAIRS: list[tuple[str, str]] = [
    ("TRADING_DAYS_PER_YEAR", "kTradingDaysPerYear"),
    ("TRADING_WEEKS_PER_YEAR", "kTradingWeeksPerYear"),
    ("US_TRADING_MINUTES_PER_DAY", "kUSMinutesPerDay"),
    ("US_TRADING_SECONDS_PER_DAY", "kUSSecondsPerDay"),
    ("MAX_LEVERAGE", "kMaxLeverage"),
    ("MIN_POSITION", "kMinPosition"),
    ("MAX_POSITION", "kMaxPosition"),
]

_PY_ASSIGN_RE = re.compile(
    r"^(?P<name>[A-Z][A-Z0-9_]*)\s*(?::[^=\n]+)?\s*=\s*(?P<value>[^\n#]+?)\s*(?:#.*)?$",
    re.MULTILINE,
)

# Multi-word C++ types like `unsigned int` need a non-greedy run of words.
_CPP_DECL_RE = re.compile(
    r"inline\s+constexpr\s+(?:[\w:]+\s+)+?(?P<name>\w+)\s*=\s*(?P<value>[^;]+);"
)


def parse_python_constants(text: str) -> dict[str, str]:
    """Return ``{name: raw_value_text}`` for top-level UPPER_SNAKE_CASE assignments."""
    return {m.group("name"): m.group("value").strip() for m in _PY_ASSIGN_RE.finditer(text)}


def parse_cpp_constants(text: str) -> dict[str, str]:
    """Return ``{name: raw_value_text}`` for ``inline constexpr`` declarations."""
    return {m.group("name"): m.group("value").strip() for m in _CPP_DECL_RE.finditer(text)}


def _normalize(value: str) -> str:
    """Compare-friendly form: drop digit-grouping underscores, trim whitespace."""
    return value.replace("_", "").strip()


def find_mismatches(py_text: str, cpp_text: str) -> list[str]:
    """Return a sorted list of human-readable mismatch messages (empty = OK)."""
    py = parse_python_constants(py_text)
    cpp = parse_cpp_constants(cpp_text)

    errors: list[str] = []
    for py_name, cpp_name in MIRROR_PAIRS:
        py_val = py.get(py_name)
        cpp_val = cpp.get(cpp_name)
        if py_val is None:
            errors.append(
                f"missing Python constant `{py_name}` (expected mirror of C++ `{cpp_name}`)"
            )
            continue
        if cpp_val is None:
            errors.append(
                f"missing C++ constant `{cpp_name}` (expected mirror of Python `{py_name}`)"
            )
            continue
        if _normalize(py_val) != _normalize(cpp_val):
            errors.append(f"drift: Python `{py_name}={py_val}` vs C++ `{cpp_name}={cpp_val}`")
    return sorted(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors = find_mismatches(PY_CONSTANTS.read_text(), CPP_TYPES.read_text())
    if errors:
        sys.stderr.write("Python/C++ constant drift detected:\n")
        for line in errors:
            sys.stderr.write(f"  - {line}\n")
        sys.stderr.write("\nFix: update one side of the mirror so both files agree, then commit.\n")
        return 1
    sys.stdout.write(f"Constants in sync ({len(MIRROR_PAIRS)} pairs checked).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
