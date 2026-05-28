"""
Tests for the Python / C++ constants drift guard.
"""

from __future__ import annotations

import pytest

from tests.conftest import REPO_ROOT, load_script_module

GUARD_SCRIPT = REPO_ROOT / "scripts" / "check_constants_sync.py"
PY_CONSTANTS = REPO_ROOT / "src" / "core" / "constants.py"
CPP_TYPES = REPO_ROOT / "cpp" / "include" / "quant" / "core" / "types.hpp"

guard = load_script_module(GUARD_SCRIPT, "check_constants_sync")

_PY_OK = """\
TRADING_DAYS_PER_YEAR: int = 252
TRADING_WEEKS_PER_YEAR: int = 52
US_TRADING_MINUTES_PER_DAY: int = 390
US_TRADING_SECONDS_PER_DAY: int = 23_400
MAX_LEVERAGE: float = 3.0
MIN_POSITION: float = -1.0
MAX_POSITION: float = 3.0
"""

_CPP_OK = """\
inline constexpr int kTradingDaysPerYear = 252;
inline constexpr int kTradingWeeksPerYear = 52;
inline constexpr int kUSMinutesPerDay = 390;
inline constexpr int kUSSecondsPerDay = 23400;
inline constexpr double kMaxLeverage = 3.0;
inline constexpr double kMinPosition = -1.0;
inline constexpr double kMaxPosition = 3.0;
"""


class TestFindMismatches:
    def test_no_drift_when_all_agree(self) -> None:
        assert guard.find_mismatches(_PY_OK, _CPP_OK) == []

    def test_underscore_grouping_ignored(self) -> None:
        py = _PY_OK.replace("23_400", "23_400")
        cpp = _CPP_OK.replace("23400", "23400")
        assert guard.find_mismatches(py, cpp) == []

    def test_reports_value_drift(self) -> None:
        cpp_drifted = _CPP_OK.replace("kTradingDaysPerYear = 252", "kTradingDaysPerYear = 250")
        errors = guard.find_mismatches(_PY_OK, cpp_drifted)
        assert len(errors) == 1
        assert "TRADING_DAYS_PER_YEAR=252" in errors[0]
        assert "kTradingDaysPerYear=250" in errors[0]

    def test_reports_missing_python_side(self) -> None:
        py_missing = _PY_OK.replace("MAX_LEVERAGE: float = 3.0\n", "")
        errors = guard.find_mismatches(py_missing, _CPP_OK)
        assert any("missing Python constant `MAX_LEVERAGE`" in e for e in errors)

    def test_reports_missing_cpp_side(self) -> None:
        cpp_missing = _CPP_OK.replace("inline constexpr double kMinPosition = -1.0;\n", "")
        errors = guard.find_mismatches(_PY_OK, cpp_missing)
        assert any("missing C++ constant `kMinPosition`" in e for e in errors)

    def test_errors_are_sorted(self) -> None:
        cpp_drifted = _CPP_OK.replace("kMaxLeverage = 3.0", "kMaxLeverage = 4.0").replace(
            "kTradingDaysPerYear = 252", "kTradingDaysPerYear = 250"
        )
        errors = guard.find_mismatches(_PY_OK, cpp_drifted)
        assert errors == sorted(errors)


class TestParsers:
    def test_python_parser_picks_up_typed_assigns(self) -> None:
        parsed = guard.parse_python_constants(_PY_OK)
        assert parsed["TRADING_DAYS_PER_YEAR"] == "252"
        assert parsed["MAX_LEVERAGE"] == "3.0"

    def test_python_parser_ignores_lower_case(self) -> None:
        parsed = guard.parse_python_constants("foo = 42\nBAR = 7\n")
        assert "foo" not in parsed
        assert parsed["BAR"] == "7"

    def test_cpp_parser_picks_up_inline_constexpr(self) -> None:
        parsed = guard.parse_cpp_constants(_CPP_OK)
        assert parsed["kTradingDaysPerYear"] == "252"
        assert parsed["kMaxLeverage"] == "3.0"

    def test_cpp_parser_ignores_non_constexpr(self) -> None:
        parsed = guard.parse_cpp_constants("constexpr int kFoo = 1;\nint kBar = 2;\n")
        assert "kFoo" not in parsed
        assert "kBar" not in parsed


class TestRepoStateIsClean:
    """
    End-to-end: the real repo's mirrored constants must agree.
    """

    def test_real_repo_constants_in_sync(self) -> None:
        errors = guard.find_mismatches(PY_CONSTANTS.read_text(), CPP_TYPES.read_text())
        assert errors == [], "Python/C++ constant drift:\n  " + "\n  ".join(errors)


def test_script_exposes_expected_public_surface() -> None:
    assert hasattr(guard, "find_mismatches")
    assert hasattr(guard, "parse_python_constants")
    assert hasattr(guard, "parse_cpp_constants")
    assert hasattr(guard, "MIRROR_PAIRS")
    assert isinstance(guard.MIRROR_PAIRS, list) and guard.MIRROR_PAIRS


def test_main_returns_zero_on_clean_repo(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["check_constants_sync"])
    assert guard.main() == 0
    captured = capsys.readouterr()
    assert "in sync" in captured.out
