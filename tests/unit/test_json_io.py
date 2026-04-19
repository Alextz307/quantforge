"""Unit tests for src/core/json_io.py — generic JSON read/write + typed
field-extraction helpers. This module has no business-logic dependencies;
these tests only exercise the IO + narrowing contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core import json_io


class TestWriteRead:
    def test_round_trip_dict(self, tmp_path: Path) -> None:
        obj: dict[str, object] = {"a": 1, "b": [1.5, 2.5], "c": {"nested": True}}
        p = tmp_path / "out.json"
        json_io.write(p, obj)
        loaded = json_io.read(p)
        assert loaded == obj

    def test_round_trip_empty_list(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        json_io.write(p, [])
        assert json_io.read(p) == []

    def test_read_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            json_io.read(tmp_path / "nope.json")


class TestReadDict:
    def test_rejects_top_level_list(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        json_io.write(path, [1, 2, 3])
        with pytest.raises(ValueError, match="must be an object"):
            json_io.read_dict(path)

    def test_accepts_object(self, tmp_path: Path) -> None:
        path = tmp_path / "obj.json"
        json_io.write(path, {"a": 1})
        assert json_io.read_dict(path) == {"a": 1}


class TestGetScalars:
    # Sample payload containing one of each expected and one wrong-type value
    # per field. Tests below pull from this single source of truth so the
    # happy-path + error-path wiring is obvious.
    SAMPLE: dict[str, object] = {
        "an_int": 7,
        "a_float": 1.5,
        "a_str": "hello",
        "a_list": [1, 2, 3],
        "str_list": ["a", "b"],
        "float_list": [0.1, 2, 3.5],  # mixed int+float, should coerce
        "a_bool": True,  # bool is int subclass — helpers must reject
    }

    def test_get_int_happy(self) -> None:
        assert json_io.get_int(self.SAMPLE, "an_int") == 7

    def test_get_int_rejects_bool(self) -> None:
        with pytest.raises(ValueError, match="must be an int"):
            json_io.get_int(self.SAMPLE, "a_bool")

    def test_get_int_rejects_float(self) -> None:
        with pytest.raises(ValueError, match="must be an int"):
            json_io.get_int(self.SAMPLE, "a_float")

    def test_get_int_missing_key(self) -> None:
        with pytest.raises(KeyError, match="missing required"):
            json_io.get_int(self.SAMPLE, "nonexistent")

    def test_get_float_happy(self) -> None:
        assert json_io.get_float(self.SAMPLE, "a_float") == 1.5

    def test_get_float_accepts_int(self) -> None:
        # Integers are valid JSON numbers; we coerce to float.
        assert json_io.get_float(self.SAMPLE, "an_int") == 7.0

    def test_get_float_rejects_bool(self) -> None:
        with pytest.raises(ValueError, match="must be a number"):
            json_io.get_float(self.SAMPLE, "a_bool")

    def test_get_float_rejects_str(self) -> None:
        with pytest.raises(ValueError, match="must be a number"):
            json_io.get_float(self.SAMPLE, "a_str")

    def test_get_str_happy(self) -> None:
        assert json_io.get_str(self.SAMPLE, "a_str") == "hello"

    def test_get_str_rejects_int(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            json_io.get_str(self.SAMPLE, "an_int")

    def test_get_bool_happy(self) -> None:
        assert json_io.get_bool(self.SAMPLE, "a_bool") is True

    def test_get_bool_rejects_int(self) -> None:
        # bool is an int subclass but int is NOT a bool subclass — the helper
        # must reject integer-typed JSON values even when they'd coerce to True.
        with pytest.raises(ValueError, match="must be a bool"):
            json_io.get_bool(self.SAMPLE, "an_int")

    def test_get_bool_missing_key(self) -> None:
        with pytest.raises(KeyError, match="missing required"):
            json_io.get_bool(self.SAMPLE, "nonexistent")


class TestGetLists:
    def test_rejects_non_list_via_typed_wrapper(self) -> None:
        # Every typed list helper (int/float/str) routes through the same
        # ``_get_list`` guard, so we exercise it via one of them.
        with pytest.raises(ValueError, match="must be a list"):
            json_io.get_float_list({"x": 7}, "x")

    def test_get_float_list_happy(self) -> None:
        assert json_io.get_float_list({"x": [0.1, 2, 3.5]}, "x") == [0.1, 2.0, 3.5]

    def test_get_float_list_rejects_string_item(self) -> None:
        with pytest.raises(ValueError, match=r"'x'\[1\] must be a number"):
            json_io.get_float_list({"x": [0.1, "nope", 3.5]}, "x")

    def test_get_float_list_rejects_bool_item(self) -> None:
        with pytest.raises(ValueError, match=r"'x'\[0\] must be a number"):
            json_io.get_float_list({"x": [True, 1.0]}, "x")

    def test_get_str_list_happy(self) -> None:
        assert json_io.get_str_list({"x": ["a", "b"]}, "x") == ["a", "b"]

    def test_get_str_list_rejects_int_item(self) -> None:
        with pytest.raises(ValueError, match=r"'x'\[1\] must be a string"):
            json_io.get_str_list({"x": ["a", 2]}, "x")

    def test_get_int_list_happy(self) -> None:
        assert json_io.get_int_list({"x": [1, 2, 3]}, "x") == [1, 2, 3]

    def test_get_int_list_rejects_float_item(self) -> None:
        with pytest.raises(ValueError, match=r"'x'\[1\] must be an int"):
            json_io.get_int_list({"x": [1, 2.5, 3]}, "x")

    def test_get_int_list_rejects_bool_item(self) -> None:
        with pytest.raises(ValueError, match=r"'x'\[0\] must be an int"):
            json_io.get_int_list({"x": [True, 1]}, "x")
