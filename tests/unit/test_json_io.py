"""
Unit tests for src/core/json_io.py - generic JSON read/write + typed
field-extraction helpers. This module has no business-logic dependencies;
these tests only exercise the IO + narrowing contract.
"""

from __future__ import annotations

import json
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


class TestDiffAgainstSnapshot:
    _SAMPLE: dict[str, object] = {"a": 1, "b": [1, 2, 3]}

    def test_returns_no_errors_when_snapshot_matches(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "snap.json"
        json_io.write(snapshot, self._SAMPLE)
        assert (
            json_io.diff_against_snapshot(
                self._SAMPLE, snapshot, label="test", fix_command="make regen"
            )
            == []
        )

    def test_reports_drift_when_snapshot_is_stale(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "snap.json"
        json_io.write(snapshot, self._SAMPLE)
        mutated: dict[str, object] = {**self._SAMPLE, "a": 2}
        errors = json_io.diff_against_snapshot(
            mutated, snapshot, label="OpenAPI snapshot", fix_command="make webapp-openapi-snapshot"
        )
        assert errors
        assert "OpenAPI snapshot" in errors[0]
        assert "stale" in errors[0]
        assert "make webapp-openapi-snapshot" in errors[1]

    def test_reports_missing_snapshot(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.json"
        errors = json_io.diff_against_snapshot(
            self._SAMPLE, missing, label="Schema-mirror snapshot", fix_command="--write"
        )
        assert errors
        assert "Schema-mirror snapshot" in errors[0]
        assert "missing" in errors[0]
        assert "--write" in errors[1]


class TestReadJsonl:
    def test_round_trip_records(self, tmp_path: Path) -> None:
        path = tmp_path / "records.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
        assert json_io.read_jsonl(path) == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "blanks.jsonl"
        path.write_text('{"a": 1}\n\n  \n{"a": 2}\n', encoding="utf-8")
        assert json_io.read_jsonl(path) == [{"a": 1}, {"a": 2}]

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert json_io.read_jsonl(path) == []

    def test_rejects_non_object_line_with_lineno(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"a": 1}\n[1, 2]\n', encoding="utf-8")
        with pytest.raises(ValueError, match="line 2 must be an object"):
            json_io.read_jsonl(path)

    def test_tolerates_partial_trailing_record(self, tmp_path: Path) -> None:
        """
        ``append_jsonl`` can be interrupted mid-write, leaving the last
        record truncated. ``read_jsonl`` drops that record instead of raising
        so future consumers (post-hoc readers of the HPO trial log) can
        recover earlier records.
        """

        path = tmp_path / "interrupted.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b": "tru', encoding="utf-8")
        assert json_io.read_jsonl(path) == [{"a": 1}, {"a": 2}]

    def test_partial_record_in_middle_still_raises(self, tmp_path: Path) -> None:
        """
        Only the FINAL line is tolerated as crash-truncated; mid-file
        corruption is real data damage and must surface.
        """

        path = tmp_path / "corrupt.jsonl"
        path.write_text('{"a": 1}\n{"a": 2, "b": "tru\n{"a": 3}\n', encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            json_io.read_jsonl(path)


class TestWriteJsonl:
    def test_round_trip_via_read_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        records = [{"a": 1}, {"a": 2}, {"a": 3}]
        json_io.write_jsonl(path, records)
        assert json_io.read_jsonl(path) == records

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        json_io.write_jsonl(path, [{"a": 1}])
        json_io.write_jsonl(path, [{"a": 2}])
        assert json_io.read_jsonl(path) == [{"a": 2}]

    def test_accepts_generator(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        json_io.write_jsonl(path, ({"i": i} for i in range(3)))
        assert json_io.read_jsonl(path) == [{"i": 0}, {"i": 1}, {"i": 2}]

    def test_sorted_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "sorted.jsonl"
        json_io.write_jsonl(path, [{"b": 2, "a": 1}])
        assert path.read_text(encoding="utf-8") == '{"a": 1, "b": 2}\n'


class TestAppendJsonl:
    def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        json_io.append_jsonl(path, {"trial": 0})
        assert json_io.read_jsonl(path) == [{"trial": 0}]

    def test_appends_to_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        json_io.append_jsonl(path, {"trial": 0})
        json_io.append_jsonl(path, {"trial": 1})
        json_io.append_jsonl(path, {"trial": 2})
        assert json_io.read_jsonl(path) == [{"trial": 0}, {"trial": 1}, {"trial": 2}]


class TestGetScalars:
    """
    Single sample payload feeds every happy + error-path test below.
    """

    SAMPLE: dict[str, object] = {
        "an_int": 7,
        "a_float": 1.5,
        "a_str": "hello",
        "a_list": [1, 2, 3],
        "str_list": ["a", "b"],
        # Mixed int+float (should coerce); bool is an int subclass that the
        # narrowing helpers must reject when an int/float is required.
        "float_list": [0.1, 2, 3.5],
        "a_bool": True,
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
        # bool is an int subclass but the helper must still reject ints -
        # otherwise an integer-typed JSON value silently coerces to True.
        with pytest.raises(ValueError, match="must be a bool"):
            json_io.get_bool(self.SAMPLE, "an_int")

    def test_get_bool_missing_key(self) -> None:
        with pytest.raises(KeyError, match="missing required"):
            json_io.get_bool(self.SAMPLE, "nonexistent")


class TestGetLists:
    def test_rejects_non_list_via_typed_wrapper(self) -> None:
        # Every typed list helper routes through the same ``_get_list`` guard.
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
