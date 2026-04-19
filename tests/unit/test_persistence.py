"""Unit tests for src/core/persistence.py helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.core.persistence import (
    ensure_model_dir,
    json_get_float,
    json_get_float_list,
    json_get_int,
    json_get_int_list,
    json_get_str,
    json_get_str_list,
    load_standard_scaler,
    read_json,
    read_json_dict,
    save_standard_scaler,
    write_json,
)

# Fixture scale constants — kept small; the helpers are pure IO and don't need
# large inputs to exercise the round-trip.
N_SAMPLES = 20
N_FEATURES = 3
SCALER_MEAN_SEED = 42
ROUND_TRIP_ATOL = 0.0


class TestWriteReadJson:
    def test_round_trip_dict(self, tmp_path: Path) -> None:
        obj: dict[str, object] = {"a": 1, "b": [1.5, 2.5], "c": {"nested": True}}
        p = tmp_path / "out.json"
        write_json(p, obj)
        loaded = read_json(p)
        assert loaded == obj

    def test_round_trip_empty_list(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        write_json(p, [])
        assert read_json(p) == []

    def test_read_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_json(tmp_path / "nope.json")


class TestEnsureModelDir:
    def test_creates_missing_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "fresh"
        result = ensure_model_dir(target)
        assert target.is_dir()
        assert result == target

    def test_reuses_empty_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "empty"
        target.mkdir()
        # Must not raise; empty existing dir is a valid save target.
        ensure_model_dir(target)

    def test_raises_on_non_empty_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "populated"
        target.mkdir()
        (target / "junk.txt").write_text("junk")
        with pytest.raises(FileExistsError, match="non-empty"):
            ensure_model_dir(target)

    def test_raises_on_file(self, tmp_path: Path) -> None:
        target = tmp_path / "file"
        target.write_text("not a dir")
        with pytest.raises(NotADirectoryError):
            ensure_model_dir(target)


class TestStandardScalerRoundTrip:
    def test_fitted_scaler_round_trips(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SCALER_MEAN_SEED)
        data = rng.normal(0.0, 1.0, size=(N_SAMPLES, N_FEATURES))
        scaler = StandardScaler()
        scaler.fit(data)

        path = tmp_path / "scaler.json"
        save_standard_scaler(scaler, path)
        loaded = load_standard_scaler(path)

        assert loaded.n_features_in_ == N_FEATURES
        np.testing.assert_array_equal(loaded.mean_, scaler.mean_)
        np.testing.assert_array_equal(loaded.scale_, scaler.scale_)
        np.testing.assert_array_equal(loaded.var_, scaler.var_)
        # The decisive round-trip check: transform output must match.
        np.testing.assert_allclose(
            loaded.transform(data),
            scaler.transform(data),
            atol=ROUND_TRIP_ATOL,
            rtol=0.0,
        )

    def test_unfitted_scaler_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="unfitted"):
            save_standard_scaler(StandardScaler(), tmp_path / "never.json")


class TestReadJsonDict:
    def test_rejects_top_level_list(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        write_json(path, [1, 2, 3])
        with pytest.raises(ValueError, match="must be an object"):
            read_json_dict(path)

    def test_accepts_object(self, tmp_path: Path) -> None:
        path = tmp_path / "obj.json"
        write_json(path, {"a": 1})
        assert read_json_dict(path) == {"a": 1}


class TestJsonGetScalars:
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
        assert json_get_int(self.SAMPLE, "an_int") == 7

    def test_get_int_rejects_bool(self) -> None:
        with pytest.raises(ValueError, match="must be an int"):
            json_get_int(self.SAMPLE, "a_bool")

    def test_get_int_rejects_float(self) -> None:
        with pytest.raises(ValueError, match="must be an int"):
            json_get_int(self.SAMPLE, "a_float")

    def test_get_int_missing_key(self) -> None:
        with pytest.raises(KeyError, match="missing required"):
            json_get_int(self.SAMPLE, "nonexistent")

    def test_get_float_happy(self) -> None:
        assert json_get_float(self.SAMPLE, "a_float") == 1.5

    def test_get_float_accepts_int(self) -> None:
        # Integers are valid JSON numbers; we coerce to float.
        assert json_get_float(self.SAMPLE, "an_int") == 7.0

    def test_get_float_rejects_bool(self) -> None:
        with pytest.raises(ValueError, match="must be a number"):
            json_get_float(self.SAMPLE, "a_bool")

    def test_get_float_rejects_str(self) -> None:
        with pytest.raises(ValueError, match="must be a number"):
            json_get_float(self.SAMPLE, "a_str")

    def test_get_str_happy(self) -> None:
        assert json_get_str(self.SAMPLE, "a_str") == "hello"

    def test_get_str_rejects_int(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            json_get_str(self.SAMPLE, "an_int")


class TestJsonGetLists:
    def test_rejects_non_list_via_typed_wrapper(self) -> None:
        # Every typed list helper (int/float/str) routes through the same
        # ``_json_get_list`` guard, so we exercise it via one of them.
        with pytest.raises(ValueError, match="must be a list"):
            json_get_float_list({"x": 7}, "x")

    def test_get_float_list_happy(self) -> None:
        assert json_get_float_list({"x": [0.1, 2, 3.5]}, "x") == [0.1, 2.0, 3.5]

    def test_get_float_list_rejects_string_item(self) -> None:
        with pytest.raises(ValueError, match=r"'x'\[1\] must be a number"):
            json_get_float_list({"x": [0.1, "nope", 3.5]}, "x")

    def test_get_float_list_rejects_bool_item(self) -> None:
        with pytest.raises(ValueError, match=r"'x'\[0\] must be a number"):
            json_get_float_list({"x": [True, 1.0]}, "x")

    def test_get_str_list_happy(self) -> None:
        assert json_get_str_list({"x": ["a", "b"]}, "x") == ["a", "b"]

    def test_get_str_list_rejects_int_item(self) -> None:
        with pytest.raises(ValueError, match=r"'x'\[1\] must be a string"):
            json_get_str_list({"x": ["a", 2]}, "x")

    def test_get_int_list_happy(self) -> None:
        assert json_get_int_list({"x": [1, 2, 3]}, "x") == [1, 2, 3]

    def test_get_int_list_rejects_float_item(self) -> None:
        with pytest.raises(ValueError, match=r"'x'\[1\] must be an int"):
            json_get_int_list({"x": [1, 2.5, 3]}, "x")

    def test_get_int_list_rejects_bool_item(self) -> None:
        with pytest.raises(ValueError, match=r"'x'\[0\] must be an int"):
            json_get_int_list({"x": [True, 1]}, "x")
