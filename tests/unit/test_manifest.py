"""
Tests for :class:`Manifest` round-trip and :func:`write_experiment_manifest`.

Catches the two classes of failure a typed manifest prevents:

* silent typos in field names (``holdoutStart`` vs ``holdout_start``) —
  verified by asserting ``from_dict`` rejects wrong-type / missing keys.
* drift between timestamp round-trip format and what consumers expect —
  verified by an exact-string check on ``holdout_start`` ISO output.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.core.persistence import EXPERIMENT_MANIFEST_JSON, write_experiment_manifest
from src.engine.scenarios import SlippageScenario
from src.orchestration.manifest import Manifest

_EXPERIMENT_ID = "20260422_AdaptiveBollinger_abc1234_zz9999"
_NAME = "manifest_test"
_CREATED_AT = datetime(2026, 4, 22, 14, 30, 0)
_GIT_SHA = "abc1234"
_SEED = 42
_DATA_HASH = "deadbeef" * 8
_HOLDOUT_ISO = "2023-06-30T00:00:00"


def _make(holdout: pd.Timestamp | None = pd.Timestamp(_HOLDOUT_ISO)) -> Manifest:
    return Manifest(
        experiment_id=_EXPERIMENT_ID,
        name=_NAME,
        created_at=_CREATED_AT,
        git_sha=_GIT_SHA,
        seed=_SEED,
        data_hash=_DATA_HASH,
        slippage_scenario=SlippageScenario.NORMAL,
        holdout_start=holdout,
    )


class TestManifestRoundTrip:
    def test_to_dict_keys_are_exhaustive(self) -> None:
        d = _make().to_dict()
        expected_keys = {
            "experiment_id",
            "name",
            "created_at",
            "git_sha",
            "seed",
            "data_hash",
            "slippage_scenario",
            "holdout_start",
        }
        assert set(d.keys()) == expected_keys

    def test_holdout_start_serializes_as_iso_string(self) -> None:
        d = _make().to_dict()
        assert d["holdout_start"] == _HOLDOUT_ISO

    def test_slippage_scenario_serializes_as_enum_value(self) -> None:
        d = _make().to_dict()
        assert d["slippage_scenario"] == SlippageScenario.NORMAL.value

    def test_roundtrip_preserves_every_field(self) -> None:
        original = _make()
        revived = Manifest.from_dict(original.to_dict())
        assert revived == original

    def test_holdout_none_roundtrip(self) -> None:
        original = _make(holdout=None)
        d = original.to_dict()
        assert d["holdout_start"] is None
        assert Manifest.from_dict(d) == original

    def test_from_dict_rejects_missing_field(self) -> None:
        d = _make().to_dict()
        del d["git_sha"]
        with pytest.raises(KeyError, match="git_sha"):
            Manifest.from_dict(d)

    def test_from_dict_rejects_holdout_start_of_wrong_type(self) -> None:
        d = _make().to_dict()
        d["holdout_start"] = 12345  # int, not ISO string
        with pytest.raises(ValueError, match="holdout_start"):
            Manifest.from_dict(d)


class TestWriteExperimentManifest:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        write_experiment_manifest(tmp_path, _make())
        p = tmp_path / EXPERIMENT_MANIFEST_JSON
        assert p.is_file()
        assert '"experiment_id"' in p.read_text()

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_dir"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            write_experiment_manifest(missing, _make())
