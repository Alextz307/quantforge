"""Tests for :class:`ModelArtifactManifest` round-trip + framework-version lookup.

Full ``save_model_artifact`` / ``load_model_artifact`` round-trip tests
live in ``test_standalone_training.py`` since they require a fitted
model. Here we cover the pure-logic pieces (manifest dict schema,
ModelKind dispatch, ``_framework_version`` fallback) plus load-path
negative paths that don't need a real model (corrupt manifest, missing
required keys, unknown model_kind).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.core import json_io
from src.core.persistence import (
    MODEL_ARTIFACT_MANIFEST_JSON,
    MODEL_ARTIFACT_WEIGHTS_SUBDIR,
)
from src.core.types import ModelKind
from src.orchestration.model_artifact import (
    ModelArtifactManifest,
    _framework_version,
    build_model_artifact_manifest,
    load_model_artifact,
)

_CREATED_ISO = "2026-04-23T12:00:00"


def _make_manifest(
    *,
    model_kind: ModelKind = ModelKind.PREDICTOR,
) -> ModelArtifactManifest:
    return ModelArtifactManifest(
        name="spy_hybrid_ret_2024q4",
        model_name="hybrid_return",
        model_kind=model_kind,
        created_at=datetime.fromisoformat(_CREATED_ISO),
        git_sha="abc1234",
        seed=42,
        data_hash="deadbeef" * 8,
        quant_engine_version="0.1.0",
    )


class TestModelArtifactManifestRoundTrip:
    def test_to_dict_keys_are_exhaustive(self) -> None:
        d = _make_manifest().to_dict()
        assert set(d.keys()) == {
            "name",
            "model_name",
            "model_kind",
            "created_at",
            "git_sha",
            "seed",
            "data_hash",
            "quant_engine_version",
        }

    def test_created_at_serializes_as_iso_string(self) -> None:
        d = _make_manifest().to_dict()
        assert d["created_at"] == _CREATED_ISO

    def test_model_kind_serializes_as_enum_value(self) -> None:
        d = _make_manifest(model_kind=ModelKind.CLASSIFIER).to_dict()
        assert d["model_kind"] == "classifier"

    def test_roundtrip_preserves_every_field(self) -> None:
        original = _make_manifest()
        reloaded = ModelArtifactManifest.from_dict(original.to_dict())
        assert reloaded == original

    @pytest.mark.parametrize("kind", list(ModelKind))
    def test_roundtrip_every_model_kind(self, kind: ModelKind) -> None:
        original = _make_manifest(model_kind=kind)
        assert ModelArtifactManifest.from_dict(original.to_dict()) == original


class TestFrameworkVersion:
    def test_returns_nonempty_string(self) -> None:
        """Either the installed dist version or the ``'unknown'`` fallback —
        both are non-empty strings. Stronger invariants (e.g. matches
        pyproject.toml) would fail in dev shells without an editable
        install and aren't worth the flake."""
        out = _framework_version()
        assert isinstance(out, str)
        assert out  # non-empty — ``unknown`` is 7 chars, a real version is also non-empty


class TestBuildHelper:
    def test_fills_created_at_and_version(self) -> None:
        m = build_model_artifact_manifest(
            name="n",
            model_name="hybrid_return",
            model_kind=ModelKind.PREDICTOR,
            git_sha="sha",
            seed=1,
            data_hash="h",
        )
        assert m.name == "n"
        assert m.model_name == "hybrid_return"
        assert m.model_kind == ModelKind.PREDICTOR
        assert m.quant_engine_version  # auto-filled from package metadata
        assert isinstance(m.created_at, datetime)


def _valid_manifest_dict() -> dict[str, object]:
    return {
        "name": "n",
        "model_name": "hybrid_return",
        "model_kind": "predictor",
        "created_at": _CREATED_ISO,
        "git_sha": "sha",
        "seed": 1,
        "data_hash": "h",
        "quant_engine_version": "0.1.0",
    }


class TestLoadModelArtifactNegativePaths:
    """``load_model_artifact`` must fail loudly (not silently dispatch
    to the wrong registry or return a half-constructed object) when the
    manifest on disk is corrupt or refers to something unreachable.
    """

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        # Empty directory → manifest.json doesn't exist
        with pytest.raises(FileNotFoundError):
            load_model_artifact(tmp_path)

    def test_corrupt_manifest_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / MODEL_ARTIFACT_MANIFEST_JSON).write_text("{not-valid-json")
        with pytest.raises(json.JSONDecodeError):
            load_model_artifact(tmp_path)

    def test_manifest_missing_required_key_raises(self, tmp_path: Path) -> None:
        """A manifest missing any required field must fail at ``from_dict``
        with a pointed KeyError rather than a silent None field.
        """
        d = _valid_manifest_dict()
        del d["git_sha"]
        json_io.write(tmp_path / MODEL_ARTIFACT_MANIFEST_JSON, d)
        with pytest.raises(KeyError, match="git_sha"):
            load_model_artifact(tmp_path)

    def test_unknown_model_kind_raises(self, tmp_path: Path) -> None:
        """Typo or future-version ``model_kind`` must raise, not silently
        fall through to a wrong registry.
        """
        d = _valid_manifest_dict()
        d["model_kind"] = "not-a-real-kind"
        json_io.write(tmp_path / MODEL_ARTIFACT_MANIFEST_JSON, d)
        with pytest.raises(ValueError):
            load_model_artifact(tmp_path)

    def test_unknown_model_name_for_kind_raises(self, tmp_path: Path) -> None:
        """A valid model_kind but a nonexistent model_name in that kind's
        registry fails cleanly at dispatch, not deep inside load().
        """
        d = _valid_manifest_dict()
        d["model_name"] = "phantom_model"
        json_io.write(tmp_path / MODEL_ARTIFACT_MANIFEST_JSON, d)
        with pytest.raises(KeyError):
            load_model_artifact(tmp_path)

    def test_missing_weights_subdir_raises(self, tmp_path: Path) -> None:
        """Manifest parses fine but the weights/ subdir is absent — the
        per-model ``load()`` is expected to fail with a filesystem error,
        not silently return an unfitted instance.
        """
        d = _valid_manifest_dict()
        json_io.write(tmp_path / MODEL_ARTIFACT_MANIFEST_JSON, d)
        # weights subdir deliberately not created
        assert not (tmp_path / MODEL_ARTIFACT_WEIGHTS_SUBDIR).exists()
        with pytest.raises((FileNotFoundError, OSError)):
            load_model_artifact(tmp_path)
