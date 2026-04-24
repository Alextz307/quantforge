"""Standalone-model artifact format: manifest + save / load.

Complements the experiment-run ``Manifest`` (which records a full
walk-forward run) with a sibling type for a single standalone-trained
model. The two artifact types share filename conventions
(``manifest.json``, ``config.yaml``) but not schema — an experiment
manifest carries slippage scenario + holdout boundary; a model manifest
carries the model's registry name + kind.

On-disk layout under ``experiment_results/models/<name>/``::

    manifest.json        ModelArtifactManifest
    config.yaml          frozen StandaloneModelConfig
    weights/             model.save() target (metadata.json + weights.*)

The ``weights/`` subdir is the model's own ``save()`` skeleton (shared
with Phase-5 strategy persistence): ``metadata.json`` (TrainingMetadata),
``config.json`` (ctor kwargs), and one of ``weights.pt`` / ``model.ubj``
/ ``weights.json`` depending on the model class.

``quant_engine_version`` is reserved in the manifest schema — enforcement
(warn on minor mismatch / refuse on major) is deferred until the first
v1.0 cut. Reserving the key now means a later v1.0 loader can validate
it without breaking pre-v1.0 artifacts that never had the field.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING

from src.core import json_io
from src.core.config import write_frozen_yaml
from src.core.persistence import (
    MODEL_ARTIFACT_CONFIG_YAML,
    MODEL_ARTIFACT_MANIFEST_JSON,
    MODEL_ARTIFACT_WEIGHTS_SUBDIR,
    ensure_model_dir,
)
from src.core.registry import classifier_registry, model_registry
from src.core.types import ModelKind

if TYPE_CHECKING:
    from src.core.config import StandaloneModelConfig
    from src.models.interface import IClassifier, IPredictor


_UNKNOWN_VERSION = "unknown"
_QUANT_ENGINE_DIST = "quant-engine"


def _framework_version() -> str:
    """Read ``quant-engine`` version from installed package metadata.

    Returns ``"unknown"`` when the package is not installed as a dist
    (editable installs via ``pip install -e`` are fine — they still
    register dist-info). The unknown fallback lets dev shells / CI that
    haven't run the editable install step still produce artifacts
    rather than crash.
    """
    try:
        return _pkg_version(_QUANT_ENGINE_DIST)
    except PackageNotFoundError:
        return _UNKNOWN_VERSION


@dataclass(frozen=True)
class ModelArtifactManifest:
    """Canonical, round-tripable manifest for a standalone-trained model.

    Every field is the answer to a question a consumer (strategy builder
    loading a pretrained leaf, ``experiment list-models``, a future GC
    script) MUST answer before treating the artifact as trustworthy.

    ``quant_engine_version`` is reserved but not yet enforced — see the
    module docstring for the v1.0 upgrade path.
    """

    name: str
    model_name: str
    model_kind: ModelKind
    created_at: datetime
    git_sha: str
    seed: int
    data_hash: str
    quant_engine_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "model_kind": self.model_kind.value,
            "created_at": self.created_at.isoformat(),
            "git_sha": self.git_sha,
            "seed": self.seed,
            "data_hash": self.data_hash,
            "quant_engine_version": self.quant_engine_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> ModelArtifactManifest:
        return cls(
            name=json_io.get_str(d, "name"),
            model_name=json_io.get_str(d, "model_name"),
            model_kind=ModelKind(json_io.get_str(d, "model_kind")),
            created_at=datetime.fromisoformat(json_io.get_str(d, "created_at")),
            git_sha=json_io.get_str(d, "git_sha"),
            seed=json_io.get_int(d, "seed"),
            data_hash=json_io.get_str(d, "data_hash"),
            quant_engine_version=json_io.get_str(d, "quant_engine_version"),
        )


def build_model_artifact_manifest(
    *,
    name: str,
    model_name: str,
    model_kind: ModelKind,
    git_sha: str,
    seed: int,
    data_hash: str,
) -> ModelArtifactManifest:
    """Fill ``created_at`` + ``quant_engine_version`` automatically.

    Callers pass the fields they control (run-specific) and this helper
    stamps the ambient ones. Centralised so every callsite produces a
    consistent manifest shape without juggling ``datetime.utcnow()`` and
    the package-version lookup inline.
    """
    return ModelArtifactManifest(
        name=name,
        model_name=model_name,
        model_kind=model_kind,
        created_at=datetime.now(UTC),
        git_sha=git_sha,
        seed=seed,
        data_hash=data_hash,
        quant_engine_version=_framework_version(),
    )


def save_model_artifact(
    path: str | Path,
    *,
    model: IPredictor | IClassifier,
    manifest: ModelArtifactManifest,
    config: StandaloneModelConfig,
) -> Path:
    """Write the full artifact bundle at ``path``.

    Order: create the root dir first (fail fast if the path is taken),
    then delegate weights I/O to ``model.save()`` in the ``weights/``
    subdir, then write ``manifest.json``, then write ``config.yaml``.
    Any failure mid-way leaves a partially-written directory; callers
    that need transactional durability should stage to a tmp dir and
    rename atomically.
    """
    root = ensure_model_dir(path)
    weights_dir = root / MODEL_ARTIFACT_WEIGHTS_SUBDIR
    model.save(weights_dir)
    json_io.write(root / MODEL_ARTIFACT_MANIFEST_JSON, manifest.to_dict())
    write_frozen_yaml(root / MODEL_ARTIFACT_CONFIG_YAML, config, sort_keys=False)
    return root


def load_model_artifact(
    path: str | Path,
) -> tuple[IPredictor | IClassifier, ModelArtifactManifest]:
    """Reconstruct model + manifest from ``path``.

    Dispatches via ``manifest.model_kind`` rather than trying both
    registries with a fallback — explicit routing keeps a typo'd
    ``model_name`` from silently loading the wrong class type.

    The ``weights/`` subdir is loaded via the model class's own
    ``load()`` classmethod, which owns the per-model wire format
    (``.pt`` / ``.ubj`` / ``.json``). TrainingMetadata is populated on
    the loaded instance by ``load()`` — the artifact manifest does NOT
    duplicate that information.
    """
    root = Path(path)
    raw = json_io.read_dict(root / MODEL_ARTIFACT_MANIFEST_JSON)
    manifest = ModelArtifactManifest.from_dict(raw)

    cls: type[IPredictor] | type[IClassifier]
    if manifest.model_kind == ModelKind.PREDICTOR:
        cls = model_registry.get(manifest.model_name)
    elif manifest.model_kind == ModelKind.CLASSIFIER:
        cls = classifier_registry.get(manifest.model_name)
    else:  # pragma: no cover — StrEnum total coverage
        raise ValueError(
            f"unknown model_kind {manifest.model_kind!r}; supported: {[k.value for k in ModelKind]}"
        )

    model = cls.load(root / MODEL_ARTIFACT_WEIGHTS_SUBDIR)
    return model, manifest
