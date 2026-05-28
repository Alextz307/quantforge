"""
Reconstruct in-memory experiment artifacts from a persisted run directory.

Two readers, both anchored on the canonical persistence layout
(``<run_dir>/manifest.json`` + ``fold_results.jsonl`` + ``config.yaml``):

* :func:`load_experiment_result` — used by ``experiment compare
  --reuse-runs`` so the comparison reporter can rank + bootstrap-test
  prior runs without retraining the underlying experiments.
* :func:`load_experiment_config_from_run` — reads the frozen
  ``config.yaml`` back into a typed :class:`ExperimentConfig`.
* :func:`resolve_run_dir` — path helper that joins ``<store_root>/runs/<id>``.
"""

from __future__ import annotations

from pathlib import Path

from src.core import json_io
from src.core.config import ExperimentConfig, load_experiment_config
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_MANIFEST_JSON,
    FOLD_RESULTS_JSONL,
    RUNS_SUBDIR,
)
from src.orchestration.manifest import Manifest
from src.orchestration.types import ExperimentResult, FoldRecord


def load_experiment_result(run_dir: Path) -> ExperimentResult:
    """
    Reconstruct an :class:`ExperimentResult` from a persisted run directory.

    Reads ``manifest.json`` + ``fold_results.jsonl`` only; the frozen
    ``config.yaml`` is intentionally NOT required here so the comparison
    reuse path doesn't pay the YAML+pydantic cost when it only needs
    fold-level data.

    Raises:
        FileNotFoundError: ``run_dir`` is not a directory, or either of
            the required artifacts is missing.
    """

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"experiment run directory not found: {run_dir}; "
            f"check the --reuse-runs path resolves to a previous run."
        )
    manifest_path = run_dir / EXPERIMENT_MANIFEST_JSON
    folds_path = run_dir / FOLD_RESULTS_JSONL
    for required in (manifest_path, folds_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"missing artifact {required.name} under {run_dir}; the "
                f"source run may be incomplete — re-run the experiment "
                f"or pass a different path."
            )
    manifest = Manifest.from_dict(json_io.read_dict(manifest_path))
    folds = tuple(FoldRecord.from_dict(d) for d in json_io.read_jsonl(folds_path))
    return ExperimentResult(
        experiment_id=manifest.experiment_id,
        folds=folds,
        manifest=manifest,
    )


def load_experiment_config_from_run(run_dir: Path) -> ExperimentConfig:
    """
    Read the frozen ``config.yaml`` from a persisted run directory.
    """

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"experiment run directory not found: {run_dir}; "
            f"check the --reuse-runs path resolves to a previous run."
        )
    config_path = run_dir / EXPERIMENT_CONFIG_YAML
    if not config_path.is_file():
        raise FileNotFoundError(
            f"missing {EXPERIMENT_CONFIG_YAML} under {run_dir}; the run "
            f"may be incomplete — re-run the experiment or pass a "
            f"different path."
        )
    return load_experiment_config(config_path)


def resolve_run_dir(store_root: Path, experiment_id: str) -> Path:
    """
    Resolve ``store_root / runs / <experiment_id>``.
    """

    return store_root / RUNS_SUBDIR / experiment_id
