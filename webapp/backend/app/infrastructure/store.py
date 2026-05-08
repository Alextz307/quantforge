"""Path-walking helpers over the experiment_results/ artifact tree.

The store layout under :data:`WebappSettings.store_root` is uniform: every
persisted artifact lives at some ``<root>/<arbitrary>/<subdir>/<artifact_name>/``,
regardless of whether the parent context is a single-store directory
(``thesis_demo/<subdir>/<name>``) or a study (``studies/main/<subdir>/<name>``).
The walker globs ``**/<subdir>/*/<manifest_filename>`` so both shapes surface
uniformly across runs, comparisons, regime reports, and holdout evaluations.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from src.core.persistence import (
    COMPARISONS_SUBDIR,
    EXPERIMENT_MANIFEST_JSON,
    HOLDOUT_EVAL_JSON,
    HOLDOUT_EVALS_SUBDIR,
    HPO_SUBDIR,
    REGIME_REPORTS_SUBDIR,
)
from src.optimization.checkpointing import TRIALS_JSONL_NAME
from src.orchestration.study import STUDY_STATE_FILENAME

STUDIES_SUBDIR = "studies"


class ArtifactNotFoundError(LookupError):
    """Raised when an artifact name does not match anything under the store root."""


class RunNotFoundError(ArtifactNotFoundError):
    """Raised when an ``experiment_id`` does not match any run under the store root."""


class ComparisonNotFoundError(ArtifactNotFoundError):
    """Raised when a comparison name does not match anything under the store root."""


class RegimeReportNotFoundError(ArtifactNotFoundError):
    """Raised when a regime-report name does not match anything under the store root."""


class HoldoutEvalNotFoundError(ArtifactNotFoundError):
    """Raised when a holdout-eval name does not match anything under the store root."""


class StudyNotFoundError(ArtifactNotFoundError):
    """Raised when a study name does not match anything under the store root."""


class HpoStudyNotFoundError(ArtifactNotFoundError):
    """Raised when an HPO-study name does not match anything under the store root."""


_RUNS_SUBDIR = "runs"


def iter_artifact_dirs(root: Path, subdir: str, manifest_filename: str) -> Iterator[Path]:
    """Yield every artifact directory of one kind under ``root``.

    An artifact directory is the parent of its identity file (``manifest.json``
    for runs/comparisons/regime, ``holdout_eval.json`` for holdouts). The
    glob ``**/<subdir>/*/<manifest_filename>`` matches both flat
    (``<store>/<subdir>/<name>``) and nested (``studies/<x>/<subdir>/<name>``)
    layouts.
    """
    if not root.is_dir():
        return
    for manifest in root.glob(f"**/{subdir}/*/{manifest_filename}"):
        yield manifest.parent


def find_artifact_dir(
    root: Path,
    subdir: str,
    manifest_filename: str,
    name: str,
    *,
    not_found: type[ArtifactNotFoundError],
) -> Path:
    """Resolve an artifact ``name`` to its directory.

    Raises ``not_found`` (a subclass of :class:`ArtifactNotFoundError`) when
    no artifact with that name exists under ``root``.
    """
    if root.is_dir():
        for manifest in root.glob(f"**/{subdir}/{name}/{manifest_filename}"):
            return manifest.parent
    raise not_found(f"{subdir} not found: {name}")


def store_label(artifact_dir: Path, root: Path) -> str:
    """Human-readable provenance label for an artifact (path of its parent dir relative to root).

    For ``<root>/hpo/<name>`` returns ``"hpo"``;
    for ``<root>/thesis_demo/runs/<name>`` returns ``"thesis_demo/runs"``;
    for ``<root>/studies/main/hpo/<name>`` returns ``"studies/main/hpo"``.

    Includes the kind subdir so flat-rooted artefacts (default-store tunes
    submitted from the webapp) render as ``"hpo"`` rather than ``"."``.
    """
    return artifact_dir.parent.relative_to(root).as_posix()


def iter_run_dirs(root: Path) -> Iterator[Path]:
    """Yield every run directory under ``root``."""
    return iter_artifact_dirs(root, _RUNS_SUBDIR, EXPERIMENT_MANIFEST_JSON)


def find_run_dir(root: Path, experiment_id: str) -> Path:
    """Resolve an ``experiment_id`` to its run directory."""
    return find_artifact_dir(
        root,
        _RUNS_SUBDIR,
        EXPERIMENT_MANIFEST_JSON,
        experiment_id,
        not_found=RunNotFoundError,
    )


def iter_comparison_dirs(root: Path) -> Iterator[Path]:
    """Yield every comparison directory under ``root``."""
    return iter_artifact_dirs(root, COMPARISONS_SUBDIR, EXPERIMENT_MANIFEST_JSON)


def find_comparison_dir(root: Path, name: str) -> Path:
    """Resolve a comparison ``name`` to its directory."""
    return find_artifact_dir(
        root,
        COMPARISONS_SUBDIR,
        EXPERIMENT_MANIFEST_JSON,
        name,
        not_found=ComparisonNotFoundError,
    )


def iter_regime_report_dirs(root: Path) -> Iterator[Path]:
    """Yield every regime-report directory under ``root``."""
    return iter_artifact_dirs(root, REGIME_REPORTS_SUBDIR, EXPERIMENT_MANIFEST_JSON)


def find_regime_report_dir(root: Path, name: str) -> Path:
    """Resolve a regime-report ``name`` to its directory."""
    return find_artifact_dir(
        root,
        REGIME_REPORTS_SUBDIR,
        EXPERIMENT_MANIFEST_JSON,
        name,
        not_found=RegimeReportNotFoundError,
    )


def iter_holdout_eval_dirs(root: Path) -> Iterator[Path]:
    """Yield every holdout-eval directory under ``root``."""
    return iter_artifact_dirs(root, HOLDOUT_EVALS_SUBDIR, HOLDOUT_EVAL_JSON)


def find_holdout_eval_dir(root: Path, name: str) -> Path:
    """Resolve a holdout-eval ``name`` to its directory."""
    return find_artifact_dir(
        root,
        HOLDOUT_EVALS_SUBDIR,
        HOLDOUT_EVAL_JSON,
        name,
        not_found=HoldoutEvalNotFoundError,
    )


def iter_study_dirs(root: Path) -> Iterator[Path]:
    """Yield every study directory under ``root``."""
    return iter_artifact_dirs(root, STUDIES_SUBDIR, STUDY_STATE_FILENAME)


def find_study_dir(root: Path, name: str) -> Path:
    """Resolve a study ``name`` to its directory."""
    return find_artifact_dir(
        root,
        STUDIES_SUBDIR,
        STUDY_STATE_FILENAME,
        name,
        not_found=StudyNotFoundError,
    )


def iter_hpo_study_dirs(root: Path) -> Iterator[Path]:
    """Yield every HPO-study directory under ``root``."""
    return iter_artifact_dirs(root, HPO_SUBDIR, TRIALS_JSONL_NAME)


def find_hpo_study_dir(root: Path, name: str) -> Path:
    """Resolve an HPO-study ``name`` to its directory."""
    return find_artifact_dir(
        root,
        HPO_SUBDIR,
        TRIALS_JSONL_NAME,
        name,
        not_found=HpoStudyNotFoundError,
    )
