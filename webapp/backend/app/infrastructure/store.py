"""Path-walking helpers over the experiment_results/ artifact tree.

The store layout under :data:`WebappSettings.store_root` is uniform: every
persisted run lives at some ``<root>/<arbitrary>/runs/<experiment_id>/``,
regardless of whether the parent context is a single-store directory
(``thesis_demo/runs/<id>``) or a study (``studies/main/runs/<id>``). The
walker globs ``**/runs/*/manifest.json`` so both shapes surface uniformly.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from src.core.persistence import EXPERIMENT_MANIFEST_JSON


class RunNotFoundError(LookupError):
    """Raised when an ``experiment_id`` does not match any run under the store root."""


def iter_run_dirs(root: Path) -> Iterator[Path]:
    """Yield every run directory under ``root`` (the parent of ``manifest.json``)."""
    if not root.is_dir():
        return
    for manifest in root.glob(f"**/runs/*/{EXPERIMENT_MANIFEST_JSON}"):
        yield manifest.parent


def find_run_dir(root: Path, experiment_id: str) -> Path:
    """Resolve an ``experiment_id`` to its run directory.

    Raises:
        RunNotFoundError: no run directory matches the given id.
    """
    if root.is_dir():
        for manifest in root.glob(f"**/runs/{experiment_id}/{EXPERIMENT_MANIFEST_JSON}"):
            return manifest.parent
    raise RunNotFoundError(f"run not found: {experiment_id}")


def store_label(run_dir: Path, root: Path) -> str:
    """Human-readable provenance label for a run (path of its store relative to root).

    For ``<root>/thesis_demo/runs/<id>`` returns ``"thesis_demo"``;
    for ``<root>/studies/main/runs/<id>`` returns ``"studies/main"``.
    """
    return run_dir.parent.parent.relative_to(root).as_posix()
