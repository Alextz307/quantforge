"""
Static-file access under artifact subdirectories with traversal-safe path resolution.
"""

from __future__ import annotations

from pathlib import Path

PLOTS_DIRNAME = "plots"
TABLES_DIRNAME = "tables"

# Hide artifacts produced by features that have been removed from the
# framework but still exist on disk for historical runs. Filenames are
# matched as case-sensitive prefixes against the stem (extension stripped).
_RETIRED_ARTIFACT_PREFIXES: tuple[str, ...] = (
    "regime_",
    "strategy_x_regime",
    "fold_stability",
)


def _is_retired(name: str) -> bool:
    stem = name.split(".", 1)[0]
    return any(stem.startswith(prefix) for prefix in _RETIRED_ARTIFACT_PREFIXES)


class PlotNotFoundError(LookupError):
    """
    Raised when a requested artifact file does not exist under its expected subdir.
    """


def list_files_under(artifact_dir: Path, subdir: str) -> list[str]:
    """
    Return the sorted filenames under ``<artifact_dir>/<subdir>/``.

    Filenames produced by removed features (regime analysis, fold-stability
    scatter) are filtered out so legacy run/study directories don't surface
    artifacts the rest of the framework no longer supports.
    """

    target = artifact_dir / subdir
    if not target.is_dir():
        return []
    return sorted(p.name for p in target.iterdir() if p.is_file() and not _is_retired(p.name))


def resolve_file_under(artifact_dir: Path, subdir: str, name: str) -> Path:
    """
    Resolve ``name`` to an absolute path under ``<artifact_dir>/<subdir>/``.

    Blocks ``..`` traversal by requiring the resolved candidate to stay
    inside the resolved subdirectory. Raises :class:`PlotNotFoundError`
    on traversal attempts, on missing files, and on filenames produced
    by retired features (so stale URLs cannot bypass the list filter).
    """

    if _is_retired(name):
        raise PlotNotFoundError(f"file not found: {artifact_dir.name}/{subdir}/{name}")
    base = (artifact_dir / subdir).resolve()
    candidate = (base / name).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise PlotNotFoundError(f"file not found: {artifact_dir.name}/{subdir}/{name}")
    return candidate


def list_plots(artifact_dir: Path) -> list[str]:
    """
    Return the sorted filenames under ``<artifact_dir>/plots/``.
    """

    return list_files_under(artifact_dir, PLOTS_DIRNAME)


def resolve_plot_path(artifact_dir: Path, plot_name: str) -> Path:
    """
    Resolve ``plot_name`` to an absolute path under ``<artifact_dir>/plots/``.
    """

    return resolve_file_under(artifact_dir, PLOTS_DIRNAME, plot_name)
