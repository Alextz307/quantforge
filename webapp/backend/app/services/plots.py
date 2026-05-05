"""Static-file access under artifact subdirectories with traversal-safe path resolution."""

from __future__ import annotations

from pathlib import Path

PLOTS_DIRNAME = "plots"
TABLES_DIRNAME = "tables"


class PlotNotFoundError(LookupError):
    """Raised when a requested artifact file does not exist under its expected subdir."""


def list_files_under(artifact_dir: Path, subdir: str) -> list[str]:
    """Return the sorted filenames under ``<artifact_dir>/<subdir>/``."""
    target = artifact_dir / subdir
    if not target.is_dir():
        return []
    return sorted(p.name for p in target.iterdir() if p.is_file())


def resolve_file_under(artifact_dir: Path, subdir: str, name: str) -> Path:
    """Resolve ``name`` to an absolute path under ``<artifact_dir>/<subdir>/``.

    Blocks ``..`` traversal by requiring the resolved candidate to stay
    inside the resolved subdirectory. Raises :class:`PlotNotFoundError`
    on traversal attempts and on missing files.
    """
    base = (artifact_dir / subdir).resolve()
    candidate = (base / name).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise PlotNotFoundError(f"file not found: {artifact_dir.name}/{subdir}/{name}")
    return candidate


def list_plots(artifact_dir: Path) -> list[str]:
    """Return the sorted filenames under ``<artifact_dir>/plots/``."""
    return list_files_under(artifact_dir, PLOTS_DIRNAME)


def resolve_plot_path(artifact_dir: Path, plot_name: str) -> Path:
    """Resolve ``plot_name`` to an absolute path under ``<artifact_dir>/plots/``."""
    return resolve_file_under(artifact_dir, PLOTS_DIRNAME, plot_name)
