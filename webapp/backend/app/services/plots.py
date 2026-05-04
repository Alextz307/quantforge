"""Plot listing + safe path resolution under an artifact's ``plots/`` directory."""

from __future__ import annotations

from pathlib import Path

PLOTS_DIRNAME = "plots"


class PlotNotFoundError(LookupError):
    """Raised when a requested plot file does not exist under an artifact's ``plots/``."""


def list_plots(artifact_dir: Path) -> list[str]:
    """Return the sorted filenames under ``<artifact_dir>/plots/``."""
    plots_dir = artifact_dir / PLOTS_DIRNAME
    if not plots_dir.is_dir():
        return []
    return sorted(p.name for p in plots_dir.iterdir() if p.is_file())


def resolve_plot_path(artifact_dir: Path, plot_name: str) -> Path:
    """Resolve ``plot_name`` to an absolute path under ``<artifact_dir>/plots/``.

    Blocks ``..`` traversal by requiring the resolved candidate to stay
    inside the resolved plots directory. Raises :class:`PlotNotFoundError`
    on traversal attempts and on missing files.
    """
    plots_dir = (artifact_dir / PLOTS_DIRNAME).resolve()
    candidate = (plots_dir / plot_name).resolve()
    if not candidate.is_relative_to(plots_dir) or not candidate.is_file():
        raise PlotNotFoundError(f"plot not found: {artifact_dir.name}/{plot_name}")
    return candidate
