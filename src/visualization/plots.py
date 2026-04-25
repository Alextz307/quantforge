"""Shared matplotlib primitives for experiment + HPO + regime reporters.

Pins the Agg backend UNCONDITIONALLY (before ``pyplot`` is imported anywhere)
and the figure geometry so every PNG / SVG produced by this codebase renders
at exactly the same size and DPI across macOS / Linux / CI. The
:class:`BenchmarkReporter` uses identical constants — drift between the two
would show up as a subtle thesis-figure inconsistency.

Import this module (or any symbol from it) before any other module touches
matplotlib.pyplot — the Agg backend setting is global and sticky.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  — set backend before importing pyplot

from matplotlib.figure import Figure  # noqa: E402

from src.core.fs import ensure_parent_dir  # noqa: E402

FIGURE_WIDTH_IN = 6.5
FIGURE_HEIGHT_IN = 4.0
FIGURE_DPI = 150

# Shared report-bundle layout. Every reporter under this package writes
# under ``out_dir / PLOTS_SUBDIR`` and ``out_dir / TABLES_SUBDIR`` and
# emits its identity sidecar at ``out_dir / MANIFEST_FILENAME`` so a
# rename here moves all four reporters in lockstep.
PLOTS_SUBDIR = "plots"
TABLES_SUBDIR = "tables"
MANIFEST_FILENAME = "manifest.json"


def save_png_and_svg(fig: Figure, png_path: Path) -> Path:
    """Save ``fig`` as both PNG (at ``png_path``) and SVG (``png_path`` with
    ``.svg`` suffix), ensuring the parent directory exists. Returns
    ``png_path`` for chaining. Callers remain responsible for ``plt.close(fig)``.

    Why both formats: PNG for README previews + GitHub rendering, SVG for
    vector-quality inclusion in LaTeX via ``\\includegraphics``.
    """
    ensure_parent_dir(png_path)
    fig.savefig(png_path)
    fig.savefig(png_path.with_suffix(".svg"))
    return png_path
