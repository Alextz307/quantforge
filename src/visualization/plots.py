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

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  — set backend before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.core.fs import ensure_parent_dir  # noqa: E402
from src.core.logging import get_logger  # noqa: E402

_logger = get_logger(__name__)

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


def normalise_to_unit_base(curve: Sequence[float]) -> list[float] | None:
    """Divide ``curve`` by its first value so the series starts at 1.0.

    Returns ``None`` if the first value is missing, non-finite, or
    non-positive — cases where naive division would produce a misleading
    plot (NaN propagation through matplotlib silently leaves an
    unexplained gap; a non-positive base inverts the sign of the visual
    narrative). Callers downgrade to a placeholder (skip the plot, log a
    warning) on ``None``.
    """
    if not curve:
        return None
    base = curve[0]
    if not math.isfinite(base) or base <= 0.0:
        return None
    return [v / base for v in curve]


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


def render_value_heatmap(
    matrix: npt.NDArray[np.float64],
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    placeholder_log_label: str,
) -> Path:
    """Render a 2D value matrix as a viridis heatmap with masked NaN cells.

    NaN cells render in light grey, finite cells in viridis with .3f text
    coloured white below the midpoint and black above. ``placeholder_log_label``
    appears in the warning when every cell is non-finite — keeps the call
    site identifiable in logs.
    """
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgrey")
    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        _logger.warning(
            "%s heatmap has no finite cells — rendering placeholder", placeholder_log_label
        )
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(finite_values.min()), float(finite_values.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(list(col_labels), rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(list(row_labels))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    midpoint = (vmin + vmax) / 2
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if np.isnan(matrix[i, j]):
                continue
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] < midpoint else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_png_and_svg(fig, out_path)
    plt.close(fig)
    return out_path
