"""
Shared matplotlib primitives for experiment + HPO reporters.

Pins the Agg backend UNCONDITIONALLY (before ``pyplot`` is imported anywhere)
and the figure geometry so every PNG / SVG produced by this codebase renders
at exactly the same size and DPI across macOS / Linux / CI.

Import this module (or any symbol from it) before any other module touches
matplotlib.pyplot - the Agg backend setting is global and sticky.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

from src.core.fs import atomic_write_path
from src.core.logging import get_logger

matplotlib.use("Agg")  # must precede any pyplot import

# Imported for effect: pins pyplot under the Agg backend so any later
# `import matplotlib.pyplot` in the process inherits it.
import matplotlib.pyplot  # noqa: E402, F401
from matplotlib.figure import Figure  # noqa: E402

_logger = get_logger(__name__)

FIGURE_WIDTH_IN = 6.5
FIGURE_HEIGHT_IN = 4.0
FIGURE_DPI = 150

PLOTS_SUBDIR = "plots"
TABLES_SUBDIR = "tables"
MANIFEST_FILENAME = "manifest.json"


def normalise_to_unit_base(curve: Sequence[float]) -> list[float] | None:
    """
    Divide ``curve`` by its first value so the series starts at 1.0.

    Returns ``None`` if the first value is missing, non-finite, or
    non-positive - cases where naive division would produce a misleading
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


def atomic_savefig(fig: Figure, path: Path) -> Path:
    """
    Atomically write ``fig`` to ``path`` via :func:`atomic_write_path`.

    Lives in ``visualization`` rather than ``src.core.fs`` because ``fs`` is
    intentionally matplotlib-free; this helper carries the Figure-dependent
    half of the atomic-write convenience pair.
    """

    with atomic_write_path(path) as tmp:
        fig.savefig(tmp)
    return path


def save_png_and_svg(fig: Figure, png_path: Path) -> Path:
    """
    Save ``fig`` as both PNG (at ``png_path``) and SVG (``png_path`` with
    ``.svg`` suffix), ensuring the parent directory exists. Returns
    ``png_path`` for chaining. Callers remain responsible for ``plt.close(fig)``.

    Why both formats: PNG for README previews + GitHub rendering, SVG for
    vector-quality inclusion in LaTeX via ``\\includegraphics``.
    """

    atomic_savefig(fig, png_path)
    atomic_savefig(fig, png_path.with_suffix(".svg"))
    return png_path
