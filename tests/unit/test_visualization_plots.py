"""
Tests for :mod:`src.visualization.plots` — shared matplotlib primitives.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

from src.visualization.plots import (
    FIGURE_DPI,
    FIGURE_HEIGHT_IN,
    FIGURE_WIDTH_IN,
    save_png_and_svg,
)


def test_agg_backend_is_active() -> None:
    """
    Importing the module must force the Agg backend before any pyplot
    import touches the GUI — otherwise CI jobs crash on the headless box."""

    assert matplotlib.get_backend().lower() == "agg"


def test_dimensions_are_thesis_sized() -> None:
    """
    Figure geometry stays pinned so every PNG / SVG across subsystems
    renders identically in thesis margins."""

    assert FIGURE_WIDTH_IN == 6.5
    assert FIGURE_HEIGHT_IN == 4.0
    assert FIGURE_DPI == 150


def test_save_png_and_svg_writes_both(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)
    ax.plot([1, 2, 3], [1, 4, 9])
    png = tmp_path / "nested" / "plot.png"
    save_png_and_svg(fig, png)
    plt.close(fig)
    assert png.is_file()
    assert png.with_suffix(".svg").is_file()


def test_save_png_and_svg_creates_parent(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots()
    target = tmp_path / "deep" / "tree" / "plot.png"
    save_png_and_svg(fig, target)
    plt.close(fig)
    assert target.parent.is_dir()
