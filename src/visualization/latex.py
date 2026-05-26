"""LaTeX table builder shared by strategy / HPO reporters.

``build_booktabs_table`` is the single call site between pandas DataFrames
and a booktabs-styled LaTeX ``tabular`` environment. Every reporter writes
``.tex`` files by calling through here — keeps caption/label conventions
uniform and lets a future style tweak (e.g. swap to ``tabularx``) land in
one file instead of five.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.core.fs import ensure_parent_dir

LATEX_FLOAT_FORMAT = "%.3f"

# A LaTeX label / file-friendly slug: starts with a letter, then any mix
# of letters, digits, ``_``, ``-``, ``:``. The ``:`` is allowed because
# the project's existing labels use it (``tab:metrics_demo``); ``_``
# and ``-`` are universally safe in LaTeX. Anything else (spaces,
# braces, percent, hash) breaks ``\\ref{...}`` or the .tex itself.
_PUBLISH_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_:\-]*$")


def validate_publish_label(slug: str) -> str:
    """Return ``slug`` unchanged when valid; raise :class:`ValueError` otherwise.

    Used by every reporter that accepts a ``publish_label`` override —
    one regex, one error message, no per-reporter drift.
    """
    if not _PUBLISH_LABEL_RE.fullmatch(slug):
        raise ValueError(
            f"invalid publish_label '{slug}': must start with a letter "
            f"and contain only letters, digits, '_', '-', ':' (LaTeX "
            f"label-friendly chars). Fix by passing a slug like "
            f"'metrics_volatility_targeting_spy'."
        )
    return slug


def build_booktabs_table(
    df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    float_format: str = LATEX_FLOAT_FORMAT,
    index: bool = False,
) -> str:
    """Render ``df`` as a booktabs LaTeX table string.

    Uses :meth:`pandas.DataFrame.to_latex` with booktabs styling and
    ``escape=False`` so callers can pass LaTeX math directly in cell values
    (e.g. ``$\\pm$``, greek letters). Callers that need cell-level escaping
    should escape upstream via ``pd.io.formats.style.Styler`` or similar;
    this builder is deliberately thin so the thesis-wide LaTeX conventions
    live in one place.
    """
    return df.to_latex(
        index=index,
        escape=False,
        float_format=float_format,
        caption=caption,
        label=label,
    )


def write_booktabs_table(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    float_format: str = LATEX_FLOAT_FORMAT,
    index: bool = False,
) -> Path:
    """Convenience wrapper: build table + write UTF-8 bytes, ensuring parent dir."""
    latex = build_booktabs_table(
        df, caption=caption, label=label, float_format=float_format, index=index
    )
    ensure_parent_dir(path)
    path.write_text(latex, encoding="utf-8")
    return path
