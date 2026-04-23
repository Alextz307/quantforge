"""Tests for :mod:`src.visualization.latex` — booktabs table builder.

Verifies structural LaTeX properties, not source-text patterns (per the
testing-philosophy rule). A regression here would silently corrupt every
``.tex`` file the thesis pulls via ``\\input``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization.latex import build_booktabs_table, write_booktabs_table


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"fold": 0, "sharpe": 1.234, "sortino": 1.8},
            {"fold": 1, "sharpe": 0.7, "sortino": 0.9},
        ]
    )


class TestBuildBooktabsTable:
    def test_output_is_brace_balanced(self) -> None:
        """Malformed LaTeX tables break thesis compilation with cryptic errors;
        the brace count is the cheapest structural invariant."""
        out = build_booktabs_table(_sample_df(), caption="x", label="tab:x")
        assert out.count("{") == out.count("}")

    def test_caption_and_label_embedded(self) -> None:
        out = build_booktabs_table(_sample_df(), caption="my caption", label="tab:mine")
        assert "my caption" in out
        assert "tab:mine" in out

    def test_contains_booktabs_rules(self) -> None:
        """Styled via ``to_latex(..., escape=False)`` which emits toprule /
        midrule / bottomrule under the hood — sanity check the styling
        actually kicked in."""
        out = build_booktabs_table(_sample_df(), caption="x", label="tab:x")
        assert "toprule" in out
        assert "bottomrule" in out

    def test_index_not_written_by_default(self) -> None:
        out = build_booktabs_table(_sample_df(), caption="x", label="tab:x")
        # Pandas writes a column for the index when index=True; the default
        # should skip it so table columns match DataFrame columns.
        lines = out.splitlines()
        header_idx = next(i for i, line in enumerate(lines) if "sharpe" in line)
        assert "index" not in lines[header_idx].lower()


class TestWriteBooktabsTable:
    def test_creates_file_and_parent(self, tmp_path: Path) -> None:
        target = tmp_path / "tables" / "metrics.tex"
        write_booktabs_table(_sample_df(), target, caption="x", label="tab:x")
        assert target.is_file()
        assert "toprule" in target.read_text()
