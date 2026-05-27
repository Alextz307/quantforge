"""Cross-strategy comparison report generator.

Consumes an in-memory :class:`StrategyComparisonReport` and writes:

* ``manifest.json``               — scalar identity (out_name, timestamp,
                                    git sha) + per-strategy stats + the
                                    name→experiment_id map so a reader can
                                    drill into any specific run.
* ``tables/ranking.tex``          — booktabs LaTeX of the rank DataFrame.
* ``tables/pairwise_significance.tex`` — upper-triangular pairwise table
                                    (only written when at least one pair
                                    was computed).
* ``plots/equity_overlay.png/svg`` — overlaid per-strategy concatenated
                                    equity curves, each normalised to 1.0
                                    at the first bar so strategies with
                                    different starting equity remain
                                    visually comparable.

The reporter is a separate class from :class:`StrategyReporter` rather
than an extension — single-experiment plots and cross-experiment tables
don't share much code, and keeping them in sibling classes matches the
HPOReporter / StrategyReporter split already in place.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from src.core import json_io
from src.core.logging import get_logger
from src.orchestration.types import (
    FoldRecord,
    PairwiseSignificance,
    StrategyComparisonReport,
)
from src.visualization.latex import validate_publish_label, write_booktabs_table
from src.visualization.plots import (
    FIGURE_DPI,
    FIGURE_HEIGHT_IN,
    FIGURE_WIDTH_IN,
    MANIFEST_FILENAME,
    PLOTS_SUBDIR,
    TABLES_SUBDIR,
    save_png_and_svg,
)

_logger = get_logger(__name__)

_RANKING_FILENAME = "ranking.tex"
_PAIRWISE_FILENAME = "pairwise_significance.tex"
EQUITY_OVERLAY_FILENAME = "equity_overlay.png"


class ComparisonReporter:
    """Generate the cross-strategy report bundle."""

    def generate_full_report(
        self,
        report: StrategyComparisonReport,
        out_dir: Path,
        *,
        folds_by_strategy: dict[str, tuple[FoldRecord, ...]] | None = None,
        publish_label: str | None = None,
    ) -> Path:
        """Write every artifact under ``out_dir`` and return ``out_dir``.

        ``folds_by_strategy`` feeds the equity overlay — it lives
        alongside :class:`StrategyComparisonReport` rather than inside it
        because fold records are heavy and the comparison orchestrator
        already has them in hand. If ``None`` (report loaded from disk
        without fold data), the overlay plot is skipped and a log line
        records why.

        ``publish_label`` overrides ``report.out_name`` in every emitted
        ``\\caption`` / ``\\label``. Useful when the on-disk directory
        name diverges from the citation slug (e.g. a re-run committed
        under a fresh ``--out-name`` that should still ``\\ref`` to the
        original label in thesis prose).
        """

        out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = out_dir / PLOTS_SUBDIR
        tables_dir = out_dir / TABLES_SUBDIR

        slug = (
            validate_publish_label(publish_label) if publish_label is not None else report.out_name
        )

        _logger.info(
            "generating comparison report '%s' with %d strategies, %d pairwise entries",
            report.out_name,
            len(report.per_strategy_stats),
            len(report.pairwise),
        )

        json_io.write(out_dir / MANIFEST_FILENAME, _build_manifest_dict(report))

        write_booktabs_table(
            report.ranking,
            tables_dir / _RANKING_FILENAME,
            caption=f"Strategy ranking — comparison {slug}",
            label=f"tab:ranking_{slug}",
        )

        if report.pairwise:
            write_booktabs_table(
                self._build_pairwise_table(report.pairwise),
                tables_dir / _PAIRWISE_FILENAME,
                caption=(
                    f"Pairwise Sharpe differential, 95\\% stationary bootstrap CI "
                    f"(row - col) — comparison {slug}"
                ),
                label=f"tab:pairwise_{slug}",
            )

        if folds_by_strategy is None:
            _logger.info("skipping equity overlay: folds_by_strategy not provided")
        else:
            self._plot_equity_overlay(folds_by_strategy, plots_dir / EQUITY_OVERLAY_FILENAME)

        return out_dir

    def _build_pairwise_table(self, pairwise: tuple[PairwiseSignificance, ...]) -> pd.DataFrame:
        """Upper-triangular pairwise Sharpe-differential matrix.

        Each cell reads ``point [low, high]*`` where the trailing star
        indicates significance at the stored confidence level (CI
        excludes zero). Lower triangle + diagonal are blank so the
        asymmetric ``row - col`` semantics are unambiguous.
        """

        names = _unique_names(pairwise)
        df = pd.DataFrame("", index=names, columns=names)
        lookup = {(p.name_a, p.name_b): p for p in pairwise}
        for i, row in enumerate(names):
            for j, col in enumerate(names):
                if i >= j:
                    continue
                p = lookup.get((row, col))
                if p is None:
                    continue
                star = "*" if p.significant else ""
                df.at[row, col] = (
                    f"{p.point_differential:+.3f} [{p.lower:+.3f}, {p.upper:+.3f}]{star}"
                )
        df.insert(0, "strategy", df.index)
        df = df.reset_index(drop=True)
        return df

    def _plot_equity_overlay(
        self,
        folds_by_strategy: dict[str, tuple[FoldRecord, ...]],
        out_path: Path,
    ) -> Path:
        """Overlay each strategy's concatenated equity curve.

        Each concatenated curve is normalised to 1.0 at its first bar —
        strategies with very different absolute equity levels would
        otherwise dominate the y-axis and collapse the interesting
        visual comparison.
        """

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)
        plotted = 0
        for name, folds in sorted(folds_by_strategy.items()):
            curve = _concatenated_equity_normalised(folds)
            if curve is None:
                _logger.warning(
                    "strategy '%s' has no plottable equity curve — skipping from overlay",
                    name,
                )
                continue
            ax.plot(curve, label=name, linewidth=1.2)
            plotted += 1
        ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("bar index (concatenated across folds)")
        ax.set_ylabel("equity (normalised to first bar)")
        ax.set_title("per-strategy equity overlay")
        ax.grid(True, which="both", alpha=0.3)
        if plotted > 0:
            ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        save_png_and_svg(fig, out_path)
        plt.close(fig)
        return out_path


def _build_manifest_dict(report: StrategyComparisonReport) -> dict[str, object]:
    """Flatten :class:`StrategyComparisonReport` identity + stats for ``manifest.json``.

    DataFrames and heavy per-fold data are omitted; the ranking is covered
    by ``ranking.tex`` and fold-level data belongs under ``runs/``.
    Pairwise records round-trip via ``to_dict`` / ``from_dict``.
    """

    payload: dict[str, object] = {
        "out_name": report.out_name,
        "created_at": report.created_at.isoformat(),
        "git_sha": report.git_sha,
        "per_strategy_experiment_id": dict(report.per_strategy_experiment_id),
        "per_strategy_stats": {
            name: stats.to_dict() for name, stats in report.per_strategy_stats.items()
        },
        "pairwise": [p.to_dict() for p in report.pairwise],
    }
    return payload


def _unique_names(pairwise: tuple[PairwiseSignificance, ...]) -> list[str]:
    """Preserve first-seen order from the pairwise list (matches comparison order)."""

    return list(dict.fromkeys(name for p in pairwise for name in (p.name_a, p.name_b)))


def _concatenated_equity_normalised(
    folds: tuple[FoldRecord, ...],
) -> npt.NDArray[np.float64] | None:
    """Concatenate fold equity curves and normalise to ``curve[0] = 1.0``.

    Returns ``None`` if the concatenated curve is empty or its first
    value is non-finite / non-positive — matplotlib would silently NaN
    through a bad divisor and produce a misleading overlay.
    """

    pieces = [
        np.asarray(fold.equity_curve, dtype=np.float64)
        for fold in folds
        if len(fold.equity_curve) > 0
    ]
    if not pieces:
        return None
    curve = np.concatenate(pieces)
    if curve.size == 0:
        return None
    base = float(curve[0])
    if not math.isfinite(base) or base <= 0.0:
        return None
    normalised: npt.NDArray[np.float64] = curve / base
    return normalised
