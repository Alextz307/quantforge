"""Per-experiment report generator.

Consumes a persisted or in-memory :class:`ExperimentResult` and writes
thesis-ready artifacts under ``<run_dir>/plots/`` + ``<run_dir>/tables/``.

The two artifacts Chapter 7 relies on at the single-experiment level:

* ``plots/equity_curves.png/svg`` — per-fold equity curves overlaid, each
  series normalised to 1.0 at fold start so a reader eyeballs the fold's
  own performance rather than being dominated by compounding across folds.
* ``tables/metrics_summary.tex`` — booktabs LaTeX, one row per fold with
  Sharpe / Sortino / Calmar / MaxDD / TotalReturn / TradeCount. Per-run
  mean is NOT included here (it's a cross-run concern; ``comparison``
  reporter owns that).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.core.logging import get_logger
from src.orchestration.types import ExperimentResult, FoldRecord
from src.visualization.latex import validate_publish_label, write_booktabs_table
from src.visualization.plots import (
    FIGURE_DPI,
    FIGURE_HEIGHT_IN,
    FIGURE_WIDTH_IN,
    PLOTS_SUBDIR,
    TABLES_SUBDIR,
    normalise_to_unit_base,
    save_png_and_svg,
)

_logger = get_logger(__name__)

_EQUITY_FILENAME = "equity_curves.png"
_METRICS_FILENAME = "metrics_summary.tex"


class StrategyReporter:
    """Generate the single-experiment report bundle."""

    def generate_full_report(
        self,
        result: ExperimentResult,
        out_dir: Path,
        *,
        publish_label: str | None = None,
    ) -> Path:
        """Write every artifact under ``out_dir/{plots,tables}/`` and return ``out_dir``.

        ``publish_label`` overrides the volatile ``experiment_id`` in the
        ``\\caption`` text and ``\\label`` of the metrics table; pass it
        when the .tex is committed and referenced from prose so a rerun
        of the experiment doesn't churn the citation slug.

        Safe on an empty ``result.folds`` (the walk-forward validator could
        in principle yield zero folds if someone mis-configures): writes a
        placeholder metrics table with zero rows and no plots, so the
        consuming command doesn't silently produce an empty directory.
        """

        out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = out_dir / PLOTS_SUBDIR
        tables_dir = out_dir / TABLES_SUBDIR

        if publish_label is not None:
            slug = validate_publish_label(publish_label)
            caption = f"Fold metrics — {slug}"
            label = f"tab:metrics_{slug}"
        else:
            caption = f"Fold metrics — experiment {result.experiment_id}"
            label = f"tab:metrics_{result.experiment_id}"

        metrics_df = self._build_metrics_dataframe(result.folds)
        write_booktabs_table(
            metrics_df,
            tables_dir / _METRICS_FILENAME,
            caption=caption,
            label=label,
        )

        if result.folds:
            self._plot_equity_curves(result.folds, plots_dir / _EQUITY_FILENAME)

        return out_dir

    def _build_metrics_dataframe(self, folds: tuple[FoldRecord, ...]) -> pd.DataFrame:
        if not folds:
            return pd.DataFrame(
                columns=[
                    "fold",
                    "sharpe",
                    "sortino",
                    "calmar",
                    "max_drawdown",
                    "total_return",
                    "trades",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "fold": f.fold_index,
                    "sharpe": f.sharpe_ratio,
                    "sortino": f.sortino_ratio,
                    "calmar": f.calmar_ratio,
                    "max_drawdown": f.max_drawdown,
                    "total_return": f.total_return,
                    "trades": f.trade_count,
                }
                for f in folds
            ]
        )

    def _plot_equity_curves(self, folds: tuple[FoldRecord, ...], out_path: Path) -> Path:
        """Overlay per-fold equity curves normalised to 1.0 at fold start.

        Normalisation is per-fold (divide by the first value) so folds with
        very different absolute equity levels stay visually comparable —
        otherwise the lowest-equity fold crushes the plot's y-axis.

        Folds whose initial equity is non-finite (NaN/inf — zero-trade fold
        with degenerate metrics) or non-positive (catastrophic loss leaving
        debt at fold start) are logged + skipped rather than dividing by a
        bad base. NaN propagates silently through matplotlib's plotter and
        a negative base inverts the visual narrative; either produces a
        plot that mis-tells Chapter 7's story.
        """

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)
        plotted = 0
        for fold in folds:
            normalised = normalise_to_unit_base(fold.equity_curve)
            if normalised is None:
                _logger.warning(
                    "fold %d: equity_curve[0]=%s is non-finite or non-positive — "
                    "skipping from equity plot",
                    fold.fold_index,
                    fold.equity_curve[0] if fold.equity_curve else "empty",
                )
                continue
            ax.plot(normalised, label=f"fold {fold.fold_index}", linewidth=1.0)
            plotted += 1
        ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("bar index within fold")
        ax.set_ylabel("equity (normalised to fold start)")
        ax.set_title("per-fold equity curves")
        ax.grid(True, which="both", alpha=0.3)
        if plotted > 0:
            ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        save_png_and_svg(fig, out_path)
        plt.close(fig)
        return out_path
