"""Per-experiment report generator.

Consumes a persisted or in-memory :class:`ExperimentResult` and writes
thesis-ready artifacts under ``<run_dir>/plots/`` + ``<run_dir>/tables/``.

The three artifacts Chapter 7 relies on at the single-experiment level:

* ``plots/equity_curves.png/svg`` — per-fold equity curves overlaid, each
  series normalised to 1.0 at fold start so a reader eyeballs the fold's
  own performance rather than being dominated by compounding across folds.
* ``plots/fold_stability.png/svg`` — fold-index vs. Sharpe dots with a
  horizontal zero line. Walk-forward stability check: tight cluster = the
  strategy is robust across windows; wide scatter = overfit or regime-
  sensitive.
* ``tables/metrics_summary.tex`` — booktabs LaTeX, one row per fold with
  Sharpe / Sortino / Calmar / MaxDD / TotalReturn / TradeCount. Per-run
  mean is NOT included here (it's a cross-run concern; ``comparison``
  reporter owns that).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.core.logging import get_logger
from src.orchestration.types import ExperimentResult, FoldRecord
from src.visualization.latex import write_booktabs_table
from src.visualization.plots import FIGURE_DPI, FIGURE_HEIGHT_IN, FIGURE_WIDTH_IN, save_png_and_svg

_logger = get_logger(__name__)

_PLOTS_SUBDIR = "plots"
_TABLES_SUBDIR = "tables"
_EQUITY_FILENAME = "equity_curves.png"
_STABILITY_FILENAME = "fold_stability.png"
_METRICS_FILENAME = "metrics_summary.tex"


class StrategyReporter:
    """Generate the single-experiment report bundle."""

    def generate_full_report(
        self,
        result: ExperimentResult,
        out_dir: Path,
    ) -> Path:
        """Write every artifact under ``out_dir/{plots,tables}/`` and return ``out_dir``.

        Safe on an empty ``result.folds`` (the walk-forward validator could
        in principle yield zero folds if someone mis-configures): writes a
        placeholder metrics table with zero rows and no plots, so the
        consuming command doesn't silently produce an empty directory.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = out_dir / _PLOTS_SUBDIR
        tables_dir = out_dir / _TABLES_SUBDIR

        metrics_df = self._build_metrics_dataframe(result.folds)
        write_booktabs_table(
            metrics_df,
            tables_dir / _METRICS_FILENAME,
            caption=f"Fold metrics — experiment {result.experiment_id}",
            label=f"tab:metrics_{result.experiment_id}",
        )

        if result.folds:
            self._plot_equity_curves(result.folds, plots_dir / _EQUITY_FILENAME)
            self._plot_fold_stability(result.folds, plots_dir / _STABILITY_FILENAME)

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
            normalised = _normalise_curve(fold.equity_curve)
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

    def _plot_fold_stability(self, folds: tuple[FoldRecord, ...], out_path: Path) -> Path:
        """Sharpe per fold as a scatter with a horizontal zero line.

        Thesis-narrative check: a strategy that's robust across windows
        produces a tight cluster above 0; wide scatter + frequent dips
        below 0 means the walk-forward didn't generalise.

        Folds with non-finite Sharpe (zero-trade or zero-volatility windows)
        are logged + filtered before plotting — matplotlib silently skips
        NaN points, which would leave an unexplained gap in the scatter.
        """
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)
        indices: list[int] = []
        sharpes: list[float] = []
        for fold in folds:
            if not math.isfinite(fold.sharpe_ratio):
                _logger.warning(
                    "fold %d: sharpe_ratio=%s is non-finite — skipping from stability plot",
                    fold.fold_index,
                    fold.sharpe_ratio,
                )
                continue
            indices.append(fold.fold_index)
            sharpes.append(fold.sharpe_ratio)
        ax.scatter(indices, sharpes, s=40, alpha=0.8)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("fold index")
        ax.set_ylabel("Sharpe ratio")
        ax.set_title("walk-forward stability — per-fold Sharpe")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        save_png_and_svg(fig, out_path)
        plt.close(fig)
        return out_path


def _normalise_curve(curve: tuple[float, ...]) -> list[float] | None:
    """Divide ``curve`` by its first value. Returns ``None`` if the first
    value is missing, non-finite, or non-positive — cases where naive
    division would produce a misleading plot (NaN propagation or a
    sign-inverted series).
    """
    if not curve:
        return None
    base = curve[0]
    if not math.isfinite(base) or base <= 0.0:
        return None
    return [v / base for v in curve]
