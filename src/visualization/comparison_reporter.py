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
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from src.analysis.metrics_aggregator import AggregateStats
from src.core import json_io
from src.core.logging import get_logger
from src.orchestration.types import (
    MIXED_REGIME_LABEL,
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
    render_value_heatmap,
    save_png_and_svg,
)

_logger = get_logger(__name__)

_RANKING_FILENAME = "ranking.tex"
_PAIRWISE_FILENAME = "pairwise_significance.tex"
_EQUITY_OVERLAY_FILENAME = "equity_overlay.png"
_STRATEGY_X_REGIME_PLOT_FILENAME = "strategy_x_regime_heatmap.png"
_STRATEGY_X_REGIME_TABLE_FILENAME = "strategy_x_regime.tex"


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
            self._plot_equity_overlay(folds_by_strategy, plots_dir / _EQUITY_OVERLAY_FILENAME)

        if report.per_strategy_per_regime_stats is not None:
            write_booktabs_table(
                _build_strategy_x_regime_df(report.per_strategy_per_regime_stats),
                tables_dir / _STRATEGY_X_REGIME_TABLE_FILENAME,
                caption=(
                    f"Per-strategy mean Sharpe ($\\pm$ std) split by regime — comparison {slug}"
                ),
                label=f"tab:strategy_x_regime_{slug}",
            )
            self._plot_strategy_x_regime_heatmap(
                report.per_strategy_per_regime_stats,
                plots_dir / _STRATEGY_X_REGIME_PLOT_FILENAME,
            )
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

    def _plot_strategy_x_regime_heatmap(
        self,
        per_strategy_per_regime_stats: Mapping[str, Mapping[str, AggregateStats]],
        out_path: Path,
    ) -> Path:
        """Render the strategy × regime heatmap (cell = ``sharpe_mean``).

        Cells without a fold (``n_folds == 0``) are masked grey.
        :data:`MIXED_REGIME_LABEL` is pinned last when present.
        """
        strategies = list(per_strategy_per_regime_stats)
        labels = _ordered_regime_labels(per_strategy_per_regime_stats)
        if not strategies or not labels:
            _logger.warning(
                "strategy × regime heatmap has no cells (strategies=%d, regimes=%d) — skipping",
                len(strategies),
                len(labels),
            )
            return out_path

        matrix = np.full((len(strategies), len(labels)), np.nan, dtype=np.float64)
        for i, strategy in enumerate(strategies):
            per_regime = per_strategy_per_regime_stats[strategy]
            for j, label in enumerate(labels):
                stats = per_regime.get(label)
                if stats is None or stats.n_folds == 0:
                    continue
                matrix[i, j] = stats.sharpe_mean

        return render_value_heatmap(
            matrix,
            row_labels=strategies,
            col_labels=labels,
            out_path=out_path,
            title="strategy × regime (Sharpe)",
            xlabel="regime",
            ylabel="strategy",
            placeholder_log_label="strategy × regime",
        )

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

    DataFrames (``ranking``) and heavy per-fold data are deliberately
    omitted — the ranking is covered by ``ranking.tex`` and fold-level
    data belongs in the per-run directories under ``runs/``.
    """
    payload: dict[str, object] = {
        "out_name": report.out_name,
        "created_at": report.created_at.isoformat(),
        "git_sha": report.git_sha,
        "per_strategy_experiment_id": dict(report.per_strategy_experiment_id),
        "per_strategy_stats": {
            name: stats.to_dict() for name, stats in report.per_strategy_stats.items()
        },
    }
    if report.per_strategy_per_regime_stats is not None:
        payload["per_strategy_per_regime_stats"] = {
            name: {label: stats.to_dict() for label, stats in per_regime.items()}
            for name, per_regime in report.per_strategy_per_regime_stats.items()
        }
    return payload


def _ordered_regime_labels(
    per_strategy_per_regime_stats: Mapping[str, Mapping[str, AggregateStats]],
) -> list[str]:
    """Union of regime labels seen, sorted with ``mixed`` pinned last."""
    seen = {label for per_regime in per_strategy_per_regime_stats.values() for label in per_regime}
    real = sorted(label for label in seen if label != MIXED_REGIME_LABEL)
    if MIXED_REGIME_LABEL in seen:
        real.append(MIXED_REGIME_LABEL)
    return real


def _build_strategy_x_regime_df(
    per_strategy_per_regime_stats: Mapping[str, Mapping[str, AggregateStats]],
) -> pd.DataFrame:
    """LaTeX-friendly DataFrame: one row per strategy, one column per regime.

    Cell format mirrors :func:`RegimeReporter._build_summary_df`:
    ``+sharpe_mean ± sharpe_std`` for non-empty regimes, ``--`` for
    regimes the strategy did not encounter (or encountered with zero
    folds after the majority-threshold filter).
    """
    labels = _ordered_regime_labels(per_strategy_per_regime_stats)
    rows: list[dict[str, object]] = []
    for strategy, per_regime in per_strategy_per_regime_stats.items():
        row: dict[str, object] = {"strategy": strategy}
        for label in labels:
            stats = per_regime.get(label)
            if stats is None or stats.n_folds == 0:
                row[label] = "--"
            else:
                row[label] = f"{stats.sharpe_mean:+.3f} $\\pm$ {stats.sharpe_std:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


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
