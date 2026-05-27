"""Render a :class:`ConsolidatedStudyReport` to disk.

Produces the cross-leg artifact tree required by the empirical-study
writeup (master ranking, holdout-vs-dev scatter, strategy × universe
heatmap, etc.). Output goes directly under the study directory the
consolidator was pointed at — alongside the existing ``runs/`` /
``holdout_evals/`` / ``comparisons/`` per-leg trees the orchestrator
wrote, NOT into a nested subdirectory. Callers commit only the
consolidated subset (``manifest.json``, ``tables/``, ``plots/``); the
per-leg ephemera remain gitignored.

This reporter is the visualization-layer counterpart to
:mod:`src.orchestration.study_report`. It does not load anything from
disk other than the per-leg PNG/SVG copies for equity overlays and
holdout equity curves — every scalar already lives on the in-memory
:class:`ConsolidatedStudyReport`.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.metrics_aggregator import AggregateStats
from src.core import json_io
from src.core.logging import get_logger
from src.core.persistence import (
    COMPARISONS_SUBDIR,
    HOLDOUT_EVALS_SUBDIR,
)
from src.orchestration.study import make_leg_id
from src.orchestration.study_report import ConsolidatedStudyReport, HoldoutSnapshot
from src.orchestration.types import PairwiseSignificance
from src.visualization.comparison_reporter import EQUITY_OVERLAY_FILENAME
from src.visualization.holdout_eval_reporter import HOLDOUT_EQUITY_FILENAME
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

_MASTER_RANKING_FILENAME = "master_ranking"
_PER_UNIVERSE_RANKING_FILENAME = "per_universe_ranking"
_HOLDOUT_RESULTS_FILENAME = "holdout_results"
_PAIRWISE_LONG_CSV_FILENAME = "pairwise_significance.csv"
_PAIRWISE_PER_UNIVERSE_SUBDIR = "pairwise_significance"

_STRATEGY_X_UNIVERSE_HEATMAP_FILENAME = "strategy_x_universe_heatmap.png"
_HOLDOUT_DEV_SCATTER_FILENAME = "holdout_dev_scatter.png"

_EQUITY_OVERLAYS_SUBDIR = "per_universe_equity_overlays"
_HOLDOUT_EQUITY_CURVES_SUBDIR = "holdout_equity_curves"


class StudyReportReporter:
    """Consume a :class:`ConsolidatedStudyReport` and write the artifact tree."""

    def generate_full_report(
        self,
        report: ConsolidatedStudyReport,
        out_dir: Path,
        *,
        publish_label: str | None = None,
    ) -> Path:
        """Write every artifact under ``out_dir`` and return ``out_dir``.

        ``publish_label`` overrides ``report.study_name`` in every
        emitted ``\\caption`` and ``\\label``. Pass it for committed
        artifacts referenced from thesis prose so a re-run doesn't churn
        citation slugs.
        """

        out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = out_dir / PLOTS_SUBDIR
        tables_dir = out_dir / TABLES_SUBDIR

        slug = (
            validate_publish_label(publish_label)
            if publish_label is not None
            else report.study_name
        )

        _logger.info(
            "consolidating study '%s': %d strategies, %d universes, %d completed legs, "
            "%d incomplete legs",
            report.study_name,
            len(report.strategies),
            len(report.universes),
            len(report.per_leg_aggregate),
            len(report.incomplete_leg_ids),
        )

        json_io.write(out_dir / MANIFEST_FILENAME, _build_manifest_dict(report, slug=slug))

        ranking_df = _build_ranking_df(report.per_leg_aggregate)
        self._write_master_ranking(ranking_df, tables_dir, slug=slug)
        self._write_per_universe_ranking(ranking_df, tables_dir, slug=slug)
        self._write_holdout_results(report, tables_dir, slug=slug)
        self._write_pairwise_significance(report, tables_dir, slug=slug)

        self._plot_strategy_x_universe_heatmap(report, plots_dir)
        self._plot_holdout_dev_scatter(report, plots_dir)
        self._copy_equity_overlays(report, plots_dir)
        self._copy_holdout_equity_curves(report, plots_dir)

        return out_dir

    def _write_master_ranking(
        self, ranking_df: pd.DataFrame, tables_dir: Path, *, slug: str
    ) -> None:
        if ranking_df.empty:
            _logger.info("no completed legs — skipping master_ranking")
            return

        df = ranking_df.sort_values("sharpe_mean", ascending=False).reset_index(drop=True)
        _write_table_pair(
            df,
            tables_dir,
            stem=_MASTER_RANKING_FILENAME,
            caption=f"Master ranking across all (strategy, universe) legs — {slug}",
            label=f"tab:master_ranking_{slug}",
        )

    def _write_per_universe_ranking(
        self, ranking_df: pd.DataFrame, tables_dir: Path, *, slug: str
    ) -> None:
        if ranking_df.empty:
            return

        df = ranking_df.sort_values(["universe", "sharpe_mean"], ascending=[True, False])
        df = df.reset_index(drop=True)
        _write_table_pair(
            df,
            tables_dir,
            stem=_PER_UNIVERSE_RANKING_FILENAME,
            caption=f"Per-universe ranking, sorted by Sharpe within each universe — {slug}",
            label=f"tab:per_universe_ranking_{slug}",
        )

    def _write_holdout_results(
        self, report: ConsolidatedStudyReport, tables_dir: Path, *, slug: str
    ) -> None:
        if not report.per_leg_holdout:
            _logger.info("no holdout-eval bundles on any leg — skipping holdout_results")
            return
        df = _build_holdout_df(report.per_leg_holdout, report.per_leg_aggregate)
        if df.empty:
            return
        df = df.sort_values("holdout_sharpe", ascending=False).reset_index(drop=True)
        _write_table_pair(
            df,
            tables_dir,
            stem=_HOLDOUT_RESULTS_FILENAME,
            caption=f"Honest out-of-sample (holdout) vs. dev metrics, per leg — {slug}",
            label=f"tab:holdout_results_{slug}",
        )

    def _write_pairwise_significance(
        self, report: ConsolidatedStudyReport, tables_dir: Path, *, slug: str
    ) -> None:
        if not report.per_universe_pairwise:
            _logger.info(
                "no pairwise comparisons on disk — skipping pairwise_significance "
                "(every universe was single-strategy or `--skip-compares` was set)"
            )
            return
        long_df = _build_pairwise_long_df(report.per_universe_pairwise)
        long_df.to_csv(tables_dir / _PAIRWISE_LONG_CSV_FILENAME, index=False)

        per_universe_dir = tables_dir / _PAIRWISE_PER_UNIVERSE_SUBDIR
        per_universe_dir.mkdir(parents=True, exist_ok=True)
        for universe, pairs in sorted(report.per_universe_pairwise.items()):
            matrix_df = _build_pairwise_matrix_df(pairs)
            write_booktabs_table(
                matrix_df,
                per_universe_dir / f"{universe}.tex",
                caption=(
                    f"Pairwise Sharpe differential, 95\\% bootstrap CI "
                    f"(row $-$ col) — universe {universe}, study {slug}"
                ),
                label=f"tab:pairwise_{slug}_{universe}",
            )

    def _plot_strategy_x_universe_heatmap(
        self, report: ConsolidatedStudyReport, plots_dir: Path
    ) -> None:
        if not report.per_leg_aggregate:
            return

        strategies = report.strategies
        universes = report.universes
        matrix = np.full((len(strategies), len(universes)), np.nan, dtype=np.float64)
        for i, strategy in enumerate(strategies):
            for j, universe in enumerate(universes):
                stats = report.per_leg_aggregate.get((strategy, universe))
                if stats is None or stats.n_folds == 0:
                    continue
                matrix[i, j] = stats.sharpe_mean

        render_value_heatmap(
            matrix,
            row_labels=strategies,
            col_labels=universes,
            out_path=plots_dir / _STRATEGY_X_UNIVERSE_HEATMAP_FILENAME,
            title="strategy × universe (mean Sharpe)",
            xlabel="universe",
            ylabel="strategy",
            placeholder_log_label="strategy × universe",
        )

    def _plot_holdout_dev_scatter(self, report: ConsolidatedStudyReport, plots_dir: Path) -> None:
        if not report.per_leg_holdout:
            return

        out_path = plots_dir / _HOLDOUT_DEV_SCATTER_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)

        by_strategy: dict[str, tuple[list[float], list[float]]] = {}
        all_xs: list[float] = []
        all_ys: list[float] = []
        for (strategy, universe), holdout in report.per_leg_holdout.items():
            dev = report.per_leg_aggregate.get((strategy, universe))
            if dev is None or dev.n_folds == 0:
                continue
            xs, ys = by_strategy.setdefault(strategy, ([], []))
            xs.append(dev.sharpe_mean)
            ys.append(holdout.sharpe_ratio)
            all_xs.append(dev.sharpe_mean)
            all_ys.append(holdout.sharpe_ratio)

        cmap = plt.get_cmap("tab10")
        for idx, strategy in enumerate(sorted(by_strategy)):
            xs, ys = by_strategy[strategy]
            ax.scatter(xs, ys, label=strategy, color=cmap(idx % 10), s=40, alpha=0.8)

        # y=x reference: dev=holdout. Anchor to actual data extents — the
        # matplotlib auto-limit is unset before the first scatter call and
        # legitimately includes 0.0 once data lands, so deriving from the
        # plotted xs/ys avoids clobbering a real lower bound.
        if all_xs:
            finite_lo = min(min(all_xs), min(all_ys))
            finite_hi = max(max(all_xs), max(all_ys))
        else:
            finite_lo, finite_hi = -1.0, 1.0

        ax.plot(
            [finite_lo, finite_hi],
            [finite_lo, finite_hi],
            color="black",
            linewidth=0.5,
            alpha=0.5,
            linestyle="--",
        )
        ax.axhline(0.0, color="grey", linewidth=0.3, alpha=0.5)
        ax.axvline(0.0, color="grey", linewidth=0.3, alpha=0.5)
        ax.set_xlabel("dev mean Sharpe (walk-forward)")
        ax.set_ylabel("holdout Sharpe (single OOS window)")
        ax.set_title("dev vs. holdout Sharpe — points below dashed line = overfit gap")
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        save_png_and_svg(fig, out_path)
        plt.close(fig)

    def _copy_equity_overlays(self, report: ConsolidatedStudyReport, plots_dir: Path) -> None:
        dest_dir = plots_dir / _EQUITY_OVERLAYS_SUBDIR
        for universe in report.universes:
            src_png = (
                report.study_dir
                / COMPARISONS_SUBDIR
                / universe
                / PLOTS_SUBDIR
                / EQUITY_OVERLAY_FILENAME
            )
            _copy_per_leg_artifact(src_png, dest_dir / f"{universe}.png", missing_label=universe)

    def _copy_holdout_equity_curves(self, report: ConsolidatedStudyReport, plots_dir: Path) -> None:
        if not report.per_leg_holdout:
            return

        dest_dir = plots_dir / _HOLDOUT_EQUITY_CURVES_SUBDIR
        for strategy, universe in sorted(report.per_leg_holdout):
            leg_id = make_leg_id(strategy, universe)
            src_png = (
                report.study_dir
                / HOLDOUT_EVALS_SUBDIR
                / leg_id
                / PLOTS_SUBDIR
                / HOLDOUT_EQUITY_FILENAME
            )
            _copy_per_leg_artifact(src_png, dest_dir / f"{leg_id}.png", missing_label=str(src_png))


def _write_table_pair(
    df: pd.DataFrame,
    tables_dir: Path,
    *,
    stem: str,
    caption: str,
    label: str,
) -> None:
    """Write the same DataFrame as a booktabs ``.tex`` and a flat ``.csv``.

    The two formats serve different consumers: ``.tex`` for direct LaTeX
    inclusion in the writeup; ``.csv`` for re-pivoting in pandas at
    writeup time without re-running the consolidator.
    """

    tables_dir.mkdir(parents=True, exist_ok=True)
    write_booktabs_table(df, tables_dir / f"{stem}.tex", caption=caption, label=label)
    df.to_csv(tables_dir / f"{stem}.csv", index=False)


def _build_ranking_df(
    per_leg_aggregate: Mapping[tuple[str, str], AggregateStats],
) -> pd.DataFrame:
    """Build a long-form DataFrame: one row per (strategy, universe) leg."""

    rows: list[dict[str, object]] = []
    for (strategy, universe), stats in per_leg_aggregate.items():
        if stats.n_folds == 0:
            continue
        rows.append(
            {
                "strategy": strategy,
                "universe": universe,
                "n_folds": stats.n_folds,
                "sharpe_mean": stats.sharpe_mean,
                "sharpe_std": stats.sharpe_std,
                "sortino_mean": stats.sortino_mean,
                "calmar_mean": stats.calmar_mean,
                "max_drawdown_worst": stats.max_drawdown_worst,
                "total_return_mean": stats.total_return_mean,
                "win_rate_mean": stats.win_rate_mean,
                "trade_count_total": stats.trade_count_total,
            }
        )
    return pd.DataFrame(rows)


def _build_holdout_df(
    per_leg_holdout: Mapping[tuple[str, str], HoldoutSnapshot],
    per_leg_aggregate: Mapping[tuple[str, str], AggregateStats],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (strategy, universe), holdout in per_leg_holdout.items():
        dev = per_leg_aggregate.get((strategy, universe))
        dev_sharpe = dev.sharpe_mean if dev is not None and dev.n_folds > 0 else float("nan")
        rows.append(
            {
                "strategy": strategy,
                "universe": universe,
                "dev_sharpe": dev_sharpe,
                "holdout_sharpe": holdout.sharpe_ratio,
                "holdout_sortino": holdout.sortino_ratio,
                "holdout_calmar": holdout.calmar_ratio,
                "holdout_max_drawdown": holdout.max_drawdown,
                "holdout_total_return": holdout.total_return,
                "holdout_win_rate": holdout.win_rate,
                "holdout_trades": holdout.trade_count,
                "holdout_start": holdout.holdout_start.isoformat(),
                "n_dev_bars": holdout.n_dev_bars,
                "n_holdout_bars": holdout.n_holdout_bars,
            }
        )
    return pd.DataFrame(rows)


def _build_pairwise_long_df(
    per_universe_pairwise: Mapping[str, tuple[PairwiseSignificance, ...]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for universe, pairs in per_universe_pairwise.items():
        for p in pairs:
            rows.append(
                {
                    "universe": universe,
                    "strategy_a": p.name_a,
                    "strategy_b": p.name_b,
                    "point_differential": p.point_differential,
                    "ci_low": p.lower,
                    "ci_high": p.upper,
                    "confidence": p.confidence,
                    "significant": p.significant,
                }
            )
    return pd.DataFrame(rows)


def _build_pairwise_matrix_df(pairs: tuple[PairwiseSignificance, ...]) -> pd.DataFrame:
    """Upper-triangular matrix DataFrame: ``point [low, high]*`` per cell.

    Mirrors :meth:`ComparisonReporter._build_pairwise_table` so the
    consolidated per-universe ``.tex`` matches the per-comparison
    ``.tex`` already under ``comparisons/<universe>/tables/`` cell-for-cell.
    """

    names = list(dict.fromkeys(name for p in pairs for name in (p.name_a, p.name_b)))
    df = pd.DataFrame("", index=names, columns=names)
    lookup = {(p.name_a, p.name_b): p for p in pairs}
    for i, row in enumerate(names):
        for j, col in enumerate(names):
            if i >= j:
                continue
            p = lookup.get((row, col))
            if p is None:
                continue
            star = "*" if p.significant else ""
            df.at[row, col] = f"{p.point_differential:+.3f} [{p.lower:+.3f}, {p.upper:+.3f}]{star}"
    df.insert(0, "strategy", df.index)
    df = df.reset_index(drop=True)
    return df


def _copy_per_leg_artifact(src_png: Path, dst_png: Path, *, missing_label: str) -> None:
    """Copy a PNG (and sibling SVG when present) from a per-leg artifact tree.

    Skips with an info log when ``src_png`` does not exist — single-strategy
    universes have no equity overlay; older runs may lack the SVG twin.
    """

    try:
        dst_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_png, dst_png)
    except FileNotFoundError:
        _logger.info("source artifact missing at %s — skipping copy", missing_label)
        return
    src_svg = src_png.with_suffix(".svg")
    dst_svg = dst_png.with_suffix(".svg")
    try:
        shutil.copyfile(src_svg, dst_svg)
    except FileNotFoundError:
        _logger.info("source SVG missing at %s — copied PNG only", src_svg)


def _build_manifest_dict(report: ConsolidatedStudyReport, *, slug: str) -> dict[str, object]:
    """Identity sidecar for the consolidated tree.

    Lists let a reader sanity-check coverage without walking the per-leg
    tree; ``per_leg_run_id`` maps leg ids back to their source run dirs.
    """

    return {
        "study_name": report.study_name,
        "publish_label": slug,
        "study_dir": str(report.study_dir),
        "created_at": report.created_at.isoformat(),
        "git_sha": report.git_sha,
        "strategies": list(report.strategies),
        "universes": list(report.universes),
        "incomplete_leg_ids": list(report.incomplete_leg_ids),
        "per_leg_run_id": {
            make_leg_id(strategy, universe): run_id
            for (strategy, universe), run_id in report.per_leg_run_id.items()
        },
        "n_legs_with_holdout": len(report.per_leg_holdout),
        "n_universes_with_pairwise": len(report.per_universe_pairwise),
    }


__all__ = ["StudyReportReporter"]
