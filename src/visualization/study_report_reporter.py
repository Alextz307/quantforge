"""
Render a :class:`ConsolidatedStudyReport` to disk.

Produces the cross-leg artifact tree required by the empirical-study
writeup (master ranking, holdout-vs-dev scatter, strategy x universe
heatmap, etc.). Output goes directly under the study directory the
consolidator was pointed at - alongside the existing ``runs/`` /
``holdout_evals/`` / ``comparisons/`` per-leg trees the orchestrator
wrote, NOT into a nested subdirectory. Callers commit only the
consolidated subset (``manifest.json``, ``tables/``, ``plots/``); the
per-leg ephemera remain gitignored.

This reporter is the visualization-layer counterpart to
:mod:`src.orchestration.study_report`. It does not load anything from
disk other than the per-leg PNG/SVG copies for equity overlays and
holdout equity curves - every scalar already lives on the in-memory
:class:`ConsolidatedStudyReport`.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.feature_importance import AggregatedImportance, ImportanceMethod
from src.analysis.metrics_aggregator import AggregateStats
from src.analysis.significance import DeflatedSharpe
from src.core import json_io
from src.core.fs import atomic_write_path
from src.core.logging import get_logger
from src.core.persistence import (
    COMPARISONS_SUBDIR,
    HOLDOUT_EVALS_SUBDIR,
)
from src.orchestration.study import make_leg_id
from src.orchestration.study_report import (
    ConsolidatedStudyReport,
    FloorBindStats,
    HoldoutSnapshot,
)
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
_FLOOR_BIND_FILENAME = "floor_bind_by_leg"
_PAIRWISE_LONG_CSV_FILENAME = "pairwise_significance.csv"
_PAIRWISE_PER_UNIVERSE_SUBDIR = "pairwise_significance"

_STRATEGY_X_UNIVERSE_HEATMAP_FILENAME = "strategy_x_universe_heatmap.png"
_HOLDOUT_DEV_SCATTER_FILENAME = "holdout_dev_scatter.png"
_FEATURE_IMPORTANCE_HEATMAP_FILENAME = "feature_importance_heatmap.png"

_EQUITY_OVERLAYS_SUBDIR = "per_universe_equity_overlays"
_HOLDOUT_EQUITY_CURVES_SUBDIR = "holdout_equity_curves"
_FEATURE_IMPORTANCE_BARS_SUBDIR = "feature_importance"


class StudyReportReporter:
    """
    Consume a :class:`ConsolidatedStudyReport` and write the artifact tree.
    """

    def generate_full_report(
        self,
        report: ConsolidatedStudyReport,
        out_dir: Path,
        *,
        publish_label: str | None = None,
    ) -> Path:
        """
        Write every artifact under ``out_dir`` and return ``out_dir``.

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

        ranking_df = _build_ranking_df(report.per_leg_aggregate, report.per_leg_dsr)
        self._write_master_ranking(ranking_df, tables_dir, slug=slug)
        self._write_per_universe_ranking(ranking_df, tables_dir, slug=slug)
        self._write_holdout_results(report, tables_dir, slug=slug)
        self._write_floor_bind_by_leg(report, tables_dir, slug=slug)
        self._write_pairwise_significance(report, tables_dir, slug=slug)

        self._plot_strategy_x_universe_heatmap(report, plots_dir)
        self._plot_holdout_dev_scatter(report, plots_dir)
        self._plot_feature_importance(report, plots_dir)
        self._copy_equity_overlays(report, plots_dir)
        self._copy_holdout_equity_curves(report, plots_dir)

        return out_dir

    def _write_master_ranking(
        self, ranking_df: pd.DataFrame, tables_dir: Path, *, slug: str
    ) -> None:
        if ranking_df.empty:
            _logger.info("no completed legs - skipping master_ranking")
            return

        df = ranking_df.sort_values("sharpe_mean", ascending=False).reset_index(drop=True)
        _write_table_pair(
            df,
            tables_dir,
            stem=_MASTER_RANKING_FILENAME,
            caption=f"Master ranking across all (strategy, universe) legs - {slug}",
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
            caption=f"Per-universe ranking, sorted by Sharpe within each universe - {slug}",
            label=f"tab:per_universe_ranking_{slug}",
        )

    def _write_holdout_results(
        self, report: ConsolidatedStudyReport, tables_dir: Path, *, slug: str
    ) -> None:
        if not report.per_leg_holdout:
            _logger.info("no holdout-eval bundles on any leg - skipping holdout_results")
            return
        df = _build_holdout_df(report.per_leg_holdout, report.per_leg_aggregate)
        if df.empty:
            return
        df = df.sort_values("holdout_sharpe", ascending=False).reset_index(drop=True)
        _write_table_pair(
            df,
            tables_dir,
            stem=_HOLDOUT_RESULTS_FILENAME,
            caption=f"Honest out-of-sample (holdout) vs. dev metrics, per leg - {slug}",
            label=f"tab:holdout_results_{slug}",
        )

    def _write_floor_bind_by_leg(
        self, report: ConsolidatedStudyReport, tables_dir: Path, *, slug: str
    ) -> None:
        """
        Write the sigma_min floor-saturation table for legs that emit the diagnostic.

        Only VolatilityTargeting legs populate ``per_leg_floor_bind``; when
        the map is empty the section is skipped without emitting an empty
        table (LaTeX builds reject zero-row long tables).
        """

        if not report.per_leg_floor_bind:
            _logger.info(
                "no leg emitted floor_bind_fraction - skipping floor_bind_by_leg "
                "(non-VolatilityTargeting sweep)"
            )
            return
        df = _build_floor_bind_df(report.per_leg_floor_bind)
        if df.empty:
            return
        df = df.sort_values("floor_bind_mean", ascending=False).reset_index(drop=True)
        _write_table_pair(
            df,
            tables_dir,
            stem=_FLOOR_BIND_FILENAME,
            caption=(
                f"VolatilityTargeting $\\sigma_{{\\min}}$ floor-saturation fractions "
                f"per leg - {slug}"
            ),
            label=f"tab:floor_bind_{slug}",
        )

        payload = {
            make_leg_id(strategy, universe): stats.to_dict()
            for (strategy, universe), stats in report.per_leg_floor_bind.items()
        }
        json_io.write(tables_dir / f"{_FLOOR_BIND_FILENAME}.json", payload)

    def _write_pairwise_significance(
        self, report: ConsolidatedStudyReport, tables_dir: Path, *, slug: str
    ) -> None:
        if not report.per_universe_pairwise:
            _logger.info(
                "no pairwise comparisons on disk - skipping pairwise_significance "
                "(every universe was single-strategy or `--skip-compares` was set)"
            )
            return
        long_df = _build_pairwise_long_df(report.per_universe_pairwise)
        with atomic_write_path(tables_dir / _PAIRWISE_LONG_CSV_FILENAME) as tmp:
            long_df.to_csv(tmp, index=False)

        per_universe_dir = tables_dir / _PAIRWISE_PER_UNIVERSE_SUBDIR
        per_universe_dir.mkdir(parents=True, exist_ok=True)
        for universe, pairs in sorted(report.per_universe_pairwise.items()):
            matrix_df = _build_pairwise_matrix_df(pairs)
            write_booktabs_table(
                matrix_df,
                per_universe_dir / f"{universe}.tex",
                caption=(
                    f"Pairwise Sharpe differential, 95\\% bootstrap CI "
                    f"(row $-$ col) - universe {universe}, study {slug}"
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
            title="strategy x universe (mean Sharpe)",
            xlabel="universe",
            ylabel="strategy",
            placeholder_log_label="strategy x universe",
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

        # y=x reference: dev=holdout. Anchor to actual data extents - the
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
        ax.set_title("dev vs. holdout Sharpe - points below dashed line = overfit gap")
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        save_png_and_svg(fig, out_path)
        plt.close(fig)

    def _plot_feature_importance(self, report: ConsolidatedStudyReport, plots_dir: Path) -> None:
        """
        Per-strategy permutation-importance bars + a feature x strategy heatmap.

        Uses permutation importance (the model-agnostic method shared by every
        feature-consuming strategy) averaged across each strategy's universes.
        Skipped entirely when no leg emitted a feature-importance artifact
        (rule-based-only sweeps, or importance disabled on the run).
        """

        if not report.per_leg_feature_importance:
            _logger.info(
                "no feature-importance artifacts on any leg - skipping feature-importance plots"
            )
            return
        per_strategy = _permutation_means_by_strategy(report.per_leg_feature_importance)
        per_strategy = {s: means for s, means in per_strategy.items() if means}
        if not per_strategy:
            return

        bars_dir = plots_dir / _FEATURE_IMPORTANCE_BARS_SUBDIR
        bars_dir.mkdir(parents=True, exist_ok=True)
        for strategy in sorted(per_strategy):
            ordered = sorted(per_strategy[strategy].items(), key=lambda kv: kv[1], reverse=True)
            names = [feature for feature, _ in ordered]
            values = [value for _, value in ordered]
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)
            try:
                ax.barh(range(len(names)), values, color="tab:blue")
                # A bar left of 0 marks a feature whose permutation improved the
                # score (it hurt), not merely the least important one.
                ax.axvline(0.0, color="0.4", linewidth=0.8)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names)
                ax.invert_yaxis()
                ax.set_xlabel("permutation importance (mean OOS score drop)")
                ax.set_title(f"feature importance - {strategy}")
                ax.grid(True, axis="x", alpha=0.3)
                fig.tight_layout()
                save_png_and_svg(fig, bars_dir / f"{strategy}.png")
            finally:
                plt.close(fig)

        strategies = sorted(per_strategy)
        features = sorted({feature for means in per_strategy.values() for feature in means})
        matrix = np.full((len(features), len(strategies)), np.nan, dtype=np.float64)
        for j, strategy in enumerate(strategies):
            for i, feature in enumerate(features):
                value = per_strategy[strategy].get(feature)
                if value is not None:
                    matrix[i, j] = value
        render_value_heatmap(
            matrix,
            row_labels=features,
            col_labels=strategies,
            out_path=plots_dir / _FEATURE_IMPORTANCE_HEATMAP_FILENAME,
            title="feature x strategy (mean permutation importance)",
            xlabel="strategy",
            ylabel="feature",
            placeholder_log_label="feature x strategy importance",
        )

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
    """
    Write the same DataFrame as a booktabs ``.tex`` and a flat ``.csv``.

    The two formats serve different consumers: ``.tex`` for direct LaTeX
    inclusion in the writeup; ``.csv`` for re-pivoting in pandas at
    writeup time without re-running the consolidator.
    """

    tables_dir.mkdir(parents=True, exist_ok=True)
    write_booktabs_table(df, tables_dir / f"{stem}.tex", caption=caption, label=label)
    with atomic_write_path(tables_dir / f"{stem}.csv") as tmp:
        df.to_csv(tmp, index=False)


def _build_ranking_df(
    per_leg_aggregate: Mapping[tuple[str, str], AggregateStats],
    per_leg_dsr: Mapping[tuple[str, str], DeflatedSharpe],
) -> pd.DataFrame:
    """
    Build a long-form DataFrame: one row per (strategy, universe) leg.

    Legs that produced a ``dsr.json`` post-tune get three extra columns
    (``deflated_sharpe``, ``dsr_p_value``, ``n_trials``); legs without
    DSR (zero-trial study, single-strategy universe before tune)
    receive ``NaN`` so the column shape stays uniform.
    """

    rows: list[dict[str, object]] = []
    for (strategy, universe), stats in per_leg_aggregate.items():
        if stats.n_folds == 0:
            continue
        dsr = per_leg_dsr.get((strategy, universe))
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
                "deflated_sharpe": dsr.deflated_sharpe if dsr is not None else float("nan"),
                "dsr_p_value": dsr.p_value if dsr is not None else float("nan"),
                "n_trials": dsr.n_trials if dsr is not None else 0,
            }
        )
    return pd.DataFrame(rows)


def _permutation_means_by_strategy(
    per_leg: Mapping[tuple[str, str], tuple[AggregatedImportance, ...]],
) -> dict[str, dict[str, float]]:
    """
    Collapse per-leg importance to ``strategy -> {feature -> mean permutation importance}``.

    Averages the permutation-method importance of each feature across that
    strategy's universes (gain entries are ignored - the cross-strategy view
    uses the one method every feature-consuming strategy shares). NaN values
    (a leg that could not score the feature) are dropped before averaging so
    one failed leg cannot poison a feature's cross-universe mean; a feature
    that is NaN on every universe is omitted entirely rather than shown blank.
    """

    accumulated: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (strategy, _universe), aggs in per_leg.items():
        for agg in aggs:
            if agg.method is ImportanceMethod.PERMUTATION and not np.isnan(agg.importance):
                accumulated[strategy][agg.feature].append(agg.importance)
    return {
        strategy: {feature: float(np.mean(values)) for feature, values in features.items()}
        for strategy, features in accumulated.items()
    }


def _build_holdout_df(
    per_leg_holdout: Mapping[tuple[str, str], HoldoutSnapshot],
    per_leg_aggregate: Mapping[tuple[str, str], AggregateStats],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (strategy, universe), holdout in per_leg_holdout.items():
        dev = per_leg_aggregate.get((strategy, universe))
        dev_sharpe = dev.sharpe_mean if dev is not None and dev.n_folds > 0 else float("nan")
        bah = holdout.buy_and_hold
        rows.append(
            {
                "strategy": strategy,
                "universe": universe,
                "dev_sharpe": dev_sharpe,
                "holdout_sharpe": holdout.sharpe_ratio,
                "holdout_sharpe_ci_low": holdout.sharpe_ci.lower,
                "holdout_sharpe_ci_high": holdout.sharpe_ci.upper,
                "holdout_sortino": holdout.sortino_ratio,
                "holdout_calmar": holdout.calmar_ratio,
                "holdout_max_drawdown": holdout.max_drawdown,
                "holdout_total_return": holdout.total_return,
                "holdout_win_rate": holdout.win_rate,
                "holdout_trades": holdout.trade_count,
                "holdout_start": holdout.holdout_start.isoformat(),
                "n_dev_bars": holdout.n_dev_bars,
                "n_holdout_bars": holdout.n_holdout_bars,
                "bah_sharpe": bah.sharpe_ratio,
                "bah_total_return": bah.total_return,
                "bah_max_drawdown": bah.max_drawdown,
                "excess_sharpe_vs_bah": holdout.sharpe_ratio - bah.sharpe_ratio,
                "excess_total_return_vs_bah": holdout.total_return - bah.total_return,
                "beats_bah": holdout.sharpe_ratio > bah.sharpe_ratio,
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
    """
    Upper-triangular matrix DataFrame: ``point [low, high]*`` per cell.

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


def _build_floor_bind_df(
    per_leg_floor_bind: Mapping[tuple[str, str], FloorBindStats],
) -> pd.DataFrame:
    """
    Long-form floor-saturation table - one row per leg with stats.
    """

    rows: list[dict[str, object]] = []
    for (strategy, universe), stats in per_leg_floor_bind.items():
        rows.append(
            {
                "strategy": strategy,
                "universe": universe,
                "floor_bind_mean": stats.mean,
                "floor_bind_max": stats.max,
                "floor_bind_min": stats.min,
                "n_folds": stats.n_folds,
            }
        )
    return pd.DataFrame(rows)


def _copy_per_leg_artifact(src_png: Path, dst_png: Path, *, missing_label: str) -> None:
    """
    Copy a PNG (and sibling SVG when present) from a per-leg artifact tree.

    Skips with an info log when ``src_png`` does not exist - single-strategy
    universes have no equity overlay; older runs may lack the SVG twin.
    """

    try:
        dst_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_png, dst_png)
    except FileNotFoundError:
        _logger.info("source artifact missing at %s - skipping copy", missing_label)
        return
    src_svg = src_png.with_suffix(".svg")
    dst_svg = dst_png.with_suffix(".svg")
    try:
        shutil.copyfile(src_svg, dst_svg)
    except FileNotFoundError:
        _logger.info("source SVG missing at %s - copied PNG only", src_svg)


def _build_manifest_dict(report: ConsolidatedStudyReport, *, slug: str) -> dict[str, object]:
    """
    Identity sidecar for the consolidated tree.

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
        "n_legs_with_dsr": len(report.per_leg_dsr),
        "n_legs_with_floor_bind": len(report.per_leg_floor_bind),
        "n_legs_with_feature_importance": len(report.per_leg_feature_importance),
        "n_universes_with_pairwise": len(report.per_universe_pairwise),
    }


__all__ = ["StudyReportReporter"]
