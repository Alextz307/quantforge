"""Per-regime analysis report generator.

Consumes an in-memory :class:`RegimeReport` and writes:

* ``manifest.json``                   — scalar identity (out_name, exp_id,
                                        kind, detector_name, timestamp,
                                        git sha) + per-regime stats dicts +
                                        which fold indices contributed to
                                        which regime.
* ``tables/regime_summary.tex``       — booktabs LaTeX, one row per regime
                                        label (incl. ``mixed``), columns
                                        for the headline metrics.
* ``plots/regime_metric_heatmap.png/svg`` — regime × metric matrix
                                        rendered via ``imshow``, viridis
                                        colormap, NaN cells masked grey
                                        for empty regimes.
* ``plots/regime_timeline.png/svg``   — full-range bar tape coloured by
                                        regime label (one rectangle per
                                        contiguous slice). Useful for
                                        visualising the detector's split
                                        before drilling into per-regime
                                        stats.

Mirrors :class:`ComparisonReporter`'s structure (same plots/ + tables/
+ manifest.json layout) so a user who's used one knows the other
instantly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.analysis.metrics_aggregator import AggregateStats
from src.core import json_io
from src.core.logging import get_logger
from src.orchestration.types import MIXED_REGIME_LABEL, RegimeReport, RegimeSlice
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

_SUMMARY_FILENAME = "regime_summary.tex"
_HEATMAP_FILENAME = "regime_metric_heatmap.png"
TIMELINE_FILENAME = "regime_timeline.png"

# Heatmap column metric keys — mean fields on AggregateStats. Display
# names are LaTeX-friendly (escape=False is on).
_HEATMAP_METRICS: tuple[tuple[str, str], ...] = (
    ("sharpe_mean", "Sharpe"),
    ("sortino_mean", "Sortino"),
    ("calmar_mean", "Calmar"),
    ("max_drawdown_mean", "MaxDD"),
    ("win_rate_mean", "Win rate"),
    ("total_return_mean", "Return"),
)


class RegimeReporter:
    """Generate the per-regime report bundle."""

    def generate_full_report(
        self,
        report: RegimeReport,
        out_dir: Path,
        *,
        slices: Sequence[RegimeSlice] | None = None,
        publish_label: str | None = None,
    ) -> Path:
        """Write every artifact under ``out_dir`` and return ``out_dir``.

        ``slices`` overrides ``report.slices`` for the timeline plot —
        useful when the caller already has the full-range slices in hand
        from the detector and wants to render them without round-tripping
        through ``RegimeReport.slices``. Defaults to ``report.slices``.

        ``publish_label`` overrides the volatile ``experiment_id`` /
        ``out_name`` combination in the regime table's ``\\caption`` and
        ``\\label``; pass it when the .tex is referenced from prose so a
        rerun doesn't churn the citation slug.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = out_dir / PLOTS_SUBDIR
        tables_dir = out_dir / TABLES_SUBDIR

        n_real_regimes = len(report.per_regime_stats) - (
            1 if MIXED_REGIME_LABEL in report.per_regime_stats else 0
        )
        _logger.info(
            "generating regime report '%s' for experiment %s with %d regime(s) + %d mixed fold(s)",
            report.out_name,
            report.experiment_id,
            n_real_regimes,
            len(report.mixed_fold_indices),
        )

        if publish_label is not None:
            slug = validate_publish_label(publish_label)
            caption = f"Per-regime walk-forward summary — {slug} ({report.detector_name})"
            label = f"tab:regime_{slug}"
        else:
            caption = (
                f"Per-regime walk-forward summary — experiment "
                f"{report.experiment_id} ({report.detector_name})"
            )
            label = f"tab:regime_{report.out_name}"

        json_io.write(out_dir / MANIFEST_FILENAME, _build_manifest_dict(report))

        write_booktabs_table(
            _build_summary_df(report.per_regime_stats),
            tables_dir / _SUMMARY_FILENAME,
            caption=caption,
            label=label,
        )

        self._plot_metric_heatmap(report.per_regime_stats, plots_dir / _HEATMAP_FILENAME)

        timeline_slices = tuple(slices) if slices is not None else report.slices
        if timeline_slices:
            self._plot_timeline(timeline_slices, plots_dir / TIMELINE_FILENAME)
        else:
            _logger.info("skipping regime timeline: no slices available")

        return out_dir

    def _plot_metric_heatmap(
        self,
        per_regime_stats: Mapping[str, AggregateStats],
        out_path: Path,
    ) -> Path:
        labels = list(per_regime_stats)
        if not labels:
            _logger.warning("no regime labels in report — skipping heatmap")
            return out_path

        matrix = np.full((len(labels), len(_HEATMAP_METRICS)), np.nan, dtype=np.float64)
        for i, label in enumerate(labels):
            stats = per_regime_stats[label]
            if stats.n_folds == 0:
                continue
            for j, (key, _) in enumerate(_HEATMAP_METRICS):
                matrix[i, j] = float(getattr(stats, key))

        return render_value_heatmap(
            matrix,
            row_labels=labels,
            col_labels=[display for _, display in _HEATMAP_METRICS],
            out_path=out_path,
            title="regime × metric",
            xlabel="metric",
            ylabel="regime",
            placeholder_log_label="regime",
        )

    def _plot_timeline(
        self,
        slices: Sequence[RegimeSlice],
        out_path: Path,
    ) -> Path:
        """Render the bar-tape timeline coloured by regime label.

        Uses :meth:`matplotlib.axes.Axes.axvspan` per slice so matplotlib
        handles the datetime → numeric conversion automatically. The
        legend lists each unique label once via a synthetic
        :class:`~matplotlib.patches.Patch` per label so a regime that
        recurs multiple times still gets a single legend entry.
        ``unclassified`` (warmup) renders in light grey.
        """
        unique_labels = list(dict.fromkeys(s.label for s in slices))
        cmap = plt.get_cmap("tab10")
        colour_lookup: dict[str, tuple[float, float, float, float]] = {}
        for i, label in enumerate(unique_labels):
            if label == "unclassified":
                colour_lookup[label] = (0.85, 0.85, 0.85, 1.0)
            else:
                colour_lookup[label] = cmap(i % cmap.N)

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN * 0.5), dpi=FIGURE_DPI)
        for sl in slices:
            ax.axvspan(
                _ts_to_mpl(sl.start),
                _ts_to_mpl(sl.end),
                facecolor=colour_lookup[sl.label],
                alpha=0.85,
            )
        ax.set_xlim(_ts_to_mpl(slices[0].start), _ts_to_mpl(slices[-1].end))
        ax.xaxis_date()
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_xlabel("date")
        ax.set_title("regime timeline")
        legend_handles = [
            Patch(facecolor=colour_lookup[label], label=label) for label in unique_labels
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(4, len(unique_labels)),
        )
        fig.autofmt_xdate()
        fig.tight_layout()
        save_png_and_svg(fig, out_path)
        plt.close(fig)
        return out_path


def _build_summary_df(per_regime_stats: Mapping[str, AggregateStats]) -> pd.DataFrame:
    """Per-regime LaTeX table — one row per label, headline metrics.

    Columns: ``regime``, ``n_folds``, ``sharpe`` (mean ± std),
    ``sortino``, ``calmar``, ``max_dd``. Empty-regime rows show
    ``n_folds=0`` and dashes for the metrics so the reader can tell at a
    glance which regimes had no folds in the run.
    """
    rows: list[dict[str, object]] = []
    for label, stats in per_regime_stats.items():
        if stats.n_folds == 0:
            rows.append(
                {
                    "regime": label,
                    "n_folds": 0,
                    "sharpe": "--",
                    "sortino": "--",
                    "calmar": "--",
                    "max_dd": "--",
                }
            )
            continue
        rows.append(
            {
                "regime": label,
                "n_folds": stats.n_folds,
                "sharpe": f"{stats.sharpe_mean:+.3f} $\\pm$ {stats.sharpe_std:.3f}",
                "sortino": f"{stats.sortino_mean:+.3f} $\\pm$ {stats.sortino_std:.3f}",
                "calmar": f"{stats.calmar_mean:+.3f} $\\pm$ {stats.calmar_std:.3f}",
                "max_dd": f"{stats.max_drawdown_worst:+.3f}",
            }
        )
    return pd.DataFrame(rows)


def _build_manifest_dict(report: RegimeReport) -> dict[str, object]:
    """Flatten :class:`RegimeReport` for ``manifest.json``."""
    return {
        "out_name": report.out_name,
        "experiment_id": report.experiment_id,
        "kind": report.kind.value,
        "detector_name": report.detector_name,
        "created_at": report.created_at.isoformat(),
        "git_sha": report.git_sha,
        "per_regime_stats": {
            label: stats.to_dict() for label, stats in report.per_regime_stats.items()
        },
        "per_regime_fold_indices": {
            label: list(indices) for label, indices in report.per_regime_fold_indices.items()
        },
        "mixed_fold_indices": list(report.mixed_fold_indices),
        "slices": [s.to_dict() for s in report.slices],
    }


def _ts_to_mpl(ts: pd.Timestamp) -> float:
    """Convert a ``pd.Timestamp`` to matplotlib's float-day axis units.

    Wraps :func:`matplotlib.dates.date2num` (which the matplotlib stubs
    leave untyped) so the call site stays strict-typed without scattering
    ``cast(float, ...)`` across the plotting code.
    """
    return cast(float, mdates.date2num(ts.to_pydatetime()))  # type: ignore[no-untyped-call]


__all__ = ["RegimeReporter"]
