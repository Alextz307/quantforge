"""
Reporter for one-shot holdout-eval bundles.

Consumes a :class:`HoldoutEvalResult` and emits two artifacts under the
holdout-eval out_dir:

* ``tables/holdout_metrics.tex`` — booktabs LaTeX, two-column "metric ·
  value" layout. Single-row metric set covers Sharpe / Sortino / Calmar
  / MaxDD / Total return / Win rate / Trades — the same scalars the
  per-fold ``StrategyReporter`` table emits, just collapsed to one window.
* ``plots/holdout_equity.png/svg`` — equity curve over the holdout window
  with the dev/holdout boundary annotated. Normalised to 1.0 at holdout
  start so the y-axis tells the OOS story (return vs. baseline) rather
  than the absolute capital level.

Output goes alongside the ``holdout_eval.json`` payload — the holdout-eval
orchestration module writes the JSON, this reporter writes everything else.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.core.logging import get_logger
from src.orchestration.holdout_eval import HoldoutEvalResult
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

_METRICS_FILENAME = "holdout_metrics.tex"
HOLDOUT_EQUITY_FILENAME = "holdout_equity.png"


class HoldoutEvalReporter:
    """
    Generate the holdout-eval table + equity plot.
    """

    def generate_full_report(
        self,
        result: HoldoutEvalResult,
        out_dir: Path,
        *,
        publish_label: str | None = None,
    ) -> Path:
        """
        Write both artifacts under ``out_dir/{plots,tables}/``.

        ``out_dir`` is the same directory the orchestration module wrote
        ``holdout_eval.json`` into, so the caller passes the path
        returned from :func:`run_holdout_eval`.

        ``publish_label`` overrides ``result.source_id`` /
        ``result.out_name`` in the table's ``\\caption`` and ``\\label``;
        pass it when committed holdout artifacts are referenced from
        prose so a re-run doesn't churn the citation slug.
        """

        out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = out_dir / PLOTS_SUBDIR
        tables_dir = out_dir / TABLES_SUBDIR

        if publish_label is not None:
            slug = validate_publish_label(publish_label)
            caption = f"Holdout-eval metrics — {slug} (boundary {result.holdout_start.isoformat()})"
            label = f"tab:holdout_{slug}"
        else:
            caption = (
                f"Holdout-eval metrics — source {result.source_kind}: "
                f"{result.source_id} (boundary {result.holdout_start.isoformat()})"
            )
            label = f"tab:holdout_{result.out_name}"

        write_booktabs_table(
            _build_metrics_df(result),
            tables_dir / _METRICS_FILENAME,
            caption=caption,
            label=label,
        )

        self._plot_equity_curve(result, plots_dir / HOLDOUT_EQUITY_FILENAME)
        return out_dir

    def _plot_equity_curve(self, result: HoldoutEvalResult, out_path: Path) -> Path:
        """
        Plot the holdout equity curve normalised to 1.0 at holdout start.

        Skips plotting (with a warning) if the equity series is empty or
        starts at a non-finite / non-positive value — same defence the
        per-fold reporter uses (NaN propagates through matplotlib silently
        and a non-positive base inverts the visual narrative).
        """

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)
        normalised = normalise_to_unit_base(result.equity_curve)
        if normalised is None:
            _logger.warning(
                "holdout-eval %s: equity_curve[0]=%s is non-finite or non-positive — "
                "rendering empty equity plot",
                result.out_name,
                result.equity_curve[0] if result.equity_curve else "empty",
            )
        else:
            ax.plot(normalised, linewidth=1.2, color="#1f77b4")
            ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(f"bar index in holdout (boundary {result.holdout_start.date()})")
        ax.set_ylabel("equity (normalised to holdout start)")
        ax.set_title(f"holdout equity — {result.source_kind}: {result.source_id}")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        save_png_and_svg(fig, out_path)
        plt.close(fig)
        return out_path


def _build_metrics_df(result: HoldoutEvalResult) -> pd.DataFrame:
    """
    Two-column "metric · value" frame.

    A horizontal one-row layout would crowd the LaTeX page (8 columns at
    .3f); the two-column form reads top-to-bottom and fits inside a
    half-text-width float. The Sharpe CI and buy-and-hold reference are
    appended after the strategy block so a reader can visually compare
    the strategy's signal against the universe's long-only baseline.
    """

    bah = result.buy_and_hold
    excess_sharpe = result.sharpe_ratio - bah.sharpe_ratio
    excess_total = result.total_return - bah.total_return
    confidence_pct = int(round(result.sharpe_ci.confidence * 100))
    rows: list[tuple[str, str]] = [
        ("Sharpe", f"{result.sharpe_ratio:+.3f}"),
        (
            f"Sharpe {confidence_pct}\\% CI",
            f"[{result.sharpe_ci.lower:+.3f}, {result.sharpe_ci.upper:+.3f}]",
        ),
        ("Sortino", f"{result.sortino_ratio:+.3f}"),
        ("Calmar", f"{result.calmar_ratio:+.3f}"),
        ("Max drawdown", f"{result.max_drawdown:+.3f}"),
        ("Total return", f"{result.total_return:+.3f}"),
        ("Annualised return", f"{result.annualized_return:+.3f}"),
        ("Annualised volatility", f"{result.annualized_volatility:.3f}"),
        ("Win rate", f"{result.win_rate:.3f}"),
        ("Trades", str(result.trade_count)),
        ("Holdout bars", str(result.n_holdout_bars)),
        ("Dev bars", str(result.n_dev_bars)),
        ("Buy-and-hold Sharpe", f"{bah.sharpe_ratio:+.3f}"),
        ("Buy-and-hold total return", f"{bah.total_return:+.3f}"),
        ("Buy-and-hold max drawdown", f"{bah.max_drawdown:+.3f}"),
        ("Excess Sharpe (vs BAH)", f"{excess_sharpe:+.3f}"),
        ("Excess total return (vs BAH)", f"{excess_total:+.3f}"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


__all__ = ["HoldoutEvalReporter"]
