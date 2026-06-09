"""
Render a tree strategy's native XGBoost-gain feature importance as a top-N
horizontal bar chart, in the same style as the study report's permutation bars.

The gain values are already persisted per leg in each run's
``feature_importance.json`` (the ``xgb_gain`` method, stored alongside the
``permutation`` entries). This tool aggregates them across a strategy's legs
with the same cross-universe mean the reporter uses for permutation
(``_importance_means_by_strategy``), then draws the strongest ``--top-n``
features. The output lands next to the permutation bars at
``<study-dir>/plots/feature_importance/<Strategy>_gain.{png,svg}`` and is
referenced from the results chapter.

Run as a module (``src`` must be importable):

    python -m scripts.plot_xgb_gain \\
        --study-dir experiment_results/studies/main --strategy MomentumGatekeeper
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt

from src.analysis.feature_importance import ImportanceMethod, read_aggregated_importance

# ``plots`` pins the Agg backend (matplotlib.use("Agg")) at import, so figures
# render headless regardless of the import order above.
from src.visualization.plots import (
    FIGURE_DPI,
    FIGURE_HEIGHT_IN,
    FIGURE_WIDTH_IN,
    save_png_and_svg,
)

DEFAULT_STRATEGY = "MomentumGatekeeper"
DEFAULT_TOP_N = 20
GAIN_BAR_COLOR = "tab:orange"
FEATURE_IMPORTANCE_SUBDIR = "feature_importance"
IMPORTANCE_FILENAME = "feature_importance.json"
GRID_ALPHA = 0.3


def _mean_gain_by_feature(study_dir: Path, strategy: str) -> dict[str, float]:
    """
    Mean XGBoost gain per feature across every leg of ``strategy`` under
    ``study_dir/runs``, mirroring the reporter's cross-universe mean restricted
    to the ``XGB_GAIN`` method. NaN scores are dropped before averaging.
    """

    accumulated: defaultdict[str, list[float]] = defaultdict(list)
    runs_dir = study_dir / "runs"
    n_legs = 0
    for run_dir in sorted(runs_dir.glob(f"*_{strategy}_*")):
        try:
            payload: dict[str, object] = json.loads((run_dir / IMPORTANCE_FILENAME).read_text())
        except FileNotFoundError:
            continue
        n_legs += 1
        for agg in read_aggregated_importance(payload):
            if agg.method is ImportanceMethod.XGB_GAIN and not math.isnan(agg.importance):
                accumulated[agg.feature].append(agg.importance)
    if not accumulated:
        raise click.ClickException(f"no xgb_gain importances found for {strategy} under {runs_dir}")
    click.echo(f"aggregated xgb_gain over {n_legs} {strategy} legs")
    return {feature: sum(values) / len(values) for feature, values in accumulated.items()}


def _plot_gain(means: dict[str, float], strategy: str, top_n: int, out_dir: Path) -> Path:
    """Draw the top-``top_n`` gain features as horizontal bars and save PNG + SVG."""

    ordered = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    total = len(ordered)
    ordered = ordered[:top_n]
    names = [feature for feature, _ in ordered]
    values = [value for _, value in ordered]
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)
    try:
        ax.barh(range(len(names)), values, color=GAIN_BAR_COLOR)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("XGBoost gain (mean across legs)")
        ax.set_title(f"XGBoost gain - {strategy} (top {len(names)} of {total})")
        ax.grid(True, axis="x", alpha=GRID_ALPHA)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        return save_png_and_svg(fig, out_dir / f"{strategy}_gain.png")
    finally:
        plt.close(fig)


@click.command()
@click.option(
    "--study-dir",
    "study_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Completed study directory (contains runs/ and plots/).",
)
@click.option(
    "--strategy",
    default=DEFAULT_STRATEGY,
    show_default=True,
    help="Tree-based strategy whose xgb_gain entries are plotted.",
)
@click.option(
    "--top-n",
    "top_n",
    default=DEFAULT_TOP_N,
    show_default=True,
    type=int,
    help="Number of strongest features to display.",
)
def main(study_dir: Path, strategy: str, top_n: int) -> None:
    """Plot a tree strategy's XGBoost native-gain feature importance."""

    means = _mean_gain_by_feature(study_dir, strategy)
    out_dir = study_dir / "plots" / FEATURE_IMPORTANCE_SUBDIR
    path = _plot_gain(means, strategy, top_n, out_dir)
    click.echo(f"wrote {path} (+ .svg)")


if __name__ == "__main__":
    main()
