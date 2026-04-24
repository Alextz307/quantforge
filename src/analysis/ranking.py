"""Strategy ranking across :class:`AggregateStats` bundles.

Takes a ``{strategy_name → AggregateStats}`` mapping and produces a tidy
``pd.DataFrame`` sorted by a chosen primary metric, with deterministic
tie-breaking. The DataFrame is the direct input to the comparison
reporter's LaTeX table builder — columns + dtypes are stable across
invocations so the LaTeX output is diffable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pandas as pd

from src.analysis.metrics_aggregator import AggregateStats

RankingMetric = Literal["sharpe", "sortino", "calmar"]

_DISPLAY_COLUMNS: tuple[str, ...] = (
    "rank",
    "name",
    "sharpe_mean",
    "sortino_mean",
    "calmar_mean",
    "max_drawdown_worst",
    "n_folds",
)

# Tiebreaker pick: a strategy's secondary view of the same phenomenon.
# If the primary is Sharpe, fall back to Sortino (reward/downside-risk);
# if primary is Sortino, fall back to Sharpe; for Calmar (reward/drawdown)
# Sortino is the sharpest secondary. Keeps ties deterministic with a
# clear narrative instead of alphabetical-only.
_TIEBREAKER: Mapping[RankingMetric, str] = {
    "sharpe": "sortino_mean",
    "sortino": "sharpe_mean",
    "calmar": "sortino_mean",
}

_PRIMARY_COLUMN: Mapping[RankingMetric, str] = {
    "sharpe": "sharpe_mean",
    "sortino": "sortino_mean",
    "calmar": "calmar_mean",
}


def rank_strategies(
    per_strategy_stats: Mapping[str, AggregateStats],
    *,
    by: RankingMetric = "sharpe",
) -> pd.DataFrame:
    """Rank strategies by the chosen metric, break ties deterministically.

    Sort order: primary metric descending → secondary metric descending →
    strategy name ascending (the final alphabetical step makes the ranking
    bit-stable across invocations even when two strategies tie on every
    numeric axis, which is rare but legal).

    Returns a tidy DataFrame with columns listed in :data:`_DISPLAY_COLUMNS`.
    ``rank`` is 1-indexed and reflects the sort order above (no "dense" or
    "min" rank handling — ties are broken, so each row gets a unique rank).
    """
    if not per_strategy_stats:
        return pd.DataFrame(columns=list(_DISPLAY_COLUMNS))

    primary_col = _PRIMARY_COLUMN[by]
    secondary_col = _TIEBREAKER[by]
    rows: list[dict[str, object]] = [
        {
            "name": name,
            "sharpe_mean": stats.sharpe_mean,
            "sortino_mean": stats.sortino_mean,
            "calmar_mean": stats.calmar_mean,
            "max_drawdown_worst": stats.max_drawdown_worst,
            "n_folds": stats.n_folds,
        }
        for name, stats in per_strategy_stats.items()
    ]
    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=[primary_col, secondary_col, "name"],
        ascending=[False, False, True],
        kind="mergesort",  # stable — guarantees deterministic tie-break at the third key
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df[list(_DISPLAY_COLUMNS)]
