"""
Behavioral tests for :func:`src.analysis.ranking.rank_strategies`.

Validates the deterministic sort (Sharpe desc -> Sortino desc -> name
asc) and the tidy column set.
"""

from __future__ import annotations

from dataclasses import replace

from src.analysis.metrics_aggregator import AggregateStats
from src.analysis.ranking import rank_strategies
from tests.conftest import make_stub_aggregate_stats

_EXPECTED_COLUMNS: tuple[str, ...] = (
    "rank",
    "name",
    "sharpe_mean",
    "sortino_mean",
    "calmar_mean",
    "max_drawdown_worst",
    "n_folds",
)


class TestRankStrategiesEmpty:
    def test_empty_input_returns_empty_dataframe_with_expected_columns(self) -> None:
        df = rank_strategies({})
        assert list(df.columns) == list(_EXPECTED_COLUMNS)
        assert len(df) == 0


class TestRankStrategiesSingle:
    def test_single_strategy_is_rank_one(self) -> None:
        df = rank_strategies({"Alpha": make_stub_aggregate_stats(sharpe=0.8)})
        assert list(df["rank"]) == [1]
        assert list(df["name"]) == ["Alpha"]


class TestRankStrategiesMultiple:
    def test_sorted_descending_by_primary_metric(self) -> None:
        stats = {
            "Alpha": make_stub_aggregate_stats(sharpe=1.2),
            "Bravo": make_stub_aggregate_stats(sharpe=0.5),
            "Charlie": make_stub_aggregate_stats(sharpe=1.6),
        }
        df = rank_strategies(stats)
        assert list(df["name"]) == ["Charlie", "Alpha", "Bravo"]
        assert list(df["rank"]) == [1, 2, 3]


class TestRankStrategiesTieBreaking:
    def test_ties_on_primary_broken_by_secondary_descending(self) -> None:
        """
        Two strategies tie on Sharpe; the one with higher Sortino wins.

        Uses ``dataclasses.replace`` to diverge Sortino because
        ``make_stub_aggregate_stats`` mirrors Sortino to Sharpe.
        """

        alpha = replace(make_stub_aggregate_stats(sharpe=1.0), sortino_mean=1.0)
        bravo = replace(make_stub_aggregate_stats(sharpe=1.0), sortino_mean=1.5)
        df = rank_strategies({"Alpha": alpha, "Bravo": bravo})
        assert list(df["name"]) == ["Bravo", "Alpha"]

    def test_fully_tied_strategies_broken_by_name_alphabetical(self) -> None:
        stats: dict[str, AggregateStats] = {
            "Charlie": make_stub_aggregate_stats(sharpe=1.0),
            "Alpha": make_stub_aggregate_stats(sharpe=1.0),
            "Bravo": make_stub_aggregate_stats(sharpe=1.0),
        }
        df = rank_strategies(stats)
        assert list(df["name"]) == ["Alpha", "Bravo", "Charlie"]
