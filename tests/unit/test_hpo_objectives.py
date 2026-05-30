"""
Unit tests for :mod:`src.optimization.objectives`.

Each objective is a pure function over the ``aggregate_metrics`` dict
produced by :meth:`src.analysis.metrics_aggregator.AggregateStats.to_dict`
- no strategy or walk-forward setup required. Fixture dicts mirror the
exact keys that module writes.
"""

from __future__ import annotations

import pytest

from src.core.hpo_config import ObjectiveKind
from src.optimization.objectives import (
    CalmarObjective,
    SharpeObjective,
    SortinoMinusDrawdownPenaltyObjective,
    build_objective,
)

_SHARPE_VALUE = 1.5
_CALMAR_VALUE = 2.5
_SORTINO_VALUE = 2.0
_DRAWDOWN_VALUE = -0.15  # aggregate_metrics reports max_drawdown_worst negative


def _aggregate_fixture() -> dict[str, object]:
    """
    Sample aggregate_metrics dict shaped like the experiment runner's.
    """

    return {
        "n_folds": 4,
        "sharpe_mean": _SHARPE_VALUE,
        "sortino_mean": _SORTINO_VALUE,
        "calmar_mean": _CALMAR_VALUE,
        "max_drawdown_worst": _DRAWDOWN_VALUE,
        "total_return_mean": 0.12,
    }


class TestSharpeObjective:
    def test_reads_sharpe_mean(self) -> None:
        obj = SharpeObjective()
        assert obj(_aggregate_fixture()) == _SHARPE_VALUE

    def test_missing_key_raises_with_hint(self) -> None:
        with pytest.raises(KeyError, match="sharpe_mean"):
            SharpeObjective()({"n_folds": 0})

    def test_non_numeric_value_raises(self) -> None:
        with pytest.raises(TypeError):
            SharpeObjective()({"sharpe_mean": "1.5"})

    def test_bool_value_rejected_as_non_numeric(self) -> None:
        # ``bool`` is a subclass of ``int`` in Python - guard against a
        # truthy value silently passing as 1.0.
        with pytest.raises(TypeError):
            SharpeObjective()({"sharpe_mean": True})


class TestCalmarObjective:
    def test_reads_calmar_mean(self) -> None:
        assert CalmarObjective()(_aggregate_fixture()) == _CALMAR_VALUE


class TestSortinoMinusDrawdownPenaltyObjective:
    def test_default_penalty_applied(self) -> None:
        obj = SortinoMinusDrawdownPenaltyObjective()
        result = obj(_aggregate_fixture())
        assert result == pytest.approx(1.7)

    def test_zero_penalty_collapses_to_sortino(self) -> None:
        obj = SortinoMinusDrawdownPenaltyObjective(penalty=0.0)
        assert obj(_aggregate_fixture()) == _SORTINO_VALUE

    def test_penalty_uses_abs_of_drawdown(self) -> None:
        """
        Whether drawdown is stored negative or positive, the penalty
        should land on the same number."""

        obj = SortinoMinusDrawdownPenaltyObjective(penalty=1.0)
        negative_dd = {"sortino_mean": 1.0, "max_drawdown_worst": -0.2}
        positive_dd = {"sortino_mean": 1.0, "max_drawdown_worst": 0.2}
        assert obj(negative_dd) == pytest.approx(obj(positive_dd))
        assert obj(negative_dd) == pytest.approx(0.8)


class TestBuildObjectiveDispatch:
    @pytest.mark.parametrize(
        "kind,expected_cls",
        [
            (ObjectiveKind.SHARPE, SharpeObjective),
            (ObjectiveKind.CALMAR, CalmarObjective),
            (
                ObjectiveKind.SORTINO_MINUS_DRAWDOWN,
                SortinoMinusDrawdownPenaltyObjective,
            ),
        ],
    )
    def test_dispatch(self, kind: ObjectiveKind, expected_cls: type[object]) -> None:
        assert isinstance(build_objective(kind), expected_cls)

    def test_factory_default_penalty_matches_direct_construction(self) -> None:
        factory_obj = build_objective(ObjectiveKind.SORTINO_MINUS_DRAWDOWN)
        direct_obj = SortinoMinusDrawdownPenaltyObjective()
        assert factory_obj(_aggregate_fixture()) == direct_obj(_aggregate_fixture())
