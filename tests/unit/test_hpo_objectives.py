"""
Unit tests for :mod:`src.optimization.objectives`.

Each objective is a pure function over the ``aggregate_metrics`` dict
produced by :meth:`src.analysis.metrics_aggregator.AggregateStats.to_dict`
- no strategy or walk-forward setup required. Fixture dicts mirror the
exact keys that module writes.
"""

from __future__ import annotations

import pytest

from src.optimization.objectives import SharpeObjective

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
