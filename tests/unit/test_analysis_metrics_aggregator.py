"""
Behavioral tests for :mod:`src.analysis.metrics_aggregator`.

Verifies:

* :meth:`AggregateStats.to_dict` keys are a strict SUPERSET of the
  pre-refactor ``_aggregate_metrics`` keys - the downstream objective
  adapters and ``metrics.json`` readers rely on the old keys staying.
* Empty-fold path short-circuits to ``{"n_folds": 0}`` (objectives'
  error messages depend on the absence of other keys, not on NaN).
* Single-fold path collapses std + CI to the point estimate.
* Multi-fold path reports finite, ordered CI bounds around the sample mean.
* Determinism - two calls with the same folds produce bit-identical CIs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.metrics_aggregator import AggregateStats, aggregate_folds
from src.orchestration.types import FoldRecord

_LEGACY_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "n_folds",
        "sharpe_mean",
        "sortino_mean",
        "calmar_mean",
        "max_drawdown_worst",
        "total_return_mean",
    }
)

_FOLD_A_SHARPE = 1.2
_FOLD_B_SHARPE = 0.8
_FOLD_C_SHARPE = 1.5


def _make_fold(
    idx: int,
    *,
    sharpe: float,
    sortino: float | None = None,
    calmar: float | None = None,
    max_drawdown: float = -0.08,
    total_return: float = 0.05,
    win_rate: float = 0.55,
    trade_count: int = 30,
) -> FoldRecord:
    return FoldRecord(
        fold_index=idx,
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2020-06-30"),
        test_start=pd.Timestamp("2020-07-01"),
        test_end=pd.Timestamp("2020-12-31"),
        total_return=total_return,
        annualized_return=total_return * 2,
        annualized_volatility=0.15,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino if sortino is not None else sharpe * 1.05,
        calmar_ratio=calmar if calmar is not None else sharpe * 0.9,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        trade_count=trade_count,
        equity_curve=(1.0, 1.02, 1.05),
    )


class TestEmptyFolds:
    def test_returns_sentinel_aggregate(self) -> None:
        """
        Equality would fail on NaN scalars - check the sentinel via the
        discriminator field instead.
        """

        stats = aggregate_folds(())
        assert stats.n_folds == 0
        assert stats.trade_count_total == 0
        assert math.isnan(stats.sharpe_mean)

    def test_empty_classmethod_matches_zero_fold_aggregate(self) -> None:
        stats = aggregate_folds(())
        sentinel = AggregateStats.empty()
        assert stats.n_folds == sentinel.n_folds
        assert stats.trade_count_total == sentinel.trade_count_total

    def test_to_dict_short_circuits_to_n_folds_only(self) -> None:
        d = aggregate_folds(()).to_dict()
        assert d == {"n_folds": 0}


class TestSuperset:
    def test_to_dict_keeps_every_legacy_metric_key(self) -> None:
        stats = aggregate_folds((_make_fold(0, sharpe=_FOLD_A_SHARPE),))
        assert _LEGACY_METRIC_KEYS.issubset(stats.to_dict().keys())


class TestSingleFold:
    def test_std_is_zero_and_ci_collapses_to_point(self) -> None:
        stats = aggregate_folds((_make_fold(0, sharpe=_FOLD_A_SHARPE),))
        assert stats.n_folds == 1
        assert stats.sharpe_mean == _FOLD_A_SHARPE
        assert stats.sharpe_std == 0.0
        assert stats.sharpe_ci95_low == _FOLD_A_SHARPE
        assert stats.sharpe_ci95_high == _FOLD_A_SHARPE


class TestMultiFold:
    def test_mean_matches_numpy_mean(self) -> None:
        folds = (
            _make_fold(0, sharpe=_FOLD_A_SHARPE),
            _make_fold(1, sharpe=_FOLD_B_SHARPE),
            _make_fold(2, sharpe=_FOLD_C_SHARPE),
        )
        stats = aggregate_folds(folds)
        expected = float(np.mean([_FOLD_A_SHARPE, _FOLD_B_SHARPE, _FOLD_C_SHARPE]))
        assert stats.sharpe_mean == pytest.approx(expected)

    def test_ci_bounds_are_ordered_and_bracket_the_mean(self) -> None:
        folds = (
            _make_fold(0, sharpe=_FOLD_A_SHARPE),
            _make_fold(1, sharpe=_FOLD_B_SHARPE),
            _make_fold(2, sharpe=_FOLD_C_SHARPE),
        )
        stats = aggregate_folds(folds)
        assert stats.sharpe_ci95_low <= stats.sharpe_mean <= stats.sharpe_ci95_high

    def test_max_drawdown_worst_is_min_across_folds(self) -> None:
        folds = (
            _make_fold(0, sharpe=_FOLD_A_SHARPE, max_drawdown=-0.05),
            _make_fold(1, sharpe=_FOLD_B_SHARPE, max_drawdown=-0.30),
            _make_fold(2, sharpe=_FOLD_C_SHARPE, max_drawdown=-0.12),
        )
        stats = aggregate_folds(folds)
        assert stats.max_drawdown_worst == -0.30

    def test_trade_count_is_summed(self) -> None:
        folds = (
            _make_fold(0, sharpe=_FOLD_A_SHARPE, trade_count=10),
            _make_fold(1, sharpe=_FOLD_B_SHARPE, trade_count=25),
        )
        assert aggregate_folds(folds).trade_count_total == 35


class TestDeterminism:
    def test_two_calls_with_same_folds_produce_bit_identical_stats(self) -> None:
        folds = (
            _make_fold(0, sharpe=_FOLD_A_SHARPE),
            _make_fold(1, sharpe=_FOLD_B_SHARPE),
            _make_fold(2, sharpe=_FOLD_C_SHARPE),
        )
        a = aggregate_folds(folds)
        b = aggregate_folds(folds)
        assert a == b


class TestNanPropagation:
    def test_nan_fold_sharpe_flows_into_mean_and_ci(self) -> None:
        """
        Zero-vol folds produce NaN metrics. Aggregate must surface NaN
        rather than silently hiding the degenerate fold - callers can then
        decide whether to treat the run as invalid.
        """

        folds = (
            _make_fold(0, sharpe=float("nan")),
            _make_fold(1, sharpe=_FOLD_B_SHARPE),
        )
        stats = aggregate_folds(folds)
        assert math.isnan(stats.sharpe_mean)
        assert math.isnan(stats.sharpe_ci95_low)
        assert math.isnan(stats.sharpe_ci95_high)
