"""Tests for :func:`src.orchestration.comparison.run_comparison`.

Monkeypatches ``build_experiment`` so each fake experiment yields
deterministic equity curves (no real walk-forward). Only ``n_jobs=1``
(in-process) is exercised here — the ``n_jobs>1`` ProcessPoolExecutor
path needs module-level fakes that survive pickling, which adds more
scaffolding than benefits this test file. The in-process path covers
the orchestration logic (validation, aggregation, pairwise) end to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import ExperimentConfig
from src.orchestration import comparison as comparison_mod
from src.orchestration.comparison import SignificanceTest, run_comparison
from src.orchestration.types import ExperimentResult, StrategyComparisonReport
from tests.conftest import (
    comparison_curve_seed,
    make_log_return_equity_curve,
    make_stub_experiment_result,
    make_stub_fold_record,
)

_SPY_DATA = {
    "source": "csv",
    "tickers": ["SPY"],
    "start": "2020-01-01",
    "end": "2024-01-01",
    "interval": "daily",
}
_N_FOLDS = 3
_FOLD_CURVE_LENGTH = 40


def _build_cfg(name: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": name,
            "seed": 42,
            "data": _SPY_DATA,
            "strategy": {"name": "AdaptiveBollinger", "params": {}},
        }
    )


def _stub_result(name: str, sharpe: float) -> ExperimentResult:
    folds = tuple(
        make_stub_fold_record(
            i,
            sharpe=sharpe,
            equity_curve=make_log_return_equity_curve(
                sharpe, n=_FOLD_CURVE_LENGTH, seed=comparison_curve_seed(name, i)
            ),
        )
        for i in range(_N_FOLDS)
    )
    return make_stub_experiment_result(name, folds=folds)


class _StubExperiment:
    """Mimics the shape Experiment's ``run`` exposes to run_comparison."""

    def __init__(self, name: str, sharpe: float) -> None:
        self._name = name
        self._sharpe = sharpe

    def run(self, *, store_root: Path, write_report: bool) -> ExperimentResult:
        assert write_report is False  # comparison opts out of per-experiment reports
        return _stub_result(self._name, self._sharpe)


@pytest.fixture
def patched_build(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """Route ``build_experiment(cfg)`` to a stub whose sharpe is taken from
    a name→sharpe dict the test fills. The fixture returns the dict so
    tests can populate it before invoking ``run_comparison``.
    """
    sharpe_by_name: dict[str, float] = {}

    def _fake_build(cfg: ExperimentConfig) -> _StubExperiment:
        return _StubExperiment(name=cfg.name, sharpe=sharpe_by_name[cfg.name])

    monkeypatch.setattr(comparison_mod, "build_experiment", _fake_build)
    return sharpe_by_name


class TestRunComparisonBasic:
    def test_returns_report_and_folds_for_each_strategy(
        self, tmp_path: Path, patched_build: dict[str, float]
    ) -> None:
        patched_build["Alpha"] = 1.5
        patched_build["Bravo"] = 0.8

        report, folds = run_comparison(
            [_build_cfg("Alpha"), _build_cfg("Bravo")],
            out_name="basic",
            store_root=tmp_path,
        )

        assert isinstance(report, StrategyComparisonReport)
        assert set(report.per_strategy_stats.keys()) == {"Alpha", "Bravo"}
        assert set(folds.keys()) == {"Alpha", "Bravo"}
        assert all(len(fold_tuple) == _N_FOLDS for fold_tuple in folds.values())

    def test_ranking_orders_strategies_by_sharpe_desc(
        self, tmp_path: Path, patched_build: dict[str, float]
    ) -> None:
        patched_build["Alpha"] = 1.5
        patched_build["Bravo"] = 0.8
        patched_build["Charlie"] = 2.1

        report, _ = run_comparison(
            [_build_cfg("Alpha"), _build_cfg("Bravo"), _build_cfg("Charlie")],
            out_name="rank",
            store_root=tmp_path,
        )
        # Highest sharpe first. ``aggregate_folds`` computes from the
        # synthetic curves so ordering is observable, not hard-coded.
        names_in_order = list(report.ranking["name"])
        assert names_in_order[0] == "Charlie"


class TestRunComparisonPairwise:
    def test_bootstrap_produces_upper_triangular_entries(
        self, tmp_path: Path, patched_build: dict[str, float]
    ) -> None:
        patched_build["Alpha"] = 1.5
        patched_build["Bravo"] = 0.8
        patched_build["Charlie"] = 2.1

        report, _ = run_comparison(
            [_build_cfg("Alpha"), _build_cfg("Bravo"), _build_cfg("Charlie")],
            out_name="pairwise",
            store_root=tmp_path,
            significance_test=SignificanceTest.BOOTSTRAP,
        )
        # 3 choose 2 = 3 pairs
        assert len(report.pairwise) == 3

    def test_significance_none_leaves_pairwise_empty(
        self, tmp_path: Path, patched_build: dict[str, float]
    ) -> None:
        patched_build["Alpha"] = 1.5
        patched_build["Bravo"] = 0.8

        report, _ = run_comparison(
            [_build_cfg("Alpha"), _build_cfg("Bravo")],
            out_name="nopair",
            store_root=tmp_path,
            significance_test=SignificanceTest.NONE,
        )
        assert report.pairwise == ()


class TestRunComparisonValidation:
    def test_rejects_single_config(self, tmp_path: Path, patched_build: dict[str, float]) -> None:
        patched_build["Alpha"] = 1.0
        with pytest.raises(ValueError, match="at least 2 configs"):
            run_comparison([_build_cfg("Alpha")], out_name="single", store_root=tmp_path)

    def test_rejects_duplicate_strategy_names(
        self, tmp_path: Path, patched_build: dict[str, float]
    ) -> None:
        patched_build["Alpha"] = 1.0
        with pytest.raises(ValueError, match="unique config names"):
            run_comparison(
                [_build_cfg("Alpha"), _build_cfg("Alpha")],
                out_name="dup",
                store_root=tmp_path,
            )

    def test_rejects_invalid_n_jobs(self, tmp_path: Path, patched_build: dict[str, float]) -> None:
        patched_build["Alpha"] = 1.0
        patched_build["Bravo"] = 1.0
        with pytest.raises(ValueError, match="n_jobs"):
            run_comparison(
                [_build_cfg("Alpha"), _build_cfg("Bravo")],
                out_name="badjobs",
                store_root=tmp_path,
                n_jobs=0,
            )


class TestRunComparisonAlignment:
    def test_fold_count_mismatch_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strategies with different fold counts cannot be paired-bootstrapped
        — the orchestrator must refuse rather than silently align on the shorter.
        """

        def _fake_build_mismatched(cfg: ExperimentConfig) -> _StubExperiment:
            class _Short:
                def __init__(self, cfg_name: str) -> None:
                    self._cfg_name = cfg_name

                def run(self, *, store_root: Path, write_report: bool) -> ExperimentResult:
                    # Alpha has 3 folds, Bravo has only 2 — alignment violated.
                    n = 3 if self._cfg_name == "Alpha" else 2
                    folds = tuple(
                        make_stub_fold_record(
                            i,
                            sharpe=1.0,
                            equity_curve=make_log_return_equity_curve(
                                1.0, n=_FOLD_CURVE_LENGTH, seed=i
                            ),
                        )
                        for i in range(n)
                    )
                    return make_stub_experiment_result(self._cfg_name, folds=folds)

            return _Short(cfg.name)  # type: ignore[return-value]

        monkeypatch.setattr(comparison_mod, "build_experiment", _fake_build_mismatched)

        with pytest.raises(ValueError, match="aligned folds"):
            run_comparison(
                [_build_cfg("Alpha"), _build_cfg("Bravo")],
                out_name="mismatch",
                store_root=tmp_path,
                significance_test=SignificanceTest.BOOTSTRAP,
            )
