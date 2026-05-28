"""
Tests for :class:`StrategyReporter` — per-experiment artifact bundle.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.engine.scenarios import SlippageScenario
from src.orchestration.manifest import Manifest
from src.orchestration.types import ExperimentResult, FoldRecord
from src.visualization.plots import normalise_to_unit_base
from src.visualization.strategy_reporter import StrategyReporter

_EXPERIMENT_ID = "20260423_120000_Strat_abc1234_ff000000"


def _make_fold(fold_index: int) -> FoldRecord:
    return FoldRecord(
        fold_index=fold_index,
        train_start=pd.Timestamp("2020-01-02"),
        train_end=pd.Timestamp("2021-12-31"),
        test_start=pd.Timestamp("2022-01-03"),
        test_end=pd.Timestamp("2022-12-30"),
        total_return=0.12 + 0.01 * fold_index,
        annualized_return=0.08,
        annualized_volatility=0.15,
        sharpe_ratio=0.55 + 0.1 * fold_index,
        sortino_ratio=0.8,
        calmar_ratio=0.3,
        max_drawdown=-0.1,
        win_rate=0.55,
        trade_count=25 + fold_index,
        equity_curve=(10_000.0, 10_100.0, 10_250.0, 10_180.0, 10_300.0),
    )


def _make_result(n_folds: int = 3) -> ExperimentResult:
    manifest = Manifest(
        experiment_id=_EXPERIMENT_ID,
        name="reporter_test",
        created_at=datetime(2026, 4, 23, 12, 0, 0),
        git_sha="abc1234",
        seed=42,
        data_hash="deadbeef" * 8,
        slippage_scenario=SlippageScenario.NORMAL,
        holdout_start=pd.Timestamp("2023-01-03"),
    )
    folds = tuple(_make_fold(i) for i in range(n_folds))
    return ExperimentResult(experiment_id=_EXPERIMENT_ID, folds=folds, manifest=manifest)


class TestGenerateFullReport:
    def test_creates_plots_and_tables_subdirs(self, tmp_path: Path) -> None:
        StrategyReporter().generate_full_report(_make_result(), tmp_path)
        assert (tmp_path / "plots").is_dir()
        assert (tmp_path / "tables").is_dir()

    def test_equity_curves_png_and_svg(self, tmp_path: Path) -> None:
        StrategyReporter().generate_full_report(_make_result(), tmp_path)
        assert (tmp_path / "plots" / "equity_curves.png").is_file()
        assert (tmp_path / "plots" / "equity_curves.svg").is_file()

    def test_metrics_summary_tex_contains_fold_rows(self, tmp_path: Path) -> None:
        StrategyReporter().generate_full_report(_make_result(n_folds=3), tmp_path)
        tex = (tmp_path / "tables" / "metrics_summary.tex").read_text()
        assert "toprule" in tex
        assert _EXPERIMENT_ID in tex  # surfaced via caption + label

    def test_empty_folds_produces_metrics_but_no_plots(self, tmp_path: Path) -> None:
        """
        A zero-fold ExperimentResult (pathological config) still writes the
        table skeleton so the consuming command sees a predictable layout."""

        StrategyReporter().generate_full_report(_make_result(n_folds=0), tmp_path)
        assert (tmp_path / "tables" / "metrics_summary.tex").is_file()
        assert not (tmp_path / "plots" / "equity_curves.png").exists()


class TestNormaliseCurve:
    """
    Pure-logic tests for the ``normalise_to_unit_base`` helper. No matplotlib,
    no disk I/O — cheap to run and focused on the predicate that decides
    which folds get plotted vs logged+skipped.
    """

    def test_normal_curve_divides_by_first_value(self) -> None:
        assert normalise_to_unit_base((10_000.0, 10_200.0, 9_900.0)) == [1.0, 1.02, 0.99]

    def test_empty_curve_returns_none(self) -> None:
        assert normalise_to_unit_base(()) is None

    @pytest.mark.parametrize("bad_base", [math.nan, math.inf, -math.inf, 0.0, -1.0, -0.0001])
    def test_non_finite_or_non_positive_base_returns_none(self, bad_base: float) -> None:
        """
        Every class of degenerate first-value that the integration path
        defensively rejects: NaN (zero-trade fold), ±inf (overflow), zero
        (catastrophic exit at bar 0), negative (debt-at-start)."""

        assert normalise_to_unit_base((bad_base, 100.0, 200.0)) is None


class TestDegenerateFoldsAreSkippedIntegration:
    """
    Integration test for the defensive path — verifies the equity plot
    method logs and skips without crashing when fed a degenerate fold.
    Pure predicate logic is covered by ``TestNormaliseCurve`` above; this
    test exercises the plotting-side wiring (one fold good, one bad →
    plot still renders).
    """

    def test_nan_equity_base_is_skipped_with_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        base_result = _make_result(n_folds=2)
        bad_fold = dataclasses.replace(
            base_result.folds[0], equity_curve=(math.nan, 10_100.0, 10_050.0)
        )
        result = dataclasses.replace(base_result, folds=(bad_fold, base_result.folds[1]))
        with caplog.at_level(logging.WARNING, logger="src.visualization.strategy_reporter"):
            StrategyReporter().generate_full_report(result, tmp_path)
        assert (tmp_path / "plots" / "equity_curves.png").is_file()
        assert any("skipping from equity plot" in r.message for r in caplog.records)
