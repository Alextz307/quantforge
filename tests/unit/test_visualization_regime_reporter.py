"""Unit tests for ``RegimeReporter`` artifact generation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.metrics_aggregator import AggregateStats, aggregate_folds
from src.orchestration.types import (
    MIXED_REGIME_LABEL,
    FoldRecord,
    RegimeKind,
    RegimeReport,
    RegimeSlice,
)
from src.visualization.regime_reporter import RegimeReporter


def _stub_fold(fold_index: int, sharpe: float) -> FoldRecord:
    base = pd.Timestamp("2020-01-01")
    return FoldRecord(
        fold_index=fold_index,
        train_start=base,
        train_end=base + pd.Timedelta(days=30),
        test_start=base + pd.Timedelta(days=31),
        test_end=base + pd.Timedelta(days=60),
        total_return=0.05,
        annualized_return=0.10,
        annualized_volatility=0.15,
        sharpe_ratio=sharpe,
        sortino_ratio=sharpe * 1.05,
        calmar_ratio=sharpe * 0.9,
        max_drawdown=-0.07,
        win_rate=0.55,
        trade_count=20,
        equity_curve=(1.0, 1.05),
    )


def _stub_report(*, with_mixed: bool, with_empty: bool = False) -> RegimeReport:
    bull_stats = aggregate_folds((_stub_fold(0, 1.5), _stub_fold(1, 1.2)))
    bear_stats = aggregate_folds((_stub_fold(2, -0.5),))
    per_regime: dict[str, AggregateStats] = {
        "bull": bull_stats,
        "bear": bear_stats,
    }
    per_regime_indices: dict[str, tuple[int, ...]] = {
        "bull": (0, 1),
        "bear": (2,),
    }
    if with_empty:
        per_regime["nofolds"] = AggregateStats.empty()
        per_regime_indices["nofolds"] = ()
    mixed_indices: tuple[int, ...] = ()
    if with_mixed:
        per_regime[MIXED_REGIME_LABEL] = aggregate_folds((_stub_fold(3, 0.0),))
        per_regime_indices[MIXED_REGIME_LABEL] = (3,)
        mixed_indices = (3,)
    return RegimeReport(
        out_name="test_run",
        experiment_id="exp_abc",
        kind=RegimeKind.TREND,
        detector_name="trend",
        created_at=datetime(2026, 4, 25, tzinfo=UTC),
        git_sha="cafebab1",
        per_regime_stats=per_regime,
        per_regime_fold_indices=per_regime_indices,
        mixed_fold_indices=mixed_indices,
        slices=(
            RegimeSlice(
                label="bull",
                start=pd.Timestamp("2020-01-01"),
                end=pd.Timestamp("2020-06-01"),
            ),
            RegimeSlice(
                label="bear",
                start=pd.Timestamp("2020-06-01"),
                end=pd.Timestamp("2020-12-31"),
            ),
        ),
    )


def test_generate_full_report_writes_all_artifacts(tmp_path: Path) -> None:
    report = _stub_report(with_mixed=True)
    out_dir = tmp_path / "regime_reports" / "test_run"

    RegimeReporter().generate_full_report(report, out_dir)

    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "tables" / "regime_summary.tex").is_file()
    assert (out_dir / "plots" / "regime_metric_heatmap.png").is_file()
    assert (out_dir / "plots" / "regime_metric_heatmap.svg").is_file()
    assert (out_dir / "plots" / "regime_timeline.png").is_file()
    assert (out_dir / "plots" / "regime_timeline.svg").is_file()


def test_manifest_round_trips_identity_fields(tmp_path: Path) -> None:
    report = _stub_report(with_mixed=True)
    out_dir = tmp_path / "regime_reports" / "test_run"
    RegimeReporter().generate_full_report(report, out_dir)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["out_name"] == report.out_name
    assert manifest["experiment_id"] == report.experiment_id
    assert manifest["kind"] == RegimeKind.TREND.value
    assert manifest["detector_name"] == "trend"
    assert manifest["git_sha"] == report.git_sha
    assert set(manifest["per_regime_stats"].keys()) == set(report.per_regime_stats.keys())
    assert manifest["mixed_fold_indices"] == [3]
    assert len(manifest["slices"]) == 2


def test_summary_table_includes_every_regime_row(tmp_path: Path) -> None:
    report = _stub_report(with_mixed=True, with_empty=True)
    out_dir = tmp_path / "regime_reports" / "test_run"
    RegimeReporter().generate_full_report(report, out_dir)

    tex = (out_dir / "tables" / "regime_summary.tex").read_text()
    for label in report.per_regime_stats:
        assert label in tex


def test_empty_regime_renders_dashes_in_summary(tmp_path: Path) -> None:
    report = _stub_report(with_mixed=False, with_empty=True)
    out_dir = tmp_path / "regime_reports" / "test_run"
    RegimeReporter().generate_full_report(report, out_dir)

    tex = (out_dir / "tables" / "regime_summary.tex").read_text()
    # Empty regime row should contain the placeholder dash, not a stray NaN.
    assert "nofolds" in tex
    assert "nan" not in tex.lower()


def test_per_regime_stats_mapping_proxy_is_immutable() -> None:
    report = _stub_report(with_mixed=False)
    with pytest.raises(TypeError):
        report.per_regime_stats["new_key"] = AggregateStats.empty()  # type: ignore[index]


def test_regime_slice_round_trip_preserves_fields() -> None:
    sl = RegimeSlice(
        label="x",
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-02-01"),
    )
    assert RegimeSlice.from_dict(sl.to_dict()) == sl
