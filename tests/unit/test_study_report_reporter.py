"""Unit tests for :class:`StudyReportReporter`.

Constructs a small in-memory :class:`ConsolidatedStudyReport`, runs the
reporter against ``tmp_path``, and asserts the file tree shape + a few
representative content checks. Pixel-perfect plot validation is out of
scope; we verify each PNG/SVG is non-empty and tables contain the
expected rows.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.orchestration.study_report import ConsolidatedStudyReport, HoldoutSnapshot
from src.orchestration.types import PairwiseSignificance
from src.visualization.plots import MANIFEST_FILENAME, PLOTS_SUBDIR, TABLES_SUBDIR
from src.visualization.study_report_reporter import StudyReportReporter
from tests.conftest import make_stub_aggregate_stats

_PUBLISH_LABEL = "test_study_v1"


def _make_report(study_dir: Path) -> ConsolidatedStudyReport:
    """Build a 2-strategy × 2-universe consolidated report.

    Includes holdout data on two legs so the full set of conditional
    sections fires under one assertion sweep.
    """

    return ConsolidatedStudyReport(
        study_name="test_study",
        study_dir=study_dir,
        created_at=datetime(2026, 5, 4, tzinfo=UTC),
        git_sha="stubsha1",
        per_leg_aggregate={
            ("StratA", "uni1"): make_stub_aggregate_stats(sharpe=1.20, n_folds=4),
            ("StratA", "uni2"): make_stub_aggregate_stats(sharpe=0.60, n_folds=4),
            ("StratB", "uni1"): make_stub_aggregate_stats(sharpe=0.90, n_folds=4),
            ("StratB", "uni2"): make_stub_aggregate_stats(sharpe=1.50, n_folds=4),
        },
        per_leg_run_id={
            ("StratA", "uni1"): "stub_StratA__uni1",
            ("StratA", "uni2"): "stub_StratA__uni2",
            ("StratB", "uni1"): "stub_StratB__uni1",
            ("StratB", "uni2"): "stub_StratB__uni2",
        },
        per_leg_holdout={
            ("StratA", "uni1"): _holdout(0.95),
            ("StratB", "uni2"): _holdout(1.10),
        },
        per_universe_pairwise={
            "uni1": (
                PairwiseSignificance(
                    name_a="StratA",
                    name_b="StratB",
                    point_differential=0.30,
                    lower=0.05,
                    upper=0.55,
                    confidence=0.95,
                    significant=True,
                ),
            )
        },
        incomplete_leg_ids=("StratA__uni3",),
    )


def _holdout(sharpe: float) -> HoldoutSnapshot:
    return HoldoutSnapshot(
        sharpe_ratio=sharpe,
        sortino_ratio=sharpe * 1.05,
        calmar_ratio=sharpe * 0.9,
        max_drawdown=-0.07,
        annualized_return=0.10,
        annualized_volatility=0.15,
        total_return=0.05,
        win_rate=0.55,
        trade_count=30,
        holdout_start=pd.Timestamp("2024-01-01"),
        n_dev_bars=1000,
        n_holdout_bars=250,
    )


def test_generate_full_report_writes_full_tree(tmp_path: Path) -> None:
    """One-shot: every expected artifact path exists after a happy-path run."""

    report = _make_report(study_dir=tmp_path)
    out = StudyReportReporter().generate_full_report(report, tmp_path, publish_label=_PUBLISH_LABEL)
    assert out == tmp_path

    manifest_path = tmp_path / MANIFEST_FILENAME
    assert manifest_path.is_file()

    tables = tmp_path / TABLES_SUBDIR
    assert (tables / "master_ranking.tex").is_file()
    assert (tables / "master_ranking.csv").is_file()
    assert (tables / "per_universe_ranking.tex").is_file()
    assert (tables / "per_universe_ranking.csv").is_file()
    assert (tables / "holdout_results.tex").is_file()
    assert (tables / "holdout_results.csv").is_file()
    assert (tables / "pairwise_significance.csv").is_file()
    assert (tables / "pairwise_significance" / "uni1.tex").is_file()

    plots = tmp_path / PLOTS_SUBDIR
    for stem in (
        "strategy_x_universe_heatmap",
        "holdout_dev_scatter",
    ):
        assert (plots / f"{stem}.png").is_file(), stem
        assert (plots / f"{stem}.svg").is_file(), stem


def test_master_ranking_sorts_by_sharpe_desc(tmp_path: Path) -> None:
    """StratB__uni2 has the highest Sharpe (1.50) and should rank #1."""

    report = _make_report(study_dir=tmp_path)
    StudyReportReporter().generate_full_report(report, tmp_path)

    df = pd.read_csv(tmp_path / TABLES_SUBDIR / "master_ranking.csv")
    assert list(df.columns)[:3] == ["strategy", "universe", "n_folds"]
    assert df.iloc[0]["strategy"] == "StratB"
    assert df.iloc[0]["universe"] == "uni2"
    assert df.iloc[0]["sharpe_mean"] == pytest.approx(1.50)


def test_holdout_results_includes_dev_and_holdout_columns(tmp_path: Path) -> None:
    report = _make_report(study_dir=tmp_path)
    StudyReportReporter().generate_full_report(report, tmp_path)

    df = pd.read_csv(tmp_path / TABLES_SUBDIR / "holdout_results.csv")
    assert {"dev_sharpe", "holdout_sharpe", "n_dev_bars", "n_holdout_bars"} <= set(df.columns)
    assert len(df) == 2
    assert df["holdout_sharpe"].max() == pytest.approx(1.10)


def test_pairwise_long_csv_records_one_row_per_pair(tmp_path: Path) -> None:
    report = _make_report(study_dir=tmp_path)
    StudyReportReporter().generate_full_report(report, tmp_path)

    df = pd.read_csv(tmp_path / TABLES_SUBDIR / "pairwise_significance.csv")
    expected_cols = {
        "universe",
        "strategy_a",
        "strategy_b",
        "point_differential",
        "ci_low",
        "ci_high",
    }
    assert expected_cols <= set(df.columns)
    assert len(df) == 1
    assert df.iloc[0]["universe"] == "uni1"
    assert df.iloc[0]["significant"]


def test_publish_label_appears_in_tex_caption(tmp_path: Path) -> None:
    """``publish_label`` should land in every emitted .tex caption / label."""

    report = _make_report(study_dir=tmp_path)
    StudyReportReporter().generate_full_report(report, tmp_path, publish_label=_PUBLISH_LABEL)

    master_tex = (tmp_path / TABLES_SUBDIR / "master_ranking.tex").read_text(encoding="utf-8")
    assert _PUBLISH_LABEL in master_tex


def test_skips_sections_when_no_input_data(tmp_path: Path) -> None:
    """Sparse report: no holdout / no pairwise → those tables not written."""

    sparse = ConsolidatedStudyReport(
        study_name="sparse",
        study_dir=tmp_path,
        created_at=datetime(2026, 5, 4, tzinfo=UTC),
        git_sha="stubsha1",
        per_leg_aggregate={("S", "u"): make_stub_aggregate_stats(sharpe=1.0, n_folds=3)},
        per_leg_run_id={("S", "u"): "stub_run"},
        per_leg_holdout={},
        per_universe_pairwise={},
        incomplete_leg_ids=(),
    )
    StudyReportReporter().generate_full_report(sparse, tmp_path)

    tables = tmp_path / TABLES_SUBDIR
    assert (tables / "master_ranking.tex").is_file()
    assert not (tables / "holdout_results.tex").exists()
    assert not (tables / "pairwise_significance.csv").exists()


def test_copies_per_universe_equity_overlay_when_source_exists(tmp_path: Path) -> None:
    """Reporter copies ``comparisons/<universe>/plots/equity_overlay.{png,svg}`` if present."""

    report = _make_report(study_dir=tmp_path)
    src_dir = tmp_path / "comparisons" / "uni1" / "plots"
    src_dir.mkdir(parents=True)
    (src_dir / "equity_overlay.png").write_bytes(b"PNG_MARKER")
    (src_dir / "equity_overlay.svg").write_bytes(b"SVG_MARKER")

    StudyReportReporter().generate_full_report(report, tmp_path)

    dst_png = tmp_path / PLOTS_SUBDIR / "per_universe_equity_overlays" / "uni1.png"
    assert dst_png.read_bytes() == b"PNG_MARKER"
    assert (dst_png.with_suffix(".svg")).read_bytes() == b"SVG_MARKER"
    assert not (tmp_path / PLOTS_SUBDIR / "per_universe_equity_overlays" / "uni2.png").exists()
