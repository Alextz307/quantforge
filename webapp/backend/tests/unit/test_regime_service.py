"""Unit tests for services/regime_service.py."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.core.persistence import REGIME_REPORTS_SUBDIR
from src.orchestration.types import RegimeKind
from webapp.backend.app.infrastructure.store import RegimeReportNotFoundError
from webapp.backend.app.services.regime_service import (
    PlotNotFoundError,
    get_regime_report,
    list_regime_reports,
    resolve_plot,
)
from webapp.backend.tests.conftest import (
    PLOT_BYTES,
    PLOT_FILENAME,
    make_synthetic_regime_report,
)

NEWER_NAME = "regime_newer"
OLDER_NAME = "regime_older"
NEWER_TS = datetime(2026, 4, 2, tzinfo=UTC)
OLDER_TS = datetime(2026, 1, 2, tzinfo=UTC)


def test_list_regime_reports_sorts_newest_first(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    parent = root / "thesis_demo" / REGIME_REPORTS_SUBDIR
    make_synthetic_regime_report(parent, name=OLDER_NAME, created_at=OLDER_TS)
    make_synthetic_regime_report(parent, name=NEWER_NAME, created_at=NEWER_TS)

    summaries = list_regime_reports(root)

    assert [s.name for s in summaries] == [NEWER_NAME, OLDER_NAME]


def test_list_regime_reports_surfaces_kind_and_labels(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_regime_report(
        root / "studies" / "main" / REGIME_REPORTS_SUBDIR,
        name=NEWER_NAME,
        kind="volatility",
        detector_name="vol_quintiles",
        regime_labels=("Q1", "Q3", "Q5"),
    )

    summary = list_regime_reports(root)[0]

    assert summary.kind == RegimeKind.VOLATILITY
    assert summary.detector_name == "vol_quintiles"
    assert summary.regime_labels == ["Q1", "Q3", "Q5"]
    assert summary.store == "studies/main"


def test_get_regime_report_returns_full_detail(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_regime_report(
        root / "thesis_demo" / REGIME_REPORTS_SUBDIR,
        name=NEWER_NAME,
        regime_labels=("bull", "bear"),
    )

    detail = get_regime_report(root, NEWER_NAME)

    assert detail.name == NEWER_NAME
    assert detail.kind == RegimeKind.TREND
    assert {row.regime_label for row in detail.per_regime_stats} == {"bull", "bear"}
    assert detail.per_regime_fold_indices == {"bull": [0], "bear": [1]}
    assert detail.mixed_fold_indices == []
    assert len(detail.slices) == 1
    assert detail.slices[0].label == "bull"
    assert PLOT_FILENAME in detail.plots


def test_get_regime_report_raises_for_unknown_name(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_regime_report(root / "thesis_demo" / REGIME_REPORTS_SUBDIR, name=NEWER_NAME)

    with pytest.raises(RegimeReportNotFoundError):
        get_regime_report(root, "missing_regime")


def test_resolve_plot_returns_path_for_existing_file(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_regime_report(root / "thesis_demo" / REGIME_REPORTS_SUBDIR, name=NEWER_NAME)

    path = resolve_plot(root, NEWER_NAME, PLOT_FILENAME)

    assert path.is_file()
    assert path.read_bytes() == PLOT_BYTES


def test_resolve_plot_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_regime_report(root / "thesis_demo" / REGIME_REPORTS_SUBDIR, name=NEWER_NAME)

    with pytest.raises(PlotNotFoundError):
        resolve_plot(root, NEWER_NAME, "../../../etc/passwd")
