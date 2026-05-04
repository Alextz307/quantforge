"""Unit tests for services/holdout_service.py."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.core.persistence import HOLDOUT_EVALS_SUBDIR
from src.engine.scenarios import SlippageScenario
from webapp.backend.app.infrastructure.store import HoldoutEvalNotFoundError
from webapp.backend.app.services.holdout_service import (
    PlotNotFoundError,
    get_holdout_eval,
    list_holdout_evals,
    resolve_plot,
)
from webapp.backend.tests.conftest import (
    PLOT_BYTES,
    PLOT_FILENAME,
    make_synthetic_holdout_eval,
)

NEWER_NAME = "holdout_newer"
OLDER_NAME = "holdout_older"
NEWER_TS = datetime(2026, 4, 3, tzinfo=UTC)
OLDER_TS = datetime(2026, 1, 3, tzinfo=UTC)
HOLDOUT_BOUNDARY = datetime(2024, 1, 1, tzinfo=UTC)
EXPECTED_SHARPE = 0.6
EXPECTED_EQUITY_CURVE = [10000.0, 10100.0, 10500.0]


def test_list_holdout_evals_sorts_newest_first(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    parent = root / "studies" / "main" / HOLDOUT_EVALS_SUBDIR
    make_synthetic_holdout_eval(parent, name=OLDER_NAME, created_at=OLDER_TS)
    make_synthetic_holdout_eval(parent, name=NEWER_NAME, created_at=NEWER_TS)

    summaries = list_holdout_evals(root)

    assert [s.name for s in summaries] == [NEWER_NAME, OLDER_NAME]


def test_list_holdout_evals_surfaces_source_and_store(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_holdout_eval(
        root / "thesis_demo" / HOLDOUT_EVALS_SUBDIR,
        name=NEWER_NAME,
        source_kind="hpo",
        source_id="some_hpo_study",
        holdout_start=HOLDOUT_BOUNDARY,
    )

    summary = list_holdout_evals(root)[0]

    assert summary.source_kind == "hpo"
    assert summary.source_id == "some_hpo_study"
    assert summary.holdout_start == HOLDOUT_BOUNDARY
    assert summary.store == "thesis_demo"


def test_get_holdout_eval_returns_full_detail(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_holdout_eval(
        root / "studies" / "main" / HOLDOUT_EVALS_SUBDIR,
        name=NEWER_NAME,
        sharpe_ratio=EXPECTED_SHARPE,
    )

    detail = get_holdout_eval(root, NEWER_NAME)

    assert detail.name == NEWER_NAME
    assert detail.git_sha == "abc1234"
    assert detail.slippage_scenario == SlippageScenario.NORMAL
    assert detail.sharpe_ratio == pytest.approx(EXPECTED_SHARPE)
    assert detail.equity_curve == EXPECTED_EQUITY_CURVE
    assert PLOT_FILENAME in detail.plots


def test_get_holdout_eval_raises_for_unknown_name(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_holdout_eval(root / "studies" / "main" / HOLDOUT_EVALS_SUBDIR, name=NEWER_NAME)

    with pytest.raises(HoldoutEvalNotFoundError):
        get_holdout_eval(root, "missing_holdout")


def test_resolve_plot_returns_path_for_existing_file(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_holdout_eval(root / "studies" / "main" / HOLDOUT_EVALS_SUBDIR, name=NEWER_NAME)

    path = resolve_plot(root, NEWER_NAME, PLOT_FILENAME)

    assert path.is_file()
    assert path.read_bytes() == PLOT_BYTES


def test_resolve_plot_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_holdout_eval(root / "studies" / "main" / HOLDOUT_EVALS_SUBDIR, name=NEWER_NAME)

    with pytest.raises(PlotNotFoundError):
        resolve_plot(root, NEWER_NAME, "../../../etc/passwd")
