"""Unit tests for services/hpo_service.py."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.core.persistence import HPO_SUBDIR
from webapp.backend.app.infrastructure.store import HpoStudyNotFoundError
from webapp.backend.app.services.hpo_service import (
    get_hpo_study,
    list_hpo_studies,
    list_trials,
)
from webapp.backend.tests.conftest import make_synthetic_hpo_study

NEWER_NAME = "Hpo__newer"
OLDER_NAME = "Hpo__older"
NEWER_TS = datetime(2026, 5, 1, tzinfo=UTC)
OLDER_TS = datetime(2026, 2, 1, tzinfo=UTC)
EXPECTED_BEST_VALUE = 0.95
EXPECTED_BEST_TRIAL_NUMBER = 2
EXPECTED_N_TRIALS = 4
EXPECTED_N_COMPLETE = 3
AFTER_TRIAL_FILTER = 1


def test_list_hpo_studies_sorts_newest_first(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    parent = root / "studies" / "main" / HPO_SUBDIR
    make_synthetic_hpo_study(parent, name=OLDER_NAME, created_at=OLDER_TS)
    make_synthetic_hpo_study(parent, name=NEWER_NAME, created_at=NEWER_TS)

    summaries = list_hpo_studies(root)

    assert [s.name for s in summaries] == [NEWER_NAME, OLDER_NAME]


def test_list_hpo_studies_surfaces_best_and_store(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=NEWER_NAME,
        n_trials=EXPECTED_N_TRIALS,
        n_complete=EXPECTED_N_COMPLETE,
        best_value=EXPECTED_BEST_VALUE,
        best_trial_number=EXPECTED_BEST_TRIAL_NUMBER,
    )

    summary = list_hpo_studies(root)[0]

    assert summary.n_trials == EXPECTED_N_TRIALS
    assert summary.n_complete == EXPECTED_N_COMPLETE
    assert summary.best_value == pytest.approx(EXPECTED_BEST_VALUE)
    assert summary.best_trial_number == EXPECTED_BEST_TRIAL_NUMBER
    assert summary.store == "studies/main"


def test_get_hpo_study_returns_best_config(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=NEWER_NAME,
    )

    detail = get_hpo_study(root, NEWER_NAME)

    assert detail.name == NEWER_NAME
    assert detail.best_config["name"] == "demo"
    strategy = detail.best_config["strategy"]
    assert isinstance(strategy, dict)
    assert strategy["name"] == "AdaptiveBollinger"


def test_get_hpo_study_handles_missing_best_config(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=NEWER_NAME,
        n_complete=0,
        write_best_config=False,
    )

    detail = get_hpo_study(root, NEWER_NAME)

    assert detail.best_config == {}
    assert detail.best_value is None
    assert detail.best_trial_number is None


def test_get_hpo_study_raises_for_unknown_name(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(root / "studies" / "main" / HPO_SUBDIR, name=NEWER_NAME)

    with pytest.raises(HpoStudyNotFoundError):
        get_hpo_study(root, "missing_hpo")


def test_list_trials_returns_all_by_default(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=NEWER_NAME,
        n_trials=EXPECTED_N_TRIALS,
        n_complete=EXPECTED_N_COMPLETE,
    )

    rows = list_trials(root, NEWER_NAME)

    assert [r.number for r in rows] == list(range(EXPECTED_N_TRIALS))
    assert sum(1 for r in rows if r.state == "COMPLETE") == EXPECTED_N_COMPLETE


def test_list_trials_filters_by_after_trial(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name=NEWER_NAME,
        n_trials=EXPECTED_N_TRIALS,
    )

    rows = list_trials(root, NEWER_NAME, after_trial=AFTER_TRIAL_FILTER)

    assert [r.number for r in rows] == list(range(AFTER_TRIAL_FILTER + 1, EXPECTED_N_TRIALS))


def test_list_trials_raises_for_unknown_name(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_hpo_study(root / "studies" / "main" / HPO_SUBDIR, name=NEWER_NAME)

    with pytest.raises(HpoStudyNotFoundError):
        list_trials(root, "missing_hpo")
