"""Unit tests for the path-walking helpers under infrastructure/store.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from webapp.backend.app.infrastructure.store import (
    RunNotFoundError,
    find_run_dir,
    iter_run_dirs,
    store_label,
)
from webapp.backend.tests.conftest import make_synthetic_run

FLAT_ID = "20260101_120000_AdaptiveBollinger_abc1234_deadbeef"
STUDY_ID = "20260201_090000_PairsTrading_def5678_cafebabe"
EXPECTED_FLAT_LABEL = "thesis_demo/runs"
EXPECTED_STUDY_LABEL = "studies/main/runs"
EXPECTED_DEFAULT_HPO_LABEL = "hpo"


def test_iter_run_dirs_finds_both_layouts(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "thesis_demo" / "runs", experiment_id=FLAT_ID)
    make_synthetic_run(root / "studies" / "main" / "runs", experiment_id=STUDY_ID)

    found = {d.name for d in iter_run_dirs(root)}

    assert found == {FLAT_ID, STUDY_ID}


def test_iter_run_dirs_returns_empty_when_root_missing(tmp_path: Path) -> None:
    assert list(iter_run_dirs(tmp_path / "does_not_exist")) == []


def test_iter_run_dirs_skips_dirs_without_manifest(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    runs = root / "thesis_demo" / "runs"
    runs.mkdir(parents=True)
    (runs / "incomplete_run").mkdir()  # no manifest.json

    assert list(iter_run_dirs(root)) == []


def test_find_run_dir_resolves_known_id(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "thesis_demo" / "runs", experiment_id=FLAT_ID)

    run_dir = find_run_dir(root, FLAT_ID)

    assert run_dir.name == FLAT_ID


def test_find_run_dir_raises_for_unknown_id(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    make_synthetic_run(root / "thesis_demo" / "runs", experiment_id=FLAT_ID)

    with pytest.raises(RunNotFoundError):
        find_run_dir(root, "20990101_000000_Missing_0000000_00000000")


def test_store_label_for_flat_layout(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    run_dir = make_synthetic_run(root / "thesis_demo" / "runs", experiment_id=FLAT_ID)

    assert store_label(run_dir, root) == EXPECTED_FLAT_LABEL


def test_store_label_for_study_nested_layout(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    run_dir = make_synthetic_run(root / "studies" / "main" / "runs", experiment_id=STUDY_ID)

    assert store_label(run_dir, root) == EXPECTED_STUDY_LABEL


def test_store_label_for_default_root_layout(tmp_path: Path) -> None:
    root = tmp_path / "experiment_results"
    run_dir = make_synthetic_run(root / "runs", experiment_id=FLAT_ID)

    assert store_label(run_dir, root) == "runs"
