"""
Unit tests for :mod:`src.orchestration.clean`.

Validates the dry-run/apply split, the ``--keep`` preserve set, and the
git-tracked-file refusal. Each test synthesises a fake ``store_root`` on
``tmp_path`` so we never touch the real ``experiment_results/`` tree.

The git-tracked-file case is exercised via a fresh ``git init`` repo on
``tmp_path`` rather than the parent repo's git state, so the test stays
hermetic and runs equally on CI and dev machines.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.orchestration.clean import (
    apply_clean,
    format_plan,
    plan_clean,
)

_KEEP_OVERRIDE_NAME = "studies"


def _populate_store(store_root: Path, dirs: tuple[str, ...]) -> None:
    """
    Create a few empty subdirs under ``store_root`` for the planner to walk.
    """

    store_root.mkdir(parents=True, exist_ok=True)
    for name in dirs:
        (store_root / name).mkdir()
        (store_root / name / "stale.txt").write_text("stale", encoding="utf-8")


def test_plan_clean_lists_every_subdir_by_default(tmp_path: Path) -> None:
    """
    With no ``--keep``, every immediate-child directory is a candidate.
    """

    store = tmp_path / "experiment_results"
    _populate_store(store, ("_profile_a", "_profile_b", _KEEP_OVERRIDE_NAME))
    plan = plan_clean(store)
    candidate_names = {c.path.name for c in plan.candidates}
    assert candidate_names == {"_profile_a", "_profile_b", _KEEP_OVERRIDE_NAME}
    assert plan.preserved == ()


def test_plan_clean_preserves_keep_set(tmp_path: Path) -> None:
    """
    ``--keep`` names are preserved; the matching dir drops out of candidates.
    """

    store = tmp_path / "experiment_results"
    _populate_store(store, ("_profile_a", "_profile_b", _KEEP_OVERRIDE_NAME))
    plan = plan_clean(store, keep=(_KEEP_OVERRIDE_NAME,))
    candidate_names = {c.path.name for c in plan.candidates}
    assert candidate_names == {"_profile_a", "_profile_b"}
    assert _KEEP_OVERRIDE_NAME in plan.preserved


def test_apply_clean_wipes_safe_candidates_keeping_empty_dirs(tmp_path: Path) -> None:
    """
    Happy path: candidate dirs survive as empty placeholders; only their contents go.
    """

    store = tmp_path / "experiment_results"
    _populate_store(store, ("_profile_a", "_profile_b"))
    plan = plan_clean(store)
    wiped = apply_clean(plan)
    assert sorted(p.name for p in wiped) == ["_profile_a", "_profile_b"]
    assert (store / "_profile_a").is_dir()
    assert (store / "_profile_b").is_dir()
    assert list((store / "_profile_a").iterdir()) == []
    assert list((store / "_profile_b").iterdir()) == []


def test_apply_clean_refuses_when_tracked_files_present(tmp_path: Path) -> None:
    """
    A directory with a git-tracked file forces a hard error from ``apply``.
    """

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "x@y.z"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "x"], cwd=tmp_path, check=True)

    store = tmp_path / "experiment_results"
    store.mkdir()
    committed = store / "committed_dir"
    committed.mkdir()
    (committed / "file.txt").write_text("tracked", encoding="utf-8")
    subprocess.run(
        ["git", "add", "experiment_results/committed_dir/file.txt"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "commit", "-q", "-m", "track"], cwd=tmp_path, check=True)

    (store / "_ephemeral").mkdir()
    (store / "_ephemeral" / "x.txt").write_text("x", encoding="utf-8")

    plan = plan_clean(store, repo_root=tmp_path)
    refused_names = {c.path.name for c in plan.refused}
    deletable_names = {c.path.name for c in plan.deletable}
    assert refused_names == {"committed_dir"}
    assert deletable_names == {"_ephemeral"}

    with pytest.raises(ValueError, match="git-tracked"):
        apply_clean(plan)
    assert (store / "committed_dir").is_dir()


def test_format_plan_empty_store(tmp_path: Path) -> None:
    """
    Missing or empty store -> human-readable 'nothing to wipe' line.
    """

    store = tmp_path / "nonexistent"
    plan = plan_clean(store)
    output = format_plan(plan)
    assert "nothing to wipe" in output


def test_format_plan_lists_size_and_action(tmp_path: Path) -> None:
    """
    Dry-run text shows WIPE / REFUSE markers + a tally line.
    """

    store = tmp_path / "experiment_results"
    _populate_store(store, ("_profile_a", "_profile_b"))
    plan = plan_clean(store)
    output = format_plan(plan)
    assert "WIPE" in output
    assert "_profile_a" in output
    assert "would free" in output


def test_plan_clean_ignores_files_at_top_level(tmp_path: Path) -> None:
    """
    A stray top-level file (e.g., README) is left alone - only dirs and the
    sweep-tracking allowlist are candidates.
    """

    store = tmp_path / "experiment_results"
    store.mkdir()
    (store / "README.md").write_text("notes", encoding="utf-8")
    (store / "_dir").mkdir()
    plan = plan_clean(store)
    candidate_names = {c.path.name for c in plan.candidates}
    assert candidate_names == {"_dir"}
    assert plan.stray_files == ()
    assert (store / "README.md").is_file()


_STRAY_FILE_NAMES = (".sweep_pid", ".sweep_started_at", ".sweep_log_path", "sweep_2026-05-22.log")
_PRESERVED_TOP_LEVEL_FILE = "README.md"


def _populate_top_level_files(store_root: Path) -> None:
    """
    Drop the stray sweep-tracking allowlist plus one preserved file.
    """

    store_root.mkdir(parents=True, exist_ok=True)
    for name in (*_STRAY_FILE_NAMES, _PRESERVED_TOP_LEVEL_FILE):
        (store_root / name).write_text("x", encoding="utf-8")


def test_plan_clean_collects_stray_sweep_files(tmp_path: Path) -> None:
    """
    The sweep-tracking allowlist is collected; other top-level files are not.
    """

    store = tmp_path / "experiment_results"
    _populate_top_level_files(store)
    plan = plan_clean(store)
    assert {p.name for p in plan.stray_files} == set(_STRAY_FILE_NAMES)
    assert _PRESERVED_TOP_LEVEL_FILE not in {p.name for p in plan.stray_files}


def test_apply_clean_removes_stray_files_keeps_other_top_level_files(tmp_path: Path) -> None:
    """
    Apply unlinks the allowlist files and leaves every other top-level file.
    """

    store = tmp_path / "experiment_results"
    _populate_top_level_files(store)
    plan = plan_clean(store)
    apply_clean(plan)
    for name in _STRAY_FILE_NAMES:
        assert not (store / name).exists()
    assert (store / _PRESERVED_TOP_LEVEL_FILE).is_file()


def test_plan_clean_keep_protects_stray_file_name(tmp_path: Path) -> None:
    """
    A ``--keep`` entry matching a stray file name excludes it from removal.
    """

    store = tmp_path / "experiment_results"
    _populate_top_level_files(store)
    plan = plan_clean(store, keep=(".sweep_pid",))
    assert ".sweep_pid" not in {p.name for p in plan.stray_files}
    assert ".sweep_log_path" in {p.name for p in plan.stray_files}


def test_plan_clean_leaves_tracked_stray_file(tmp_path: Path) -> None:
    """
    A git-tracked file matching the allowlist is left in place, not removed.
    """

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "x@y.z"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "x"], cwd=tmp_path, check=True)

    store = tmp_path / "experiment_results"
    store.mkdir()
    tracked_log = store / "sweep_tracked.log"
    tracked_log.write_text("tracked", encoding="utf-8")
    subprocess.run(
        ["git", "add", "experiment_results/sweep_tracked.log"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "commit", "-q", "-m", "track"], cwd=tmp_path, check=True)

    plan = plan_clean(store, repo_root=tmp_path)
    assert plan.stray_files == ()
    apply_clean(plan)
    assert tracked_log.is_file()


def test_format_plan_shows_stray_files(tmp_path: Path) -> None:
    """
    Dry-run text lists the stray files and counts them in the tally line.
    """

    store = tmp_path / "experiment_results"
    _populate_top_level_files(store)
    output = format_plan(plan_clean(store))
    assert "RM-FILE" in output
    assert ".sweep_pid" in output
    assert "stray file(s)" in output
