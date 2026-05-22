"""Unit tests for :mod:`src.orchestration.clean`.

Validates the dry-run/apply split, the ``thesis_demo`` preservation,
the ``--keep`` extension, and the git-tracked-file refusal. Each test
synthesises a fake ``store_root`` on ``tmp_path`` so we never touch the
real ``experiment_results/`` tree.

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
    """Create a few empty subdirs under ``store_root`` for the planner to walk."""
    store_root.mkdir(parents=True, exist_ok=True)
    for name in dirs:
        (store_root / name).mkdir()
        # Add a sentinel file so the size calculator returns nonzero.
        (store_root / name / "stale.txt").write_text("stale", encoding="utf-8")


def test_plan_clean_lists_every_subdir_except_thesis_demo(tmp_path: Path) -> None:
    """``thesis_demo`` is hard-preserved; everything else is a candidate."""
    store = tmp_path / "experiment_results"
    _populate_store(store, ("thesis_demo", "_profile_a", "_profile_b", _KEEP_OVERRIDE_NAME))
    plan = plan_clean(store)
    candidate_names = {c.path.name for c in plan.candidates}
    assert candidate_names == {"_profile_a", "_profile_b", _KEEP_OVERRIDE_NAME}
    assert "thesis_demo" in plan.preserved


def test_plan_clean_extends_preserve_set_with_keep(tmp_path: Path) -> None:
    """``--keep`` adds to the preserve set; the matching dir drops out of candidates."""
    store = tmp_path / "experiment_results"
    _populate_store(store, ("thesis_demo", "_profile_a", _KEEP_OVERRIDE_NAME))
    plan = plan_clean(store, keep=(_KEEP_OVERRIDE_NAME,))
    candidate_names = {c.path.name for c in plan.candidates}
    assert candidate_names == {"_profile_a"}
    assert _KEEP_OVERRIDE_NAME in plan.preserved


def test_apply_clean_wipes_safe_candidates_keeping_empty_dirs(tmp_path: Path) -> None:
    """Happy path: candidate dirs survive as empty placeholders; only their contents go."""
    store = tmp_path / "experiment_results"
    _populate_store(store, ("thesis_demo", "_profile_a", "_profile_b"))
    plan = plan_clean(store)
    wiped = apply_clean(plan)
    assert sorted(p.name for p in wiped) == ["_profile_a", "_profile_b"]
    assert (store / "_profile_a").is_dir()
    assert (store / "_profile_b").is_dir()
    assert list((store / "_profile_a").iterdir()) == []
    assert list((store / "_profile_b").iterdir()) == []
    assert (store / "thesis_demo").is_dir()


def test_apply_clean_refuses_when_tracked_files_present(tmp_path: Path) -> None:
    """A directory with a git-tracked file forces a hard error from ``apply``."""
    # Bootstrap a git repo on tmp_path so ls-files is meaningful.
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

    # Add an ephemeral dir alongside.
    (store / "_ephemeral").mkdir()
    (store / "_ephemeral" / "x.txt").write_text("x", encoding="utf-8")

    plan = plan_clean(store, repo_root=tmp_path)
    refused_names = {c.path.name for c in plan.refused}
    deletable_names = {c.path.name for c in plan.deletable}
    assert refused_names == {"committed_dir"}
    assert deletable_names == {"_ephemeral"}

    with pytest.raises(ValueError, match="git-tracked"):
        apply_clean(plan)
    # Refused dir must remain on disk after the failed apply.
    assert (store / "committed_dir").is_dir()


def test_format_plan_empty_store(tmp_path: Path) -> None:
    """Missing or empty store → human-readable 'nothing to wipe' line."""
    store = tmp_path / "nonexistent"
    plan = plan_clean(store)
    output = format_plan(plan)
    assert "nothing to wipe" in output


def test_format_plan_lists_size_and_action(tmp_path: Path) -> None:
    """Dry-run text shows WIPE / REFUSE markers + a tally line."""
    store = tmp_path / "experiment_results"
    _populate_store(store, ("thesis_demo", "_profile_a"))
    plan = plan_clean(store)
    output = format_plan(plan)
    assert "WIPE" in output
    assert "_profile_a" in output
    assert "would free" in output


def test_plan_clean_ignores_files_at_top_level(tmp_path: Path) -> None:
    """A stray top-level file (e.g., README) is left alone — only dirs are candidates."""
    store = tmp_path / "experiment_results"
    store.mkdir()
    (store / "README.md").write_text("notes", encoding="utf-8")
    (store / "_dir").mkdir()
    plan = plan_clean(store)
    candidate_names = {c.path.name for c in plan.candidates}
    assert candidate_names == {"_dir"}
    assert (store / "README.md").is_file()
