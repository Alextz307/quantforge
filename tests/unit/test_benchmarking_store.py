"""JSONL store round-trip and filter-query tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmarking.store import BenchmarkStore
from src.benchmarking.types import BenchmarkRun
from tests.conftest import make_benchmark_hardware, make_benchmark_result, make_benchmark_run

PRE_TIMESTAMP = "2026-04-01T12:00:00Z"
POST_TIMESTAMP = "2026-04-20T12:00:00Z"
PRE_RUN_ID = "2026-04-01T12-00-00Z_aaa1234"
POST_RUN_ID = "2026-04-20T12-00-00Z_bbb5678"
SHA_A = "aaa1234deadbeef"
SHA_B = "bbb5678deadbeef"

BASELINE_NAME = "pre-optimization"


def _run(run_id: str, timestamp: str, sha: str, tags: tuple[str, ...] = ()) -> BenchmarkRun:
    return make_benchmark_run(
        (make_benchmark_result("BM_RSI/10000"),),
        run_id=run_id,
        timestamp=timestamp,
        tags=tags,
        hardware=make_benchmark_hardware(git_sha=sha),
    )


def test_save_and_load_run_round_trip(tmp_path: Path) -> None:
    store = BenchmarkStore(tmp_path)
    run = _run(PRE_RUN_ID, PRE_TIMESTAMP, SHA_A)
    path = store.save_run(run)
    assert path.exists()
    loaded = store.load_run(path)
    assert loaded == run


def test_baseline_round_trip(tmp_path: Path) -> None:
    store = BenchmarkStore(tmp_path)
    run = _run(PRE_RUN_ID, PRE_TIMESTAMP, SHA_A)
    store.save_baseline(run, BASELINE_NAME)
    assert store.list_baselines() == (BASELINE_NAME,)
    loaded = store.load_baseline(BASELINE_NAME)
    assert loaded == run


def test_baseline_refuses_to_overwrite_without_flag(tmp_path: Path) -> None:
    store = BenchmarkStore(tmp_path)
    run = _run(PRE_RUN_ID, PRE_TIMESTAMP, SHA_A)
    store.save_baseline(run, BASELINE_NAME)
    with pytest.raises(FileExistsError, match=BASELINE_NAME):
        store.save_baseline(run, BASELINE_NAME)
    store.save_baseline(run, BASELINE_NAME, overwrite=True)  # no raise


def test_load_missing_baseline_raises(tmp_path: Path) -> None:
    store = BenchmarkStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing-name"):
        store.load_baseline("missing-name")


def test_load_runs_filters_by_since_tag_commit(tmp_path: Path) -> None:
    store = BenchmarkStore(tmp_path)
    pre = _run(PRE_RUN_ID, PRE_TIMESTAMP, SHA_A, tags=("pre-optimization",))
    post = _run(POST_RUN_ID, POST_TIMESTAMP, SHA_B, tags=("post-optimization",))
    store.save_run(pre)
    store.save_run(post)

    since_all = store.load_runs(since="2026-01-01T00:00:00Z")
    assert {r.run_id for r in since_all} == {PRE_RUN_ID, POST_RUN_ID}

    since_post = store.load_runs(since="2026-04-10T00:00:00Z")
    assert {r.run_id for r in since_post} == {POST_RUN_ID}

    tagged = store.load_runs(tags=["pre-optimization"])
    assert {r.run_id for r in tagged} == {PRE_RUN_ID}

    by_commit = store.load_runs(commit="bbb5678")
    assert {r.run_id for r in by_commit} == {POST_RUN_ID}
