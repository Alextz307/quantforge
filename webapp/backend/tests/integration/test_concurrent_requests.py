"""
Regression test: parallel requests must not trip SQLite's same-thread guard.

FastAPI resolves a `with`-yielding sync dependency in its threadpool, then may
invoke the consuming endpoint on a different worker. Without
`check_same_thread=False` the second touch on the connection raises
`sqlite3.ProgrammingError`. This surfaced when the run-detail page started
firing /api/runs/{id} and /api/runs/{id}/folds in parallel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path

from fastapi.testclient import TestClient

FLAT_ID = "20260101_120000_AdaptiveBollinger_abc1234_deadbeef"
STUDY_ID = "20260201_090000_PairsTrading_def5678_cafebabe"
PARALLEL_WORKERS = 8


def test_parallel_run_detail_requests_do_not_trip_threadpool(
    webapp_store: Path,
    authed_client: TestClient,
) -> None:
    targets = [
        f"/api/runs/{FLAT_ID}",
        f"/api/runs/{FLAT_ID}/folds",
        f"/api/runs/{STUDY_ID}",
        f"/api/runs/{STUDY_ID}/folds",
    ] * 2

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        responses = list(executor.map(authed_client.get, targets))

    for path, response in zip(targets, responses, strict=True):
        assert response.status_code == HTTPStatus.OK, f"{path} -> {response.status_code}"
