"""Real-spawn lifecycle tests for ProcessManager + JobEventBroker."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from src.core import json_io
from src.core.persistence import EXPERIMENT_MANIFEST_JSON, RUNS_SUBDIR
from webapp.backend.app.infrastructure.process_manager import (
    JobEventBroker,
    ProcessManager,
    _resolve_experiment_id,
)
from webapp.backend.app.schemas.jobs import (
    JobLogFrame,
    JobStatus,
    JobStatusFrame,
    JobStreamFrame,
)

JOB_ID = "fake-job-id"
COMPLETION_TIMEOUT = 15.0
CANCEL_SLEEP_SECONDS = 60


async def _drain_queue(
    queue: asyncio.Queue[JobStreamFrame | None],
) -> list[JobStreamFrame]:
    frames: list[JobStreamFrame] = []
    while True:
        frame = await asyncio.wait_for(queue.get(), timeout=COMPLETION_TIMEOUT)
        if frame is None:
            return frames
        frames.append(frame)


def _split(frames: list[JobStreamFrame]) -> tuple[list[JobLogFrame], list[JobStatusFrame]]:
    logs = [f for f in frames if isinstance(f, JobLogFrame)]
    statuses = [f for f in frames if isinstance(f, JobStatusFrame)]
    return logs, statuses


def test_spawn_streams_logs_and_completes(tmp_path: Path) -> None:
    log_path = tmp_path / "job.log"
    completions: list[tuple[str, JobStatus, int | None, str | None]] = []

    async def on_complete(
        job_id: str,
        status: JobStatus,
        exit_code: int | None,
        experiment_id: str | None,
    ) -> None:
        completions.append((job_id, status, exit_code, experiment_id))

    async def scenario() -> tuple[list[JobLogFrame], list[JobStatusFrame]]:
        broker = JobEventBroker()
        manager = ProcessManager(broker, on_complete=on_complete)
        queue = await broker.subscribe(JOB_ID)
        cmd = (sys.executable, "-c", "print('hello'); print('world')")
        await manager.spawn(job_id=JOB_ID, command=cmd, log_path=log_path, store_root=tmp_path)
        frames = await _drain_queue(queue)
        return _split(frames)

    logs, statuses = asyncio.run(scenario())
    assert "hello" in [log.line for log in logs]
    assert "world" in [log.line for log in logs]
    assert len(statuses) == 1
    assert statuses[0].status is JobStatus.COMPLETED
    assert statuses[0].exit_code == 0
    assert completions[0][1] is JobStatus.COMPLETED


def test_cancel_sigterms_running_process(tmp_path: Path) -> None:
    log_path = tmp_path / "job.log"
    completions: list[tuple[str, JobStatus, int | None, str | None]] = []

    async def on_complete(
        job_id: str,
        status: JobStatus,
        exit_code: int | None,
        experiment_id: str | None,
    ) -> None:
        completions.append((job_id, status, exit_code, experiment_id))

    async def scenario() -> JobStatusFrame:
        broker = JobEventBroker()
        manager = ProcessManager(broker, on_complete=on_complete)
        queue = await broker.subscribe(JOB_ID)
        cmd = (sys.executable, "-c", f"import time; time.sleep({CANCEL_SLEEP_SECONDS})")
        await manager.spawn(job_id=JOB_ID, command=cmd, log_path=log_path, store_root=tmp_path)
        await asyncio.sleep(0.2)
        cancelled = await manager.cancel(JOB_ID)
        assert cancelled is True
        frames = await _drain_queue(queue)
        _, statuses = _split(frames)
        return statuses[-1]

    final = asyncio.run(scenario())
    assert final.status is JobStatus.CANCELLED
    assert completions[0][1] is JobStatus.CANCELLED


def test_failed_exit_code_classified_as_failed(tmp_path: Path) -> None:
    log_path = tmp_path / "job.log"

    async def on_complete(
        job_id: str,
        status: JobStatus,
        exit_code: int | None,
        experiment_id: str | None,
    ) -> None:
        return None

    async def scenario() -> JobStatusFrame:
        broker = JobEventBroker()
        manager = ProcessManager(broker, on_complete=on_complete)
        queue = await broker.subscribe(JOB_ID)
        cmd = (sys.executable, "-c", "import sys; sys.exit(2)")
        await manager.spawn(job_id=JOB_ID, command=cmd, log_path=log_path, store_root=tmp_path)
        frames = await _drain_queue(queue)
        _, statuses = _split(frames)
        return statuses[-1]

    final = asyncio.run(scenario())
    assert final.status is JobStatus.FAILED
    assert final.exit_code == 2


def test_resolve_experiment_id_finds_run_by_name(tmp_path: Path) -> None:
    runs_dir = tmp_path / RUNS_SUBDIR
    expected_id = "20260101_120000_TestStrategy_abc1234_deadbeef"
    run_dir = runs_dir / expected_id
    run_dir.mkdir(parents=True)
    json_io.write(
        run_dir / EXPERIMENT_MANIFEST_JSON,
        {"name": JOB_ID, "experiment_id": expected_id},
    )

    assert _resolve_experiment_id(tmp_path, JOB_ID) == expected_id
    assert _resolve_experiment_id(tmp_path, "other-job") is None


def test_resolve_experiment_id_returns_none_when_runs_dir_absent(tmp_path: Path) -> None:
    assert _resolve_experiment_id(tmp_path, JOB_ID) is None


@pytest.mark.parametrize("job_id", [JOB_ID, "other"])
def test_broker_unsubscribe_removes_queue(job_id: str) -> None:
    async def scenario() -> int:
        broker = JobEventBroker()
        queue = await broker.subscribe(job_id)
        await broker.unsubscribe(job_id, queue)
        return len(broker._subscribers.get(job_id, []))  # noqa: SLF001

    assert asyncio.run(scenario()) == 0


def test_broker_close_signals_all_subscribers() -> None:
    async def scenario() -> tuple[object, object]:
        broker = JobEventBroker()
        q1 = await broker.subscribe(JOB_ID)
        q2 = await broker.subscribe(JOB_ID)
        await broker.close(JOB_ID)
        return await q1.get(), await q2.get()

    result = asyncio.run(scenario())
    assert result == (None, None)
