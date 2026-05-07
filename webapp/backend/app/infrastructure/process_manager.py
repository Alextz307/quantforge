"""Spawn + supervise CLI subprocesses; fan-out log + status frames to WS clients."""

from __future__ import annotations

import asyncio
import logging
import signal
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from src.core import json_io
from src.core.fs import ensure_parent_dir
from src.core.persistence import EXPERIMENT_MANIFEST_JSON
from webapp.backend.app.infrastructure.log_tailer import tail_log
from webapp.backend.app.infrastructure.store import iter_run_dirs
from webapp.backend.app.schemas.jobs import (
    JobLogFrame,
    JobStatus,
    JobStatusFrame,
    JobStreamFrame,
)

CANCEL_GRACE_SECONDS = 10.0

logger = logging.getLogger(__name__)


OnCompleteCallback = Callable[[str, JobStatus, int | None, str | None], Awaitable[None]]


class JobEventBroker:
    """Per-job pub/sub; queues unbounded so a slow client can't backpressure the producer."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[JobStreamFrame | None]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> asyncio.Queue[JobStreamFrame | None]:
        async with self._lock:
            queue: asyncio.Queue[JobStreamFrame | None] = asyncio.Queue()
            self._subscribers.setdefault(job_id, []).append(queue)
            return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue[JobStreamFrame | None]) -> None:
        async with self._lock:
            queues = self._subscribers.get(job_id)
            if queues is not None and queue in queues:
                queues.remove(queue)
                if not queues:
                    del self._subscribers[job_id]

    async def publish(self, job_id: str, frame: JobStreamFrame) -> None:
        async with self._lock:
            queues = list(self._subscribers.get(job_id, []))
        for queue in queues:
            queue.put_nowait(frame)

    async def close(self, job_id: str) -> None:
        async with self._lock:
            queues = self._subscribers.pop(job_id, [])
        for queue in queues:
            queue.put_nowait(None)


@dataclass
class _RunningProcess:
    process: subprocess.Popen[bytes]
    watch_task: asyncio.Task[None]
    tail_task: asyncio.Task[None]
    stop_event: asyncio.Event


def build_run_command(*, config_path: Path, job_id: str, store_root: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "scripts.experiment",
        "run",
        "--config",
        str(config_path),
        "--name",
        job_id,
        "--store-root",
        str(store_root),
        "--no-progress",
    )


def _resolve_experiment_id(store_root: Path, job_id: str) -> str | None:
    """Resolve a finished job's run dir basename by scanning manifests for ``name == job_id``."""
    for run_dir in iter_run_dirs(store_root):
        try:
            manifest = json_io.read_dict(run_dir / EXPERIMENT_MANIFEST_JSON)
        except FileNotFoundError:
            continue
        if manifest.get("name") == job_id:
            return run_dir.name
    return None


class ProcessManager:
    def __init__(self, broker: JobEventBroker, on_complete: OnCompleteCallback) -> None:
        self._broker = broker
        self._on_complete = on_complete
        self._running: dict[str, _RunningProcess] = {}

    async def spawn(
        self,
        *,
        job_id: str,
        command: tuple[str, ...],
        log_path: Path,
        store_root: Path,
        cwd: Path | None = None,
    ) -> int:
        # buffering=0 + dup'd FD: child writes raw, parent closes immediately so
        # the tailer sees lines as soon as the child flushes.
        log_handle = ensure_parent_dir(log_path).open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=cwd,
            )
        finally:
            log_handle.close()
        stop_event = asyncio.Event()
        watch_task = asyncio.create_task(self._watch(job_id, process, stop_event, store_root))
        tail_task = asyncio.create_task(self._tail(job_id, log_path, stop_event))
        self._running[job_id] = _RunningProcess(
            process=process,
            watch_task=watch_task,
            tail_task=tail_task,
            stop_event=stop_event,
        )
        return process.pid

    def is_alive(self, job_id: str) -> bool:
        proc = self._running.get(job_id)
        return proc is not None and proc.process.poll() is None

    async def cancel(self, job_id: str) -> bool:
        """Send SIGTERM, escalate to SIGKILL after the grace period."""
        proc = self._running.get(job_id)
        if proc is None or proc.process.poll() is not None:
            return False
        try:
            proc.process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return False
        try:
            await asyncio.wait_for(
                asyncio.to_thread(proc.process.wait),
                timeout=CANCEL_GRACE_SECONDS,
            )
        except TimeoutError:
            proc.process.kill()
            await asyncio.to_thread(proc.process.wait)
        return True

    async def shutdown(self) -> None:
        for job_id, proc in list(self._running.items()):
            if proc.process.poll() is None:
                try:
                    proc.process.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
            proc.stop_event.set()
            for task in (proc.watch_task, proc.tail_task):
                task.cancel()
            for task in (proc.watch_task, proc.tail_task):
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            await self._broker.close(job_id)
        self._running.clear()

    async def _watch(
        self,
        job_id: str,
        process: subprocess.Popen[bytes],
        stop_event: asyncio.Event,
        store_root: Path,
    ) -> None:
        try:
            exit_code = await asyncio.to_thread(process.wait)
        except asyncio.CancelledError:
            stop_event.set()
            raise
        # One extra tick lets the tail loop flush lines that landed between
        # the last poll and process exit.
        await asyncio.sleep(0)
        stop_event.set()
        try:
            await self._running[job_id].tail_task
        except (KeyError, asyncio.CancelledError):
            pass
        experiment_id = await asyncio.to_thread(_resolve_experiment_id, store_root, job_id)
        status = self._classify_exit(exit_code)
        try:
            await self._on_complete(job_id, status, exit_code, experiment_id)
        except Exception:
            logger.exception("on_complete callback raised for job %s", job_id)
        await self._broker.publish(
            job_id,
            JobStatusFrame(status=status, exit_code=exit_code, experiment_id=experiment_id),
        )
        await self._broker.close(job_id)
        self._running.pop(job_id, None)

    async def _tail(self, job_id: str, log_path: Path, stop_event: asyncio.Event) -> None:
        try:
            async for line in tail_log(log_path, stop=stop_event):
                await self._broker.publish(job_id, JobLogFrame(line=line))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("log tailer crashed for job %s", job_id)

    @staticmethod
    def _classify_exit(exit_code: int) -> JobStatus:
        if exit_code == 0:
            return JobStatus.COMPLETED
        # POSIX: negative exit_code = killed by signal -N. SIGTERM/SIGKILL
        # mean cancellation; any other signal or non-zero code is failure.
        if exit_code in (-signal.SIGTERM, -signal.SIGKILL):
            return JobStatus.CANCELLED
        return JobStatus.FAILED
