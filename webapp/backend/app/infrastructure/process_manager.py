"""Spawn + supervise CLI subprocesses; fan-out log + status + trial frames to WS clients."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from src.core import json_io
from src.core.fs import ensure_parent_dir
from src.core.persistence import EXPERIMENT_MANIFEST_JSON, HPO_SUBDIR
from webapp.backend.app.infrastructure.event_broker import EventBroker
from webapp.backend.app.infrastructure.log_tailer import tail_log
from webapp.backend.app.infrastructure.store import iter_run_dirs
from webapp.backend.app.schemas.hpo import TrialFrame
from webapp.backend.app.schemas.jobs import (
    JobKind,
    JobLogFrame,
    JobStatus,
    JobStatusFrame,
    JobStreamFrame,
)

CANCEL_GRACE_SECONDS = 10.0

logger = logging.getLogger(__name__)


JobEventBroker = EventBroker[JobStreamFrame]
HpoEventBroker = EventBroker[TrialFrame]


OnCompleteCallback = Callable[[str, JobStatus, int | None, str | None], Awaitable[None]]


@dataclass(frozen=True)
class TrialTailSpec:
    """Inputs the trials.jsonl tailer needs to publish ``TrialFrame``s."""

    study_name: str
    trial_jsonl_path: Path


@dataclass
class _RunningProcess:
    process: subprocess.Popen[bytes]
    watch_task: asyncio.Task[None]
    tail_tasks: tuple[asyncio.Task[None], ...]
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


def build_tune_command(
    *,
    experiment_config_path: Path,
    hpo_config_path: Path,
    store_root: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "scripts.experiment",
        "tune",
        "--config",
        str(experiment_config_path),
        "--hpo-config",
        str(hpo_config_path),
        "--store-root",
        str(store_root),
        "--no-progress",
    )


def _resolve_run_experiment_id(store_root: Path, job_id: str) -> str | None:
    """Scan run manifests for ``manifest.name == job_id``."""
    for run_dir in iter_run_dirs(store_root):
        try:
            manifest = json_io.read_dict(run_dir / EXPERIMENT_MANIFEST_JSON)
        except FileNotFoundError:
            continue
        if manifest.get("name") == job_id:
            return run_dir.name
    return None


def _resolve_tune_experiment_id(store_root: Path, study_name: str) -> str | None:
    """For TUNE jobs the experiment_id IS the study_name once the dir exists."""
    if (store_root / HPO_SUBDIR / study_name).is_dir():
        return study_name
    return None


def _resolve_experiment_id(
    kind: JobKind,
    store_root: Path,
    job_id: str,
    study_name: str | None,
) -> str | None:
    """Single dispatch for "which artifact directory belongs to this finished job?"."""
    if kind is JobKind.RUN:
        return _resolve_run_experiment_id(store_root, job_id)
    if kind is JobKind.TUNE:
        if study_name is None:
            return None
        return _resolve_tune_experiment_id(store_root, study_name)
    return None


class ProcessManager:
    def __init__(
        self,
        broker: JobEventBroker,
        on_complete: OnCompleteCallback,
        *,
        hpo_broker: HpoEventBroker | None = None,
    ) -> None:
        self._broker = broker
        self._on_complete = on_complete
        self._hpo_broker = hpo_broker
        self._running: dict[str, _RunningProcess] = {}

    async def spawn(
        self,
        *,
        job_id: str,
        kind: JobKind,
        command: tuple[str, ...],
        log_path: Path,
        store_root: Path,
        cwd: Path | None = None,
        trial_tail: TrialTailSpec | None = None,
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
        study_name = trial_tail.study_name if trial_tail is not None else None
        watch_task = asyncio.create_task(
            self._watch(job_id, kind, process, stop_event, store_root, study_name)
        )
        log_tail_task = asyncio.create_task(self._tail(job_id, log_path, stop_event))
        tail_tasks: list[asyncio.Task[None]] = [log_tail_task]
        if trial_tail is not None and self._hpo_broker is not None:
            tail_tasks.append(
                asyncio.create_task(
                    self._tail_trials(
                        trial_tail.study_name, trial_tail.trial_jsonl_path, stop_event
                    )
                )
            )
        self._running[job_id] = _RunningProcess(
            process=process,
            watch_task=watch_task,
            tail_tasks=tuple(tail_tasks),
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
            for task in (proc.watch_task, *proc.tail_tasks):
                task.cancel()
            for task in (proc.watch_task, *proc.tail_tasks):
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            await self._broker.close(job_id)
        self._running.clear()

    async def _watch(
        self,
        job_id: str,
        kind: JobKind,
        process: subprocess.Popen[bytes],
        stop_event: asyncio.Event,
        store_root: Path,
        study_name: str | None,
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
        running = self._running.get(job_id)
        if running is not None:
            for task in running.tail_tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        experiment_id = await asyncio.to_thread(
            _resolve_experiment_id, kind, store_root, job_id, study_name
        )
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
        if study_name is not None and self._hpo_broker is not None:
            await self._hpo_broker.close(study_name)
        self._running.pop(job_id, None)

    async def _tail(self, job_id: str, log_path: Path, stop_event: asyncio.Event) -> None:
        try:
            async for line in tail_log(log_path, stop=stop_event):
                await self._broker.publish(job_id, JobLogFrame(line=line))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("log tailer crashed for job %s", job_id)

    async def _tail_trials(
        self,
        study_name: str,
        trial_jsonl_path: Path,
        stop_event: asyncio.Event,
    ) -> None:
        # Defer the import: hpo_service pulls in optuna + reporters via its
        # transitive imports, and most processes don't tune.
        from webapp.backend.app.services.hpo_service import trial_row_from_record

        broker = self._hpo_broker
        if broker is None:
            return
        try:
            async for line in tail_log(trial_jsonl_path, stop=stop_event):
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("malformed trials.jsonl line for study %s: %r", study_name, line)
                    continue
                if not isinstance(parsed, dict):
                    logger.warning(
                        "non-object trials.jsonl line for study %s: %r", study_name, line
                    )
                    continue
                row = trial_row_from_record(parsed)
                await broker.publish(study_name, TrialFrame(trial=row))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("trial tailer crashed for study %s", study_name)

    @staticmethod
    def _classify_exit(exit_code: int) -> JobStatus:
        if exit_code == 0:
            return JobStatus.COMPLETED
        # POSIX: negative exit_code = killed by signal -N. SIGTERM/SIGKILL
        # mean cancellation; any other signal or non-zero code is failure.
        if exit_code in (-signal.SIGTERM, -signal.SIGKILL):
            return JobStatus.CANCELLED
        return JobStatus.FAILED
