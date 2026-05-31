"""
Spawn + supervise CLI subprocesses; fan-out log + status frames to WS clients.
"""

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
from src.core.persistence import (
    COMPARISONS_SUBDIR,
    EXPERIMENT_MANIFEST_JSON,
    HOLDOUT_EVALS_SUBDIR,
    HPO_SUBDIR,
)
from src.orchestration.comparison import SignificanceTest
from src.orchestration.holdout_eval import SourceKind
from webapp.backend.app.infrastructure.event_broker import EventBroker
from webapp.backend.app.infrastructure.log_tailer import tail_log
from webapp.backend.app.infrastructure.store import STUDIES_SUBDIR, iter_run_dirs
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


OnCompleteCallback = Callable[[str, JobStatus, int | None, str | None], Awaitable[None]]


@dataclass
class _RunningProcess:
    process: subprocess.Popen[bytes]
    watch_task: asyncio.Task[None]
    tail_tasks: tuple[asyncio.Task[None], ...]
    stop_event: asyncio.Event


def build_run_command(
    *, config_path: Path, job_id: str, store_root: Path, feature_importance: bool = False
) -> tuple[str, ...]:
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
        *(("--feature-importance",) if feature_importance else ()),
    )


def build_importance_command(*, run_dir: Path, store_root: Path, job_id: str) -> tuple[str, ...]:
    """
    ``experiment importance`` invocation: recompute a finished run's importance.

    ``--name job_id`` makes a diverged re-run's ``manifest.name`` equal the job
    id, so :func:`_resolve_run_experiment_id` resolves the job to that new run
    (a reproduced backfill writes no new run, leaving experiment_id unresolved -
    the frontend then sees importance attached to the original run).
    """

    return (
        sys.executable,
        "-m",
        "scripts.experiment",
        "importance",
        "--run-dir",
        str(run_dir),
        "--store-root",
        str(store_root),
        "--name",
        job_id,
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


def build_compare_command(
    *,
    config_paths: tuple[Path, ...],
    reuse_run_dirs: tuple[Path, ...],
    out_name: str,
    significance_test: SignificanceTest,
    n_jobs: int,
    write_report: bool,
    publish_label: str | None,
    store_root: Path,
) -> tuple[str, ...]:
    """
    ``experiment compare`` invocation in ``--reuse-runs`` mode.

    ``config_paths`` and ``reuse_run_dirs`` must be the same length and in
    matching order - the CLI pairs them positionally.
    """

    cmd: list[str] = [
        sys.executable,
        "-m",
        "scripts.experiment",
        "compare",
        "--out-name",
        out_name,
        "--significance-test",
        significance_test.value,
        "--n-jobs",
        str(n_jobs),
        "--store-root",
        str(store_root),
        "--report" if write_report else "--no-report",
        "--reuse-runs",
        ",".join(str(p) for p in reuse_run_dirs),
    ]
    for cp in config_paths:
        cmd.extend(("--config", str(cp)))
    if publish_label is not None:
        cmd.extend(("--publish-label", publish_label))
    return tuple(cmd)


def build_holdout_command(
    *,
    source_kind: SourceKind,
    source_path: Path,
    out_name: str | None,
    write_report: bool,
    publish_label: str | None,
    store_root: Path,
) -> tuple[str, ...]:
    """
    ``experiment holdout-eval`` invocation; source picks ``--run-dir`` vs ``--hpo-best``.
    """

    cmd: list[str] = [
        sys.executable,
        "-m",
        "scripts.experiment",
        "holdout-eval",
        "--store-root",
        str(store_root),
        "--report" if write_report else "--no-report",
    ]
    if source_kind == "run":
        cmd.extend(("--run-dir", str(source_path)))
    else:
        cmd.extend(("--hpo-best", str(source_path)))
    if out_name is not None:
        cmd.extend(("--out-name", out_name))
    if publish_label is not None:
        cmd.extend(("--publish-label", publish_label))
    return tuple(cmd)


def build_study_command(
    *,
    spec_path: Path,
    force_rerun: bool,
    only_legs: tuple[str, ...],
    skip_compares: bool,
    skip_holdout_eval: bool,
    store_root: Path,
) -> tuple[str, ...]:
    """
    ``experiment study run`` invocation; drives the cross-strategy x cross-universe sweep.
    """
    cmd: list[str] = [
        sys.executable,
        "-m",
        "scripts.experiment",
        "study",
        "run",
        "--spec",
        str(spec_path),
        "--store-root",
        str(store_root),
    ]
    if force_rerun:
        cmd.append("--force-rerun")
    if skip_compares:
        cmd.append("--skip-compares")
    if skip_holdout_eval:
        cmd.append("--skip-holdout-eval")
    for leg_id in only_legs:
        cmd.extend(("--only-leg", leg_id))
    return tuple(cmd)


def _resolve_run_experiment_id(store_root: Path, job_id: str) -> str | None:
    """
    Scan run manifests for ``manifest.name == job_id``.
    """

    for run_dir in iter_run_dirs(store_root):
        try:
            manifest = json_io.read_dict(run_dir / EXPERIMENT_MANIFEST_JSON)
        except FileNotFoundError:
            continue
        if manifest.get("name") == job_id:
            return run_dir.name
    return None


# Kinds whose artifact directory name is known at submission time. The watch
# task confirms the directory actually materialised once the subprocess exits
# (a CLI crash before the artifact lands leaves experiment_id = None).
_ARTIFACT_SUBDIR_BY_KIND: dict[JobKind, str] = {
    JobKind.TUNE: HPO_SUBDIR,
    JobKind.COMPARE: COMPARISONS_SUBDIR,
    JobKind.HOLDOUT: HOLDOUT_EVALS_SUBDIR,
    JobKind.STUDY: STUDIES_SUBDIR,
}


def _resolve_named_artifact_id(store_root: Path, subdir: str, artifact_name: str) -> str | None:
    if (store_root / subdir / artifact_name).is_dir():
        return artifact_name
    return None


def _resolve_experiment_id(
    kind: JobKind,
    store_root: Path,
    job_id: str,
    artifact_name: str | None,
) -> str | None:
    """
    Single dispatch for "which artifact directory belongs to this finished job?".

    RUN jobs need a manifest walk because the run dir's basename is the
    auto-generated experiment_id, not the job_id. IMPORTANCE jobs use the same
    walk: a diverged re-run is saved under ``manifest.name == job_id`` (a
    reproduced backfill writes no new run, so the walk finds nothing and the
    job's experiment_id stays None - the original run is where importance
    landed). TUNE/COMPARE/HOLDOUT pre-commit their artifact name at submission
    time so the resolver is a cheap stat.
    """

    if kind in (JobKind.RUN, JobKind.IMPORTANCE):
        return _resolve_run_experiment_id(store_root, job_id)
    subdir = _ARTIFACT_SUBDIR_BY_KIND.get(kind)
    if subdir is None or artifact_name is None:
        return None
    return _resolve_named_artifact_id(store_root, subdir, artifact_name)


class ProcessManager:
    def __init__(
        self,
        broker: JobEventBroker,
        on_complete: OnCompleteCallback,
    ) -> None:
        self._broker = broker
        self._on_complete = on_complete
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
        artifact_id: str | None = None,
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
        watch_task = asyncio.create_task(
            self._watch(job_id, kind, process, stop_event, store_root, artifact_id)
        )
        log_tail_task = asyncio.create_task(self._tail(job_id, log_path, stop_event))

        self._running[job_id] = _RunningProcess(
            process=process,
            watch_task=watch_task,
            tail_tasks=(log_tail_task,),
            stop_event=stop_event,
        )
        return process.pid

    def is_alive(self, job_id: str) -> bool:
        proc = self._running.get(job_id)
        return proc is not None and proc.process.poll() is None

    async def cancel(self, job_id: str) -> bool:
        """
        Send SIGTERM, escalate to SIGKILL after the grace period.
        """

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
        artifact_name: str | None,
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
            _resolve_experiment_id, kind, store_root, job_id, artifact_name
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
