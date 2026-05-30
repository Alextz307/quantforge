"""
Contextual logger wrapper - prefixes every message with caller-bound context.

Pure stdlib, no structlog / loguru. ``get_logger(name, **context)`` returns a
:class:`logging.LoggerAdapter` whose ``process()`` prepends the bound key/value
pairs to every emitted message:

    logger = get_logger(__name__, experiment_id="abc", strategy_name="Strat")
    logger.info("fold 2/5 complete")
    # -> INFO src.orchestration.experiment:
    #     [experiment_id=abc strategy_name=Strat] fold 2/5 complete

This approach lets the root logging format string stay simple (no custom
``%(experiment_id)s`` attributes) while still surfacing the context in every
downstream record. Modules that don't need context keep using
``logging.getLogger(__name__)`` unchanged.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.persistence import CLI_LOGS_SUBDIR, EXPERIMENT_RUN_LOG

CLI_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _ContextAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
    """
    LoggerAdapter that formats bound context as ``[k1=v1 k2=v2] <msg>``.

    ``self.extra`` is the context dict passed to :func:`get_logger`.
    ``process()`` runs on every log call - we keep it allocation-light by
    using ``" ".join`` over a generator rather than materialising an
    intermediate list.
    """

    def process(
        self,
        msg: Any,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra = self.extra or {}
        if not extra:
            return msg, kwargs
        ctx = " ".join(f"{k}={v}" for k, v in extra.items())
        return f"[{ctx}] {msg}", kwargs


def get_logger(name: str, **context: object) -> logging.LoggerAdapter:  # type: ignore[type-arg]
    """
    Return a context-bound logger.

    ``name`` is the standard ``logging.getLogger(name)`` handle; ``context``
    becomes a persistent prefix on every message. Empty context is legal -
    the adapter degrades to a zero-cost passthrough in ``process()``.

    Return-type ``type-arg`` ignore matches the reason on ``_ContextAdapter``:
    Python 3.12 made ``LoggerAdapter`` generic but the version-portable
    parameterisation isn't stable in typeshed yet.
    """

    base = logging.getLogger(name)
    extra: Mapping[str, object] = dict(context)
    return _ContextAdapter(base, extra)


@contextmanager
def _tee_root_to_file(log_path: Path) -> Iterator[Path]:
    """
    Attach a :class:`logging.FileHandler` at ``log_path`` to the root logger
    for the duration of the context, then remove and close it.

    Shared body of :func:`attach_cli_log_file` and
    :func:`attach_run_log_file`. Single source of truth for the handler
    mode (``"a"``), encoding (``utf-8``), and formatter
    (:data:`CLI_LOG_FORMAT`) - divergence between the two public
    helpers is a drift bug.
    """

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(CLI_LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield log_path
    finally:
        root.removeHandler(handler)
        handler.close()


@contextmanager
def attach_cli_log_file(root_dir: Path, command_name: str) -> Iterator[Path]:
    """
    Tee root-logger output to a timestamped file under ``root_dir/cli_logs/``.

    Used by every persistent CLI subcommand so a dropped terminal, a
    detached ``nohup`` shell, or an after-the-fact "what actually
    happened during the 13-hour sweep?" question all have the same
    answer: read the file. ``root_dir`` is the artifact root (usually
    ``store_root`` for ``experiment``/``study`` subcommands, or
    ``study_dir`` for ``study report``); the helper joins
    :data:`~src.core.persistence.CLI_LOGS_SUBDIR` so the file always
    lands at ``<root_dir>/cli_logs/<command>_<UTC_ts>_<pid>.log``.

    Filename is ``{command_name}_{YYYYMMDD_HHMMSS}_{pid}.log`` - the
    timestamp + pid suffix lets multiple concurrent invocations land
    in the same directory without colliding.

    The handler uses :data:`CLI_LOG_FORMAT` so the file is grep-compatible
    with the stdout stream set up by ``logging.basicConfig`` at the CLI
    entry point. Removed unconditionally on exit so pytest captures and
    repeat invocations don't accumulate handlers.

    Not re-entrant: the handler is added to the *root* logger, so a
    second concurrent call in the same process would tee both files'
    output to each other. Single-shot CLI invocation only.
    """

    log_dir = root_dir / CLI_LOGS_SUBDIR
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{command_name}_{timestamp}_{os.getpid()}.log"
    with _tee_root_to_file(log_path) as p:
        yield p


@contextmanager
def attach_run_log_file(run_dir: Path) -> Iterator[Path]:
    """
    Tee root-logger output to ``run_dir/run.log`` for one experiment run.

    Companion to :func:`attach_cli_log_file`: that helper captures the
    parent CLI's orchestration stream; this one isolates a single
    ``Experiment.run()`` so each strategyxuniverse leg has its own
    end-to-end log next to its ``config.yaml`` / ``manifest.json`` /
    ``metrics.json``. Under ``ProcessPoolExecutor`` fan-out (comparison /
    study) every worker process has an independent root logger, so the
    per-leg files never interleave.

    The handler is removed unconditionally on exit; safe to call inside
    pytest captures and from sequential in-process callers, where the
    same record will also reach the parent CLI log (that duplication is
    deliberate - the CLI log stays the master record).

    Not re-entrant: a nested ``Experiment.run()`` call in the same
    process would tee both legs' output to each other's ``run.log``.
    Today's callers (`_run_sequential`, `_run_parallel` workers, study
    legs) only enter this context once per process at a time.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / EXPERIMENT_RUN_LOG
    with _tee_root_to_file(log_path) as p:
        yield p


@contextmanager
def log_stage(
    logger: logging.Logger | logging.LoggerAdapter,  # type: ignore[type-arg]
    label: str,
    **fields: object,
) -> Iterator[None]:
    """
    Bracket a unit of work with start/done INFO logs and a perf timer.

    Emits ``"<label> starting (k1=v1 k2=v2)"`` on enter (or just
    ``"<label> starting"`` when no fields are bound) and ``"<label>
    done in <secs>s"`` on exit. The duration is wall-clock via
    :func:`time.perf_counter`; the ``done`` line fires from a ``finally``
    so an exception still surfaces the elapsed time before propagating.

    Designed for stage-level pipeline work (GARCH fit, LSTM fit, ARMA
    fit) where the user wants per-stage progress in the persisted log
    file. Avoid for sub-second hot-loop work - the two extra log
    records per call dwarf the work itself.
    """

    if fields:
        ctx = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.info("%s starting (%s)", label, ctx)
    else:
        logger.info("%s starting", label)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s done in %.1fs", label, time.perf_counter() - t0)
