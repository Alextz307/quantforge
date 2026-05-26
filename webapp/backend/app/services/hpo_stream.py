"""Async generator tailing ``trials.jsonl``; yields ``TrialFrame``."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from src.optimization.checkpointing import TRIALS_JSONL_NAME
from webapp.backend.app.infrastructure.log_tailer import tail_log
from webapp.backend.app.schemas.hpo import TrialFrame

logger = logging.getLogger(__name__)


async def tail_hpo_trials(
    study_dir: Path,
    *,
    stop: asyncio.Event,
    after_trial: int | None = None,
) -> AsyncIterator[TrialFrame]:
    """Yield ``TrialFrame`` for every line written to ``trials.jsonl``.

    Replays existing lines from byte 0 then live-tails new ones — the WS
    endpoint can stream both phases through a single async loop. Works
    for any HPO study regardless of who's writing (webapp tune subprocess,
    webapp study orchestrator on a nested leg, or CLI invocation) because
    the source of truth is the on-disk JSONL file, not an in-process
    broker channel.

    The caller passes an already-resolved ``study_dir`` so we don't redo
    :func:`find_hpo_study_dir_by_wire_id` (which the WS handler already
    runs for the existence check). Malformed and non-object lines are
    skipped (logged at debug); the next valid line streams normally.
    ``after_trial`` filters out trial numbers ``<= after_trial``
    symmetrically with the REST endpoint.
    """
    trial_jsonl_path = study_dir / TRIALS_JSONL_NAME
    # Defer the import: hpo_service pulls in optuna + reporters transitively
    # and the WS endpoint is hit far less often than the main HTTP routes.
    from webapp.backend.app.services.hpo_service import trial_row_from_record

    label = study_dir.name
    async for line in tail_log(trial_jsonl_path, stop=stop):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("malformed trials.jsonl line for %s: %r", label, line)
            continue
        if not isinstance(parsed, dict):
            logger.debug("non-object trials.jsonl line for %s: %r", label, line)
            continue
        row = trial_row_from_record(parsed)
        if after_trial is not None and row.number <= after_trial:
            continue
        yield TrialFrame(trial=row)
