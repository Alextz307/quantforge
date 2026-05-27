"""Async generator tailing ``study_state.json`` mtime; yields ``StudyDetail``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from src.orchestration.study import STUDY_STATE_FILENAME
from webapp.backend.app.schemas.studies import StudyDetail
from webapp.backend.app.services.study_service import build_study_detail

DEFAULT_POLL_INTERVAL = 1.0

logger = logging.getLogger(__name__)


async def tail_study_state(
    study_dir: Path,
    *,
    stop: asyncio.Event,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> AsyncIterator[StudyDetail]:
    """Yield ``StudyDetail`` on every ``study_state.json`` mtime change.

    Emits the current snapshot immediately on the first iteration, then
    on every detected mtime bump. The caller is responsible for resolving
    ``study_dir`` once before constructing the generator — building the
    detail per tick uses the already-resolved path, avoiding the
    recursive glob inside :func:`find_study_dir` on every frame. Missing
    file ticks are tolerated; the loop keeps polling until ``stop`` fires.
    A read that races with an in-flight write — partial JSON — is
    silently skipped; the next tick retries.
    """

    state_path = study_dir / STUDY_STATE_FILENAME
    last_mtime: float | None = None
    while not stop.is_set():
        try:
            mtime = state_path.stat().st_mtime
        except FileNotFoundError:
            mtime = None
        if mtime is not None and mtime != last_mtime:
            try:
                detail = build_study_detail(study_dir)
            except Exception:  # noqa: BLE001 — partial-write JSON parse race
                logger.debug(
                    "study_state.json read raced for %s; retrying next tick", study_dir.name
                )
            else:
                last_mtime = mtime
                yield detail
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_interval)
        except TimeoutError:
            continue
