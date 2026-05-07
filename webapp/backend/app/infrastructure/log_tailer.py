"""Async generator that replays + tails a log file via 100ms polling."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

DEFAULT_POLL_INTERVAL = 0.1


async def tail_log(
    log_path: Path,
    *,
    stop: asyncio.Event,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> AsyncIterator[str]:
    """Yield each line from ``log_path``; replays from byte 0, stops when ``stop`` is set."""
    pos = 0
    pending = ""
    while True:
        try:
            with log_path.open("r", encoding="utf-8") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
        except FileNotFoundError:
            chunk = ""
        if chunk:
            pending += chunk
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                yield line
        if stop.is_set():
            break
        await asyncio.sleep(poll_interval)
    if pending:
        yield pending
