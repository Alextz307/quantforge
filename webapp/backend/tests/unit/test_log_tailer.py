"""
Replay + tail behaviour of the async log generator.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from webapp.backend.app.infrastructure.log_tailer import tail_log

POLL_INTERVAL_S = 0.01
SETTLE_DELAY_S = 0.05


async def _drain(log_path: Path, *, stop: asyncio.Event) -> list[str]:
    out: list[str] = []
    async for line in tail_log(log_path, stop=stop, poll_interval=POLL_INTERVAL_S):
        out.append(line)
    return out


def test_replays_existing_content_when_already_stopped(tmp_path: Path) -> None:
    log_path = tmp_path / "test.log"
    log_path.write_text("first\nsecond\n", encoding="utf-8")

    async def scenario() -> list[str]:
        stop = asyncio.Event()
        stop.set()
        return await _drain(log_path, stop=stop)

    assert asyncio.run(scenario()) == ["first", "second"]


def test_yields_appended_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "test.log"
    log_path.write_text("alpha\n", encoding="utf-8")

    async def scenario() -> list[str]:
        stop = asyncio.Event()
        consumer = asyncio.create_task(_drain(log_path, stop=stop))
        await asyncio.sleep(SETTLE_DELAY_S)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("beta\ngamma\n")
        await asyncio.sleep(SETTLE_DELAY_S)
        stop.set()
        return await consumer

    assert asyncio.run(scenario()) == ["alpha", "beta", "gamma"]


def test_waits_for_file_to_appear(tmp_path: Path) -> None:
    log_path = tmp_path / "later.log"

    async def scenario() -> list[str]:
        stop = asyncio.Event()
        consumer = asyncio.create_task(_drain(log_path, stop=stop))
        await asyncio.sleep(SETTLE_DELAY_S)
        log_path.write_text("delayed\n", encoding="utf-8")
        await asyncio.sleep(SETTLE_DELAY_S)
        stop.set()
        return await consumer

    assert asyncio.run(scenario()) == ["delayed"]


def test_flushes_partial_trailing_line_on_stop(tmp_path: Path) -> None:
    log_path = tmp_path / "partial.log"
    log_path.write_text("complete\nstill-going", encoding="utf-8")

    async def scenario() -> list[str]:
        stop = asyncio.Event()
        stop.set()
        return await _drain(log_path, stop=stop)

    assert asyncio.run(scenario()) == ["complete", "still-going"]
