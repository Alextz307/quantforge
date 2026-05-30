"""
Generic in-process pub/sub keyed by ``str``; one queue per subscriber.
"""

from __future__ import annotations

import asyncio


class EventBroker[V]:
    """
    Per-key pub/sub of ``V``-typed frames.

    Queues are unbounded - a slow consumer can grow its own queue but
    never blocks the producer or other consumers. ``None`` is the
    close sentinel (see :meth:`close`).
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[V | None]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, key: str) -> asyncio.Queue[V | None]:
        async with self._lock:
            queue: asyncio.Queue[V | None] = asyncio.Queue()
            self._subscribers.setdefault(key, []).append(queue)
            return queue

    async def unsubscribe(self, key: str, queue: asyncio.Queue[V | None]) -> None:
        async with self._lock:
            queues = self._subscribers.get(key)
            if queues is not None and queue in queues:
                queues.remove(queue)
                if not queues:
                    del self._subscribers[key]

    async def publish(self, key: str, frame: V) -> None:
        async with self._lock:
            queues = list(self._subscribers.get(key, []))
        for queue in queues:
            queue.put_nowait(frame)

    async def close(self, key: str) -> None:
        async with self._lock:
            queues = self._subscribers.pop(key, [])
        for queue in queues:
            queue.put_nowait(None)


__all__ = ["EventBroker"]
