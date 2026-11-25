"""Batched delivery for the broker publishers (RabbitMQ, Kafka, Redis).

``publish_event`` hands an event to :meth:`BufferedPublisher.enqueue`, which only appends it
to a bounded queue: no task per event. One worker per publisher sends the queue in batches,
every ``batch_size`` events or ``flush_interval_ms`` after the first queued event, whichever
comes first. One queue and one worker keep the order. When the queue is full the event is
dropped, counted in ``tenxgraph.publish.dropped`` and warned about at most every few seconds.

Batching is always on. ``batch_size`` (default 100), ``flush_interval_ms`` (default 20) and
``queue_size`` (default 2048) in the publisher config tune it; ``publish_timeout`` (seconds)
bounds one batch send. ``tenxgraph.publish.duration`` times each batch send,
``tenxgraph.publish.batches`` and ``tenxgraph.publish.events`` count batches and events sent.

Sync ``invoke`` runs each call in a new event loop, so the worker belongs to a loop: when an
event arrives on another loop the worker (and its lock and signals) start again on that loop.
Events still queued are kept. ``CompiledGraph.aclose``, sync ``invoke``/``stream`` and
``close()`` call :meth:`flush` so nothing queued is lost.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from abc import abstractmethod
from collections import deque
from typing import Any

from tenxgraph.utils import metrics

from .base_publisher import BasePublisher
from .events import EventModel


logger = logging.getLogger("tenxgraph.publisher")

DEFAULT_BATCH_SIZE = 100
DEFAULT_FLUSH_INTERVAL_MS = 20
DEFAULT_QUEUE_SIZE = 2048
# At most one "queue full" warning per publisher in this many seconds.
DROP_WARNING_INTERVAL = 10.0


def _setting(config: dict[str, Any], key: str, default: float) -> float:
    """A positive number from the config, else ``default``."""
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        return default
    return value


class BufferedPublisher(BasePublisher):
    """Base of the broker publishers: a bounded queue and one worker that sends batches.

    Subclasses implement :meth:`publish_batch` (one send of a batch) besides ``publish``
    (one event, for direct callers), and call :meth:`stop_buffer` at the start of ``close``.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.batch_size = int(_setting(self.config, "batch_size", DEFAULT_BATCH_SIZE))
        self.flush_interval = (
            _setting(self.config, "flush_interval_ms", DEFAULT_FLUSH_INTERVAL_MS) / 1000.0
        )
        self.queue_size = int(_setting(self.config, "queue_size", DEFAULT_QUEUE_SIZE))
        self.dropped = 0
        self._queue: deque[EventModel] = deque()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: asyncio.Task[None] | None = None
        # Per loop (asyncio primitives belong to one loop); made by ``_bind``.
        self._has_items: asyncio.Event | None = None
        self._full: asyncio.Event | None = None
        self._lock: asyncio.Lock | None = None
        self._warned_at = 0.0

    @abstractmethod
    async def publish_batch(self, events: list[EventModel]) -> Any:
        """Send ``events`` to the broker in one go, in order. Raises on failure."""
        raise NotImplementedError

    def enqueue(self, event: EventModel) -> None:
        """Queue ``event`` for the worker. Never blocks and never raises."""
        if self._is_closed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._drop("no running event loop")
            return
        if len(self._queue) >= self.queue_size:
            self._drop("queue full")
            return
        self._queue.append(event)
        self._bind(loop)
        self._start_worker(loop)
        self._has_items.set()  # type: ignore[union-attr]
        if len(self._queue) >= self.batch_size:
            self._full.set()  # type: ignore[union-attr]

    async def flush(self) -> None:
        """Send everything queued now, after the batch in flight. Errors are logged."""
        if not self._queue and self._lock is None:
            return
        self._bind(asyncio.get_running_loop())
        async with self._lock:  # type: ignore[union-attr]
            while self._queue:
                await self._send_next()

    async def stop_buffer(self) -> None:
        """Flush the queue and stop the worker (the next event starts a new one).

        Called by the subclass ``close`` and before a sync ``invoke``/``stream`` closes its
        event loop, so no pending worker task is left behind on a closed loop.
        """
        try:
            await self.flush()
        finally:
            worker, self._worker = self._worker, None
            if worker is not None and worker.get_loop() is asyncio.get_running_loop():
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker

    def _bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Make the signals and the lock for ``loop`` when it is not the current one."""
        if loop is self._loop and self._lock is not None:
            return
        self._loop = loop
        self._has_items = asyncio.Event()
        self._full = asyncio.Event()
        self._lock = asyncio.Lock()
        if self._queue:
            self._has_items.set()
        if len(self._queue) >= self.batch_size:
            self._full.set()

    def _start_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        worker = self._worker
        if worker is not None and not worker.done() and worker.get_loop() is loop:
            return
        self._worker = loop.create_task(self._work(), name=f"{type(self).__name__}-events")

    async def _work(self) -> None:
        has_items, full, lock = self._has_items, self._full, self._lock
        assert has_items is not None and full is not None and lock is not None  # noqa: S101
        while True:
            await has_items.wait()
            if len(self._queue) < self.batch_size:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(full.wait(), self.flush_interval)
            async with lock:
                if self._queue:
                    await self._send_next()

    async def _send_next(self) -> None:
        """Take one batch off the queue and send it. Never raises (except a cancel)."""
        count = min(len(self._queue), self.batch_size)
        batch = [self._queue.popleft() for _ in range(count)]
        if not self._queue and self._has_items is not None:
            self._has_items.clear()
        if len(self._queue) < self.batch_size and self._full is not None:
            self._full.clear()
        if not batch:
            return
        timeout = _setting(self.config, "publish_timeout", 0) or None
        attrs = {"publisher": type(self).__name__}
        try:
            with metrics.timer("tenxgraph.publish.duration", attributes=attrs):
                await asyncio.wait_for(self.publish_batch(batch), timeout=timeout)
            metrics.counter("tenxgraph.publish.batches").inc(1, attrs)
            metrics.counter("tenxgraph.publish.events").inc(len(batch), attrs)
        except TimeoutError:
            logger.error(
                "Publishing a batch of %d events timed out after %ss; batch dropped",
                len(batch),
                timeout,
            )
            metrics.counter("tenxgraph.publish.dropped").inc(len(batch), attrs)
        except Exception as e:
            logger.error("Failed to publish a batch of %d events: %s", len(batch), e)
            metrics.counter("tenxgraph.publish.dropped").inc(len(batch), attrs)

    def _drop(self, reason: str) -> None:
        self.dropped += 1
        metrics.counter("tenxgraph.publish.dropped").inc(1, {"publisher": type(self).__name__})
        now = time.monotonic()
        if now - self._warned_at >= DROP_WARNING_INTERVAL:
            self._warned_at = now
            logger.warning(
                "%s dropped an event (%s, %d dropped so far); raise queue_size or check the broker",
                type(self).__name__,
                reason,
                self.dropped,
            )
