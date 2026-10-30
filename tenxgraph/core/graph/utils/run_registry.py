"""Active-run registry: one handle per running ``ainvoke`` / ``astream`` / ``arealtime``.

``CompiledGraph`` registers a :class:`RunHandle` when a run starts and removes it in the same
``finally`` that ends the run. The handle carries what ``aclose()`` and ``astop()`` need from
any thread or loop:

- an in-process stop flag (a ``threading.Event``), so a stop works with no checkpointer and
  when the cache (Redis) is down;
- the run's event loop and a wake-up event on it, so a waiting guard sees the stop at once;
- the node tasks the guards run (core owns them) and the run's own task, for a hard cancel.

The handle of the current run is exposed through :data:`current_run`, a context variable the
guards in ``guards.py`` read; it propagates into the tasks the run creates.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from typing import Any


logger = logging.getLogger("tenxgraph.graph")

# ``interrupt_reason`` given to a run that a graph shutdown hard-cancelled.
SHUTDOWN_CANCEL_REASON = "shutdown"


class RunHandle:
    """State of one active run. Safe to use from threads other than the run's loop."""

    def __init__(self, run_id: str, thread_id: Any, kind: str = "invoke"):
        self.run_id = run_id
        self.thread_id = thread_id
        self.kind = kind
        self.loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self.task: asyncio.Task | None = asyncio.current_task()
        self.heartbeat: asyncio.Task | None = None
        self.node_tasks: set[asyncio.Task] = set()
        self.cancel_reason: str | None = None
        self._stopped = threading.Event()
        self._wake = asyncio.Event()

    @property
    def stop_requested(self) -> bool:
        """True once a stop was requested for this run in this process."""
        return self._stopped.is_set()

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_reason is not None

    def request_stop(self) -> None:
        """Set the stop flag and wake any guard waiting on it."""
        self._stopped.set()
        self._call_in_loop(self._wake.set)

    def wake_waiter(self) -> asyncio.Task:
        """A task (on the run's loop) that finishes when a stop is requested."""
        return asyncio.ensure_future(self._wake.wait())

    def cancel(self, reason: str = SHUTDOWN_CANCEL_REASON) -> None:
        """Hard-cancel the run: the node tasks core owns, else the run's own task."""
        self.cancel_reason = reason
        self._call_in_loop(self._cancel_now)

    def _cancel_now(self) -> None:
        targets = [t for t in self.node_tasks if not t.done()]
        if not targets and self.task is not None and not self.task.done():
            targets = [self.task]
        for task in targets:
            task.cancel()

    def _call_in_loop(self, fn: Any) -> None:
        try:
            self.loop.call_soon_threadsafe(fn)
        except RuntimeError:  # the run's loop is closed: the run is over
            logger.debug("Run %s: loop closed before a request could be delivered", self.run_id)


current_run: contextvars.ContextVar[RunHandle | None] = contextvars.ContextVar(
    "tenxgraph_current_run", default=None
)


class RunRegistry:
    """The runs of one compiled graph that are active right now."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[int, RunHandle] = {}

    def register(self, handle: RunHandle) -> None:
        with self._lock:
            self._runs[id(handle)] = handle

    def unregister(self, handle: RunHandle) -> None:
        with self._lock:
            self._runs.pop(id(handle), None)

    def active(self) -> list[RunHandle]:
        with self._lock:
            return list(self._runs.values())

    def for_thread(self, thread_id: Any) -> list[RunHandle]:
        return [h for h in self.active() if h.thread_id == thread_id]
