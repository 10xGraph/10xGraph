"""Execution guards: bound how long work runs, and cancel it when stopped.

Without these, two failure modes are unreachable from the execution loop:

1. **Indefinite hang.** Nothing wrapped node or tool execution in a timeout. The
   only bound was the LLM SDK's own timeout, which does not cover custom tools,
   MCP calls, or condition functions. A node blocked on a half-open socket never
   returns, so the loop never advances a step -- the recursion limit never trips
   -- and the run blocks forever.

2. **Un-cancellable run.** Stop was cooperative and polled only *between* nodes.
   If the hang is inside a node, control never returns to the loop, so the stop
   flag is never observed. `stop()` appeared to succeed while the run kept going.

`execute_with_guards` fixes both by running the work as a task it can actually
cancel: it enforces a deadline, and while waiting it periodically asks whether a
stop has been requested, cancelling the task if so.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

from .run_registry import RunHandle, current_run


logger = logging.getLogger("tenxgraph.graph")

T = TypeVar("T")

# How often to check for an out-of-band stop request while work is in flight.
# Small enough that a stop feels responsive, large enough not to hammer the
# checkpointer's cache on every node.
DEFAULT_STOP_POLL_INTERVAL = 1.0


async def _cancel_and_wait(task: asyncio.Task) -> None:
    """Cancel a task and wait for it to actually finish unwinding.

    Awaiting after cancel matters: it lets the coroutine run its `finally`
    blocks (closing sessions, releasing connections) before we move on, and it
    prevents an orphaned task from being garbage-collected mid-flight.
    """
    task.cancel()
    # asyncio.wait never raises the task's own CancelledError, and a cancel aimed at the
    # caller while it waits here still propagates (suppressing CancelledError around
    # ``await task`` would absorb that one too).
    await asyncio.wait({task})
    if not task.cancelled():
        task.result()  # surfaces an error the cleanup raised


async def execute_with_guards(  # noqa: PLR0912
    work: Awaitable[T],
    *,
    timeout: float | None = None,
    stop_check: Callable[[], Awaitable[bool]] | None = None,
    poll_interval: float = DEFAULT_STOP_POLL_INTERVAL,
    on_timeout: Callable[[], Exception] | None = None,
    on_stop: Callable[[], Exception] | None = None,
) -> T:
    """Run `work` under a deadline, cancelling it if a stop is requested.

    Args:
        work: The coroutine to run (node execution, tool call, ...).
        timeout: Seconds before the work is cancelled. None disables the deadline.
        stop_check: Async predicate polled while waiting; when it returns True the
            work is cancelled. None disables stop-cancellation.
        poll_interval: How often to run `stop_check`.
        on_timeout: Builds the exception raised when the deadline passes.
        on_stop: Builds the exception raised when a stop is observed.

    Returns:
        Whatever `work` returned.

    Raises:
        Whatever `work` raised, or the exception from `on_timeout` / `on_stop`.
    """
    task: asyncio.Task = asyncio.ensure_future(work)
    loop = asyncio.get_running_loop()
    deadline = (loop.time() + timeout) if timeout else None
    run = current_run.get()
    wake: asyncio.Task | None = None
    if run is not None:
        run.node_tasks.add(task)
        wake = run.wake_waiter()
    last_remote_check = loop.time()

    try:
        while True:
            # An in-process stop (astop, aclose) acts at once and needs no checkpointer.
            if run is not None and run.stop_requested:
                await _stop_work(task, run, on_stop, "Stop requested while work was in flight")
            remaining: float | None = None
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    await _cancel_and_wait(task)
                    raise (on_timeout() if on_timeout else TimeoutError("Execution timed out"))

            # Wake up either at the next stop poll or at the deadline, whichever
            # comes first. With neither, block until the work completes.
            wait_for = remaining
            if stop_check is not None:
                wait_for = poll_interval if remaining is None else min(poll_interval, remaining)

            waiting: set[asyncio.Future] = {task} if wake is None else {task, wake}
            done, _ = await asyncio.wait(
                waiting, timeout=wait_for, return_when=asyncio.FIRST_COMPLETED
            )

            if task in done:
                # Surfaces the work's own exception, if it raised.
                return task.result()

            if wake is not None and wake in done:
                continue  # the stop flag is set: handled at the top of the loop

            if stop_check is not None and loop.time() - last_remote_check >= poll_interval - 1e-3:
                last_remote_check = loop.time()
                if await _remote_stop(stop_check):
                    await _stop_work(task, run, on_stop, "Stop requested while work was in flight")

    except asyncio.CancelledError:
        # The caller (or the whole run) is being cancelled: don't leak the task.
        if not task.done():
            await _cancel_and_wait(task)
        raise
    finally:
        if run is not None:
            run.node_tasks.discard(task)
        if wake is not None:
            wake.cancel()


async def _remote_stop(stop_check: Callable[[], Awaitable[bool]]) -> bool:
    """Ask the checkpointer whether a stop was requested; an unreachable cache is "no"."""
    try:
        return await stop_check()
    except Exception as e:  # the in-process flag still works when the cache is down
        logger.warning("Stop check failed (%r); relying on the in-process stop flag", e)
        return False


async def _stop_work(
    task: asyncio.Task,
    run: RunHandle | None,
    on_stop: Callable[[], Exception] | None,
    message: str,
) -> None:
    """Cancel ``task`` for a stop, then raise. A hard cancel that arrived meanwhile wins."""
    logger.info("%s; cancelling it", message)
    await _cancel_and_wait(task)
    if run is not None and run.cancel_requested:
        raise asyncio.CancelledError()
    raise (on_stop() if on_stop else asyncio.CancelledError())


class _StreamFailure:
    """A producer-side exception, carried through the queue to the consumer."""

    def __init__(self, error: BaseException):
        self.error = error


_STREAM_DONE = object()


async def stream_with_guards(  # noqa: PLR0912, PLR0915
    source: AsyncIterator[T],
    *,
    timeout: float | None = None,
    stop_check: Callable[[], Awaitable[bool]] | None = None,
    poll_interval: float = DEFAULT_STOP_POLL_INTERVAL,
    on_timeout: Callable[[], Exception] | None = None,
    on_stop: Callable[[], Exception] | None = None,
) -> AsyncIterator[T]:
    """Iterate ``source`` under a deadline, cancelling it if a stop is requested.

    The stream twin of :func:`execute_with_guards`. ``source`` is driven by a producer task
    that core owns, so a node that hangs between items (or before its first) can still be
    cancelled. The consumer reads items from a queue and, while it waits, checks the deadline
    and polls ``stop_check``. Core cancels only its own producer task, never the caller's.
    Items the producer already queued are delivered before a deadline is enforced, so a slow
    consumer is not charged for its own delay.

    Raises:
        Whatever ``source`` raised, or the exception from ``on_timeout`` / ``on_stop``.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def produce() -> None:
        try:
            async for item in source:
                queue.put_nowait(item)
        except Exception as e:
            queue.put_nowait(_StreamFailure(e))
        finally:
            # A GraphInterrupt (BaseException) ends the task with its error; the consumer
            # re-raises it after the queued items.
            queue.put_nowait(_STREAM_DONE)

    task: asyncio.Task = asyncio.ensure_future(produce())
    loop = asyncio.get_running_loop()
    deadline = (loop.time() + timeout) if timeout else None
    run = current_run.get()
    wake: asyncio.Task | None = None
    if run is not None:
        run.node_tasks.add(task)
        wake = run.wake_waiter()
    last_remote_check = loop.time()
    stop_message = "Stop requested while a node was streaming"

    try:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                if run is not None and run.stop_requested:
                    await _stop_work(task, run, on_stop, stop_message)
                remaining: float | None = None
                if deadline is not None:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        await _cancel_and_wait(task)
                        raise (on_timeout() if on_timeout else TimeoutError("Execution timed out"))
                wait_for = remaining
                if stop_check is not None:
                    wait_for = poll_interval if remaining is None else min(poll_interval, remaining)
                get_task: asyncio.Task = asyncio.ensure_future(queue.get())
                waiting: set[asyncio.Future] = {get_task} if wake is None else {get_task, wake}
                try:
                    await asyncio.wait(
                        waiting, timeout=wait_for, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    if not get_task.done():
                        get_task.cancel()
                        await asyncio.wait({get_task})
                if get_task.cancelled():
                    if run is not None and run.stop_requested:
                        continue  # handled at the top of the loop
                    if (
                        stop_check is not None
                        and loop.time() - last_remote_check >= poll_interval - 1e-3
                    ):
                        last_remote_check = loop.time()
                        if await _remote_stop(stop_check):
                            await _stop_work(task, run, on_stop, stop_message)
                    continue
                item = get_task.result()

            if item is _STREAM_DONE:
                break
            if isinstance(item, _StreamFailure):
                raise item.error
            yield item

        if task.cancelled():
            # Cancelled from outside (a hard cancel): not the same as a finished stream.
            raise asyncio.CancelledError()
        exc = task.exception()
        if exc is not None:
            raise exc
    finally:
        # Also runs when the consumer is cancelled or closes the stream early: no orphan.
        if not task.done():
            await _cancel_and_wait(task)
        if run is not None:
            run.node_tasks.discard(task)
        if wake is not None:
            wake.cancel()


def resolve_timeout(config: dict[str, Any], key: str, default: float | None) -> float | None:
    """Read a timeout from the run config, falling back to a default.

    An explicit ``None`` in the config disables the timeout, which is different
    from the key being absent (use the default). A non-positive value also
    disables it, so `0` is an easy way to opt out.
    """
    if key not in config:
        return default
    value = config[key]
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r in config; falling back to %s", key, config[key], default)
        return default
    return value if value > 0 else None
