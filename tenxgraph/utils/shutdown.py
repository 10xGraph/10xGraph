"""
Graceful shutdown utilities for agent graph applications.

This module provides utilities for handling graceful shutdown of asyncio-based
applications, including signal handling, delayed keyboard interrupt protection,
and cleanup coordination.

Based on best practices from:
- https://github.com/wbenny/python-graceful-shutdown
- https://discuss.python.org/t/asyncio-best-practices/12576
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import threading
from collections.abc import Callable
from typing import Any


logger = logging.getLogger("tenxgraph.utils")

# Windows error code for invalid handle
WINDOWS_INVALID_HANDLE_ERROR = 6

# Map signal numbers to their names for logging
SIGNAL_NAMES: dict[int, str] = {
    signal.SIGINT: "SIGINT",
    signal.SIGTERM: "SIGTERM",
}


class DelayedKeyboardInterrupt:
    """
    Context manager that delays KeyboardInterrupt (SIGINT/SIGTERM) handling.

    This is useful for protecting critical sections of code (initialization,
    cleanup) from being interrupted. The signal is caught and delayed until
    the context manager exits, then the original handler is called.

    Example:
        ```python
        with DelayedKeyboardInterrupt():
            # Critical initialization code
            initialize_resources()
        # If SIGINT/SIGTERM was received, it will be handled here
        ```

    Note:
        Use sparingly and only for truly critical sections. Extended use
        can make your application unresponsive to shutdown requests.
    """

    def __init__(self):
        """Initialize the delayed interrupt handler."""
        self._sig: int | None = None
        self._frame: Any = None
        self._old_signal_handlers: dict[int, Any] = {}

    def __enter__(self):
        """Set up signal handlers to delay interrupts."""
        self._old_signal_handlers = {sig: signal.signal(sig, self._handler) for sig in SIGNAL_NAMES}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original signal handlers and raise delayed interrupt if any."""
        # Restore original handlers
        for sig, handler in self._old_signal_handlers.items():
            signal.signal(sig, handler)

        # If a signal was received, call the original handler
        if self._sig is not None:
            signal_name = SIGNAL_NAMES.get(self._sig, str(self._sig))
            logger.info("Processing delayed %s signal", signal_name)
            self._old_signal_handlers[self._sig](self._sig, self._frame)

        return False  # Don't suppress exceptions

    def _handler(self, sig: int, frame: Any):
        """Store the signal and frame for later processing."""
        self._sig = sig
        self._frame = frame
        signal_name = SIGNAL_NAMES.get(sig, str(sig))
        logger.warning(
            "%s received; delaying interrupt until critical section completes",
            signal_name,
        )


@contextlib.contextmanager
def delayed_keyboard_interrupt():
    """
    Context manager that delays KeyboardInterrupt (SIGINT/SIGTERM) handling.

    This is a functional wrapper around DelayedKeyboardInterrupt class.

    Example:
        ```python
        with delayed_keyboard_interrupt():
            # Critical initialization code
            initialize_resources()
        ```
    """
    dki = DelayedKeyboardInterrupt()
    with dki:
        yield dki


class GracefulShutdownManager:
    """
    Manager for coordinating graceful shutdown of asyncio applications.

    Handles signal registration, shutdown coordination, and cleanup orchestration.
    Protects initialization and finalization phases from interruption while
    allowing clean shutdown during normal execution.

    Example:
        ```python
        async def main():
            shutdown_manager = GracefulShutdownManager()

            try:
                # Protected initialization
                with shutdown_manager.protect_section():
                    await initialize_application()

                # Normal execution (can be interrupted)
                await run_application()

            except KeyboardInterrupt:
                logger.info("Shutdown requested")
            finally:
                # Protected cleanup
                with shutdown_manager.protect_section():
                    await cleanup_application()


        asyncio.run(main())
        ```

    Attributes:
        shutdown_requested: Flag indicating if shutdown has been requested.
        shutdown_timeout: Default timeout for cleanup operations in seconds.
    """

    def __init__(self, shutdown_timeout: float = 30.0):
        """
        Initialize the shutdown manager.

        Args:
            shutdown_timeout: Default timeout for cleanup operations in seconds.
        """
        self.shutdown_requested = False
        self.shutdown_timeout = shutdown_timeout
        self._original_handlers: dict[int, Any] = {}
        self._shutdown_callbacks: list[Callable] = []

    def register_signal_handlers(self, loop: asyncio.AbstractEventLoop | None = None):
        """
        Register signal handlers for graceful shutdown.

        This sets up handlers for SIGINT and SIGTERM that will trigger
        graceful shutdown when received.

        Args:
            loop: The asyncio event loop to use. If None, uses the running loop.

        Note:
            On Windows, SIGTERM handling may have limited support.
        """
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                # Store original handler
                self._original_handlers[sig] = signal.getsignal(sig)

                # Set up new handler
                loop.add_signal_handler(sig, self._signal_handler, sig)
                logger.debug("Registered signal handler for %s", SIGNAL_NAMES.get(sig, str(sig)))
            except (ValueError, NotImplementedError) as e:
                # Windows doesn't support add_signal_handler
                logger.warning(
                    "Could not register signal handler for %s: %s",
                    SIGNAL_NAMES.get(sig, str(sig)),
                    e,
                )

    def unregister_signal_handlers(self):
        """
        Restore original signal handlers.

        This should be called during cleanup to restore the system to its
        original state.
        """
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
                logger.debug("Restored original handler for %s", SIGNAL_NAMES.get(sig, str(sig)))
            except (ValueError, OSError) as e:
                logger.warning(
                    "Could not restore signal handler for %s: %s",
                    SIGNAL_NAMES.get(sig, str(sig)),
                    e,
                )

    def _signal_handler(self, sig: int):
        """
        Handle received signals.

        Args:
            sig: The signal number received.
        """
        signal_name = SIGNAL_NAMES.get(sig, str(sig))
        logger.info("Received %s signal, initiating graceful shutdown", signal_name)
        self.shutdown_requested = True

        # Call shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.exception("Error in shutdown callback: %s", e)

    def add_shutdown_callback(self, callback: Callable):
        """
        Add a callback to be called when shutdown is requested.

        Args:
            callback: Callable to invoke on shutdown. Should not block.
        """
        self._shutdown_callbacks.append(callback)

    def protect_section(self) -> DelayedKeyboardInterrupt:
        """
        Get a context manager for protecting critical sections.

        Returns:
            DelayedKeyboardInterrupt context manager.
        """
        return DelayedKeyboardInterrupt()

    async def wait_for_shutdown(self, check_interval: float = 0.1):
        """
        Wait until shutdown is requested.

        Args:
            check_interval: How often to check for shutdown request in seconds.
        """
        while not self.shutdown_requested:
            await asyncio.sleep(check_interval)


def setup_exception_handler(loop: asyncio.AbstractEventLoop):
    """
    Set up exception handler for the event loop.

    This suppresses certain benign exceptions that can occur during shutdown,
    particularly on Windows with ProactorEventLoop.

    Args:
        loop: The asyncio event loop.

    Note:
        Based on guidance from https://bugs.python.org/issue39010
    """

    def exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]):
        """Handle exceptions from the event loop."""
        exception = context.get("exception")

        # Suppress ConnectionResetError during shutdown (Windows bug)
        if isinstance(exception, ConnectionResetError):
            logger.debug(
                "Suppressing ConnectionResetError during shutdown: %s",
                context.get("message"),
            )
            return

        # Suppress OSError with invalid handle (Windows bug)
        if (
            isinstance(exception, OSError) and exception.winerror == WINDOWS_INVALID_HANDLE_ERROR  # type: ignore[attr-defined]
        ):
            logger.debug("Suppressing OSError (invalid handle) during shutdown")
            return

        # For other exceptions, use default handling
        logger.error(
            "Unhandled exception in event loop: %s",
            context.get("message", "No message"),
            exc_info=exception,
        )

    loop.set_exception_handler(exception_handler)


async def shutdown_with_timeout(
    coro_or_task: Any,
    timeout: float,
    task_name: str = "task",
) -> dict[str, Any]:
    """
    Shutdown a coroutine, task, or future with a timeout.

    Args:
        coro_or_task: The coroutine, task, or future to shutdown.
        timeout: Maximum time to wait in seconds.
        task_name: Name for logging purposes.

    Returns:
        Dictionary with status information.
    """
    start_time = asyncio.get_event_loop().time()

    # Convert coroutine to task if needed
    if asyncio.iscoroutine(coro_or_task):
        coro_or_task = asyncio.create_task(coro_or_task)

    try:
        await asyncio.wait_for(coro_or_task, timeout=timeout)
        duration = asyncio.get_event_loop().time() - start_time
        logger.info("Shutdown of %s completed successfully (%.2fs)", task_name, duration)
        return {"status": "completed", "duration": duration}
    except TimeoutError:
        duration = asyncio.get_event_loop().time() - start_time
        logger.warning("Shutdown of %s timed out after %.2fs", task_name, duration)

        # Cancel if it's a task
        if isinstance(coro_or_task, asyncio.Task):
            coro_or_task.cancel()
            # Not ``suppress(CancelledError)`` around ``await``: that would also absorb a
            # cancel aimed at the caller while it waits here.
            await asyncio.wait({coro_or_task})
            if not coro_or_task.cancelled():
                coro_or_task.exception()  # mark any error as retrieved

        return {"status": "timeout", "duration": duration}
    except Exception as e:
        duration = asyncio.get_event_loop().time() - start_time
        logger.exception("Error during shutdown of %s: %s", task_name, e)
        return {"status": "error", "error": str(e), "duration": duration}


class SignalShutdown:
    """Handle returned by :func:`install_shutdown_signals`.

    Attributes:
        close_task: The task running ``graph.aclose()``; ``None`` until a signal arrives.
    """

    def __init__(
        self,
        graph: Any,
        loop: Any,
        signals: tuple[int, ...],
        close_kwargs: dict[str, Any],
    ):
        self._graph = graph
        self._loop = loop
        self._signals = signals
        self._close_kwargs = close_kwargs
        self._installed: list[int] = []
        self.close_task: Any = None

    def _on_signal(self, sig: int) -> None:
        name = SIGNAL_NAMES.get(sig, str(sig))
        if self.close_task is not None:
            logger.info("Received %s again; the graph is already closing", name)
            return
        logger.info("Received %s, closing the graph", name)
        self.close_task = self._loop.create_task(self._graph.aclose(**self._close_kwargs))

    async def wait(self) -> dict[str, Any] | None:
        """Wait for the close that a signal started; returns its stats (``None`` if none ran)."""
        if self.close_task is None:
            return None
        return await self.close_task

    def uninstall(self) -> None:
        """Remove the handlers this helper installed."""
        for sig in self._installed:
            with contextlib.suppress(Exception):
                self._loop.remove_signal_handler(sig)
        self._installed.clear()


def install_shutdown_signals(
    graph: Any,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
    signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT),
    grace: float | None = None,
    stop_timeout: float | None = None,
    timeout: float | None = None,
) -> SignalShutdown:
    """Connect SIGTERM and SIGINT to ``graph.aclose()`` on the running loop (opt-in).

    10xGraph never installs signal handlers by itself: the host program owns the process. Call
    this once, from the main thread and inside the running event loop, if you want a signal to
    end active runs gracefully (grace, then stop, then cancel; see ``CompiledGraph.aclose``)
    and release the graph's resources::

        async def main():
            graph = build_graph().compile(checkpointer=checkpointer, shutdown_timeout=30)
            shutdown = install_shutdown_signals(graph)
            try:
                await serve(graph)
            finally:
                shutdown.uninstall()

    Notes:
        - Main thread only (an event loop can add signal handlers only there); a
          ``RuntimeError`` is raised from any other thread.
        - Not available on Windows event loops (``NotImplementedError`` is logged and the
          signal is skipped).
        - It replaces the default SIGINT behaviour (``KeyboardInterrupt``) while installed.
        - The first signal starts ``aclose()``; later signals are ignored. ``grace``,
          ``stop_timeout`` and ``timeout`` go to ``aclose()`` unchanged.
        - Servers that already handle SIGTERM (uvicorn, Gunicorn) should call
          ``graph.aclose()`` from their shutdown hook instead.

    Args:
        graph: The ``CompiledGraph`` to close.
        loop: The running loop; defaults to ``asyncio.get_running_loop()``.
        signals: Signals to handle.
        grace: Passed to ``aclose()``.
        stop_timeout: Passed to ``aclose()``.
        timeout: Passed to ``aclose()``.

    Returns:
        A :class:`SignalShutdown` with ``uninstall()`` and ``wait()``.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("install_shutdown_signals() must be called from the main thread")
    if loop is None:
        loop = asyncio.get_running_loop()
    handle = SignalShutdown(
        graph,
        loop,
        tuple(signals),
        {"grace": grace, "stop_timeout": stop_timeout, "timeout": timeout},
    )
    for sig in handle._signals:
        try:
            loop.add_signal_handler(sig, handle._on_signal, sig)
            handle._installed.append(sig)
        except (NotImplementedError, ValueError, RuntimeError) as e:
            logger.warning("Could not install a handler for %s: %s", SIGNAL_NAMES.get(sig, sig), e)
    return handle
