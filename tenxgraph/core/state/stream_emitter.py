"""StreamEmitter for injecting live progress chunks into graph stream output.

This module provides the StreamEmitter class, which can be optionally injected
into tool functions during streaming execution. It allows tools to emit progress,
error, and status updates that appear directly in the ``app.stream(...)`` /
``app.astream(...)`` output consumed by the frontend.

StreamEmitter has no effect during ``invoke(...)`` / ``ainvoke(...)``; in that
path, tools receive ``emit=None`` (or the parameter is absent).

Thread safety: emit methods use ``loop.call_soon_threadsafe`` so they work
correctly from sync tools running inside ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio

from .stream_chunks import StreamChunk, StreamEvent


class StreamEmitter:
    """Inject into tool functions during streaming to emit live stream chunks.

    Create one emitter per tool call. Bind it to an ``asyncio.Queue`` owned by
    the streaming handler. The handler drains the queue and yields chunks to the
    caller of ``app.stream(...)`` / ``app.astream(...)``.

    Attributes:
        tool_name: Name of the tool being executed.
        tool_call_id: Unique identifier for this tool invocation.
        node_name: Name of the graph node executing the tool.
        thread_id: Active thread/session identifier (from config).
        run_id: Active run identifier (from config).

    Example::

        def get_weather(
            location: str,
            tool_call_id: str | None = None,
            emit: StreamEmitter | None = None,
        ) -> str:
            if emit:
                emit.progress("Fetching weather data...", data={"location": location})
            result = _fetch(location)
            return result
    """

    def __init__(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        node_name: str,
        thread_id: str | None,
        run_id: str | None,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._tool_name = tool_name
        self._tool_call_id = tool_call_id
        self._node_name = node_name
        self._thread_id = thread_id
        self._run_id = run_id
        self._queue = queue
        self._loop = loop

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, event: StreamEvent, data: dict) -> None:
        """Build a StreamChunk and schedule it on the event loop thread-safely."""
        chunk = StreamChunk(
            event=event,
            data=data,
            thread_id=self._thread_id,
            run_id=self._run_id,
        )
        # call_soon_threadsafe is safe from both thread-pool threads (sync tools
        # running in asyncio.to_thread) and from the event loop itself.
        self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)

    def _base_data(self, extra: dict | None) -> dict:
        base = {
            "tool_name": self._tool_name,
            "tool_call_id": self._tool_call_id,
            "node": self._node_name,
        }
        if extra:
            base.update(extra)
        return base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def progress(self, message: str, data: dict | None = None) -> None:
        """Emit a progress update for the running tool.

        This chunk uses ``StreamEvent.MESSAGE`` so it is visible at all
        ``ResponseGranularity`` levels including the default ``LOW`` level.

        Args:
            message: Human-readable description of the current step.
            data: Optional extra key-value pairs included in the chunk data
                  (e.g. ``{"attempt": 1, "max_attempts": 3}``).
        """
        payload = self._base_data(data)
        payload["status"] = "tool_progress"
        payload["message"] = message
        self._emit(StreamEvent.MESSAGE, payload)

    def error(self, message: str, data: dict | None = None) -> None:
        """Emit an error update for the running tool.

        The tool result is still returned normally after this call; this chunk
        is informational and does not interrupt execution.

        Args:
            message: Human-readable description of the error.
            data: Optional extra key-value pairs included in the chunk data.
        """
        payload = self._base_data(data)
        payload["status"] = "tool_failed"
        payload["message"] = message
        self._emit(StreamEvent.ERROR, payload)

    def message(self, message: str, data: dict | None = None) -> None:
        """Emit a plain message update from the running tool.

        Args:
            message: Human-readable message text.
            data: Optional extra key-value pairs included in the chunk data.
        """
        payload = self._base_data(data)
        payload["status"] = "tool_message"
        payload["message"] = message
        self._emit(StreamEvent.MESSAGE, payload)

    def update(self, data: dict) -> None:
        """Emit a generic data update from the running tool.

        Args:
            data: Arbitrary key-value pairs to include in the stream chunk.
        """
        payload = self._base_data(data)
        self._emit(StreamEvent.UPDATES, payload)
