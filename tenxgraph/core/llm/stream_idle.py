"""Opt-in idle timeout for LLM streams.

The request ``timeout`` bounds a whole request; a stream that connects and then stalls
between chunks would wait that full time. ``stream_idle_timeout`` bounds the wait for the
*next* chunk. It has no default: nothing changes until it is set, per Agent with
``stream_idle_timeout=`` in ``llm_kwargs``, globally with
:func:`set_default_stream_idle_timeout`, or with the ``TENXGRAPH_LLM_STREAM_IDLE_TIMEOUT``
environment variable (seconds).
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from tenxgraph.core.exceptions.timeout_error import LLMStreamIdleTimeoutError


logger = logging.getLogger("tenxgraph.llm")

_ENV_NAME = "TENXGRAPH_LLM_STREAM_IDLE_TIMEOUT"
_default_override: dict[str, float | None] = {"value": None}

# Set per LLM call by the Agent; read by the provider stream converters. A context
# variable, so concurrent runs on one Agent never see each other's value.
stream_idle_timeout_var: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "tenxgraph_llm_stream_idle_timeout", default=None
)


def _env_value() -> float | None:
    raw = os.getenv(_ENV_NAME)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; expected a number of seconds. Ignoring.", _ENV_NAME, raw)
        return None
    if value <= 0:
        logger.warning("%s=%s must be positive. Ignoring.", _ENV_NAME, value)
        return None
    return value


def get_default_stream_idle_timeout() -> float | None:
    """Global stream idle timeout in seconds, or ``None`` (the default: no limit)."""
    override = _default_override["value"]
    return override if override is not None else _env_value()


def set_default_stream_idle_timeout(seconds: float | None) -> None:
    """Set the global stream idle timeout; ``None`` clears it.

    Raises:
        ValueError: If ``seconds`` is not positive.
    """
    if seconds is not None and seconds <= 0:
        raise ValueError("stream idle timeout must be a positive number of seconds.")
    _default_override["value"] = seconds


def resolve_stream_idle_timeout(llm_kwargs: dict[str, Any] | None) -> float | None:
    """Per-agent ``stream_idle_timeout`` if set, else the global setting, else ``None``."""
    value = (llm_kwargs or {}).get("stream_idle_timeout")
    if value is None:
        return get_default_stream_idle_timeout()
    value = float(value)
    if value <= 0:
        raise ValueError("stream_idle_timeout must be a positive number of seconds.")
    return value


async def _close(stream: Any) -> None:
    for name in ("aclose", "close"):
        fn = getattr(stream, name, None)
        if fn is None:
            continue
        try:
            result = fn()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("closing stalled LLM stream failed", exc_info=True)
        return


async def iter_with_idle_timeout(stream: Any) -> AsyncGenerator[Any]:
    """Iterate ``stream``, enforcing the current idle timeout between chunks.

    With no timeout set this is a plain ``async for``. On a stall the stream is closed and
    :class:`LLMStreamIdleTimeoutError` is raised. A cancel during the wait propagates.
    """
    timeout = stream_idle_timeout_var.get()
    if timeout is None:
        async for item in stream:
            yield item
        return
    iterator = stream.__aiter__()
    while True:
        task = asyncio.ensure_future(iterator.__anext__())
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
        except BaseException:  # cancelled while waiting: stop the pending read, re-raise
            task.cancel()
            raise
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await _close(stream)
            raise LLMStreamIdleTimeoutError(
                f"LLM stream sent no data for {timeout:g}s; closed it",
                context={"stream_idle_timeout": timeout},
            )
        try:
            item = task.result()
        except StopAsyncIteration:
            return
        yield item


class PeekedStream:
    """An async stream whose first chunk was already read (and is replayed first)."""

    def __init__(self, stream: Any, first: Any = None, empty: bool = False) -> None:
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._pending = [] if empty else [first]
        self._empty = empty

    def __aiter__(self) -> PeekedStream:
        return self

    async def __anext__(self) -> Any:
        if self._pending:
            return self._pending.pop(0)
        if self._empty:
            raise StopAsyncIteration
        return await self._iterator.__anext__()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _PeekedManager:
    """Stream context manager (Anthropic ``messages.stream()``) entered ahead of time."""

    def __init__(self, manager: Any, stream: PeekedStream) -> None:
        self._manager = manager
        self._stream = stream

    async def __aenter__(self) -> PeekedStream:
        return self._stream

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._manager.__aexit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


async def peek_first_chunk(response: Any, timeout: float | None) -> Any:
    """Wait up to ``timeout`` for the first chunk of an opened stream.

    A stall before any chunk has reached the caller is safe to retry, so this runs inside
    the Agent's retry loop: it closes the stream and raises
    :class:`LLMStreamIdleTimeoutError`, which the loop treats like a connect timeout. Once
    a chunk is out, a later stall is raised by :func:`iter_with_idle_timeout` outside the
    loop and is never retried (a retry would repeat output already streamed). Anything that
    is not an async stream is returned unchanged.
    """
    if timeout is None:
        return response
    manager = None
    stream = response
    if hasattr(response, "__aenter__") and not hasattr(response, "__aiter__"):
        manager = response
        stream = await manager.__aenter__()
    if not hasattr(stream, "__aiter__"):
        return response
    token = stream_idle_timeout_var.set(timeout)
    try:
        probe = stream.__aiter__()
        task = asyncio.ensure_future(probe.__anext__())
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
        except BaseException:
            task.cancel()
            if manager is not None:
                await manager.__aexit__(None, None, None)
            raise
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if manager is not None:
                await manager.__aexit__(None, None, None)
            else:
                await _close(stream)
            raise LLMStreamIdleTimeoutError(
                f"LLM stream sent no data for {timeout:g}s before the first chunk; closed it",
                context={"stream_idle_timeout": timeout, "before_first_chunk": True},
            )
        try:
            peeked = PeekedStream(_Resume(stream, probe), task.result())
        except StopAsyncIteration:
            peeked = PeekedStream(_Resume(stream, probe), empty=True)
        except BaseException:
            if manager is not None:
                await manager.__aexit__(None, None, None)
            raise
    finally:
        stream_idle_timeout_var.reset(token)
    return _PeekedManager(manager, peeked) if manager is not None else peeked


class _Resume:
    """Continue an iterator that was already started by the peek."""

    def __init__(self, stream: Any, iterator: Any) -> None:
        self._stream = stream
        self._iterator = iterator

    def __aiter__(self) -> Any:
        return self._iterator

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)
