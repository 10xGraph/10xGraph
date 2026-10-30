"""
Utilities for calling sync or async functions in TAF.

This module provides helpers to detect async callables and to invoke
functions that may be synchronous or asynchronous, handling thread pool
execution and awaitables.
"""

import asyncio
import inspect
from collections.abc import Callable, Coroutine
from typing import Any


def _is_async_callable(func: Callable[..., Any]) -> bool:
    """
    Detect if a callable is a coroutine function (async def).

    Args:
        func (Callable[..., Any]): The function to check.

    Returns:
        bool: True if the function is async, False otherwise.
    """
    return inspect.iscoroutinefunction(func)


async def call_sync_or_async(func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Call a function that may be sync or async, returning its result.

    If the function is synchronous, it runs in a thread pool to avoid blocking
    the event loop. If the result is awaitable, it is awaited before returning.

    Args:
        func (Callable[..., Any]): The function to call.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        Any: The result of the function call, awaited if necessary.
    """
    if _is_async_callable(func):
        return await func(*args, **kwargs)

    # Call sync function in a thread pool
    result = await asyncio.to_thread(func, *args, **kwargs)
    # If the result is awaitable, await it
    if inspect.isawaitable(result):
        return await result
    return result


def ensure_no_running_loop(sync_name: str, async_name: str) -> None:
    """Raise a clear error when a sync API is called from inside a running event loop.

    Sync wrappers drive their own loop or block on one, so inside a running loop they
    would deadlock or fail with an obscure message.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"{sync_name}() cannot be called from a running event loop (it would block or "
        f"deadlock it). Use `await ...{async_name}(...)` instead."
    )


def run_coroutine(func: Coroutine) -> Any:
    """Run an async coroutine from a sync context.

    Raises:
        RuntimeError: If called while an event loop is running in this thread. Blocking
            on the loop from inside it deadlocks, so use the async method instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running, create one
        return asyncio.run(func)

    name = getattr(getattr(func, "cr_code", None), "co_name", None) or "the async variant"
    if inspect.iscoroutine(func):
        func.close()  # never awaited; avoid a "coroutine was never awaited" warning
    sync_name = name[1:] if name.startswith("a") and len(name) > 1 else name
    raise RuntimeError(
        f"{sync_name}() cannot be called from a running event loop (it would deadlock "
        f"the loop). Use `await ...{name}(...)` instead."
    )


def resolve_awaitable_id(value: Any) -> Any:
    """Resolve a generated id that may be an awaitable (from an async id factory).

    Works with or without a running loop: inside one, the awaitable is driven to
    completion on a short-lived loop in a worker thread, because the running loop
    cannot be blocked on from within itself.
    """
    if not inspect.isawaitable(value):
        return value

    async def _await() -> Any:
        return await value

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await())

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _await()).result()
