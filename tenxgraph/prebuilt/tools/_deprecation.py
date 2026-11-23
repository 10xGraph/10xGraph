"""Deprecation helpers for the prebuilt tools that will be removed in 2.0."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any


def _message(name: str) -> str:
    return (
        f"{name} is deprecated and will be removed in 10xgraph 2.0. "
        "Write your own tool or use an MCP server."
    )


def warn_deprecated(name: str, stacklevel: int = 3) -> None:
    """Warn that ``name`` is deprecated. The default stacklevel points at the caller of the API."""
    warnings.warn(_message(name), DeprecationWarning, stacklevel=stacklevel)


def deprecated_factory[F: Callable[..., Any]](func: F) -> F:
    """Warn each time a ``create_*`` factory is called."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warn_deprecated(func.__name__)
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def deprecated_tool[F: Callable[..., Any]](func: F) -> F:
    """Warn each time the tool runs. Apply below ``@tool`` so the metadata lands on the wrapper.

    ``functools.wraps`` keeps the name, docstring, annotations and ``__wrapped__``, so the
    schema ToolNode builds from the signature is unchanged. Async tools stay async.
    """
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(func.__name__)
            return await func(*args, **kwargs)

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warn_deprecated(func.__name__)
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
