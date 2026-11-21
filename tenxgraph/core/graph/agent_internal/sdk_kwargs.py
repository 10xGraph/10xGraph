"""Forward only the keyword arguments an SDK call defines.

What is accepted comes from the SDK's own signature, never from a list kept here, so a new
SDK parameter works without a 10xGraph change.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any


logger = logging.getLogger("tenxgraph.agent")

_warned: set[tuple[str, str]] = set()


@functools.lru_cache(maxsize=256)
def _signature_params(fn: Callable[..., Any]) -> frozenset[str] | None:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return frozenset(name for name in parameters if name != "self")


def accepted_params(fn: Callable[..., Any]) -> frozenset[str] | None:
    """Parameter names of ``fn``, or None when it takes ``**kwargs`` or cannot be inspected."""
    # A bound method is a new object on every access; its function is the stable cache key.
    return _signature_params(getattr(fn, "__func__", fn))


def filter_kwargs(
    kwargs: dict[str, Any], allowed: frozenset[str] | set[str] | None, provider: str
) -> dict[str, Any]:
    """Return ``kwargs`` without the keys ``allowed`` does not contain (None keeps all).

    Each dropped key is logged once per provider.
    """
    if allowed is None:
        return dict(kwargs)
    kept = {key: value for key, value in kwargs.items() if key in allowed}
    for key in kwargs.keys() - kept.keys():
        if (provider, key) not in _warned:
            _warned.add((provider, key))
            logger.warning(
                "'%s' is not a parameter of the %s SDK call and was not sent.", key, provider
            )
    return kept
