"""Resolve ``Inject[...]`` defaults on every call.

An ``Inject[Service]`` default is a proxy object created once, when the function is defined, and
it caches the first object it resolves. A function that is not wrapped by ``@inject`` therefore
keeps using the first checkpointer, publisher or callback manager the process ever resolved, even
after the container is rebound or a graph with its own container runs. :func:`fresh` resolves the
dependency from the active container on each call instead.

Usage::

    async def sync_data(state, checkpointer: BaseCheckpointer = Inject[BaseCheckpointer]):
        checkpointer = fresh(checkpointer)
"""

from __future__ import annotations

import inspect
from typing import Any, TypeVar

from injectq import Inject, InjectQ
from injectq.utils.exceptions import InjectQError


T = TypeVar("T")


def fresh(value: T) -> T:
    """Return ``value``, or resolve it now when it is an unresolved ``Inject[...]`` default.

    An explicitly passed argument is returned unchanged. A dependency that is not bound in the
    active container resolves to ``None``, so callers keep treating a missing optional service
    as absent. The graph binds everything the core resolves this way when it is built and
    compiled.
    """
    if type(value) is Inject:
        service_type: Any = value.service_type  # type: ignore[attr-defined]
        container = InjectQ.get_instance()
        # An unbound abstract base (BasePublisher) cannot be built. Asking injectq to try
        # leaves a failed transient factory behind that breaks later lookups of that type.
        if not container.has(service_type) and inspect.isabstract(service_type):
            return None  # type: ignore[return-value]
        try:
            # Bound services, and concrete classes injectq can build (CallbackManager).
            return container.try_get(service_type)  # type: ignore[no-any-return]
        except InjectQError:
            return None  # type: ignore[return-value]
    return value
