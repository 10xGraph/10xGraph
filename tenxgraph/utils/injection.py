"""Turn an ``Inject[...]`` default into the service it names.

Core functions take their dependencies as ``Inject[Service]`` defaults. Outside ``@inject`` such
a default is a lazy marker, not the service: it is never ``None`` and fails ``isinstance``
checks. :func:`fresh` resolves it from the active container (the running graph's own), so the
function body works with the real object, or ``None`` when the service is not bound.

Usage::

    async def sync_data(state, checkpointer: BaseCheckpointer = Inject[BaseCheckpointer]):
        checkpointer = fresh(checkpointer)
"""

from __future__ import annotations

from typing import Any, TypeVar

from injectq import Inject, InjectQ
from injectq.utils.exceptions import InjectQError


T = TypeVar("T")


def fresh(value: T) -> T:
    """Return ``value``, or resolve it now when it is an unresolved ``Inject[...]`` default.

    An explicitly passed argument is returned unchanged. An abstract service that is not bound
    in the active container (a publisher, a store) resolves to ``None``, so callers keep
    treating a missing optional service as absent; injectq builds an unbound concrete class.
    The graph binds everything the core resolves this way when it is built and compiled.
    """
    if type(value) is Inject:
        service_type: Any = value.service_type  # type: ignore[attr-defined]
        try:
            # Bound services, and concrete classes injectq can build (CallbackManager).
            return InjectQ.get_instance().try_get(service_type)  # type: ignore[no-any-return]
        except InjectQError:
            return None  # type: ignore[return-value]
    return value
