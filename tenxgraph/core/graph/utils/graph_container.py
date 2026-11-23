"""Per-graph InjectQ container, so graphs in one process never share their components.

Every ``StateGraph`` keeps the values it is built and compiled with (checkpointer, store,
callback manager, publisher, background task manager, id generator, media policy, nodes) in its
own :class:`GraphContainer`. A run makes that container the current one for its own asyncio
task (:func:`activate`), so the engine and every ``Inject[...]`` resolved during the run see
this graph's values. Keys the graph does not hold, such as a ``db`` the developer bound before
building the graph, come from the parent container (the global one, or the ``container=``
passed to ``StateGraph``).
"""

from __future__ import annotations

import contextlib
import contextvars
import weakref
from typing import Any

from injectq import InjectQ
from injectq.core.context import ContainerContext


# injectq's "current container" variable, read first by InjectQ.get_instance() and @inject.
# It is set directly (not via ContainerContext.use/set_current) because those also write a
# per-thread fallback that concurrent runs on one event loop would overwrite and leave behind.
_current: contextvars.ContextVar[InjectQ | None] = ContainerContext._context_var

_graph_containers: weakref.WeakSet[GraphContainer] = weakref.WeakSet()


class GraphContainer(InjectQ):
    """A graph's own container. Keys it does not hold are resolved by the parent container."""

    def __init__(self, parent: InjectQ) -> None:
        super().__init__()
        self.parent = parent
        _graph_containers.add(self)

    def _holds(self, service_type: Any, value: Any) -> bool:
        binding = self._registry.get_binding(service_type)
        return binding is not None and binding.implementation is value

    def _from_parent(self, service_type: Any, value: Any) -> Any:
        # The parent also keeps the values graphs bound there for code that reads it outside a
        # run (the API server). One of those belongs to another graph: never serve it here.
        if value is not None and any(
            other is not self and other._holds(service_type, value)
            for other in list(_graph_containers)
        ):
            return None
        return value

    def has(self, service_type: Any) -> bool:
        return super().has(service_type) or self.parent.has(service_type)

    def __contains__(self, service_type: Any) -> bool:
        return self.has(service_type)

    def get(self, service_type: Any) -> Any:
        if super().has(service_type):
            return super().get(service_type)
        return self._from_parent(service_type, self.parent.get(service_type))

    async def aget(self, service_type: Any) -> Any:
        if super().has(service_type):
            return await super().aget(service_type)
        return self._from_parent(service_type, await self.parent.aget(service_type))

    def get_factory(self, service_type: Any) -> Any:
        if self._registry.has_factory(service_type):
            return super().get_factory(service_type)
        return self.parent.get_factory(service_type)


def activate(container: InjectQ) -> contextvars.Token | None:
    """Make ``container`` current for this task; None when it already is."""
    if _current.get() is container:
        return None
    return _current.set(container)


def deactivate(token: contextvars.Token | None) -> None:
    """Undo :func:`activate`.

    An async generator can be closed from another context than the one that activated the
    container, where reset() raises. That context's copy then simply lapses with it.
    """
    if token is None:
        return
    with contextlib.suppress(ValueError):
        _current.reset(token)
