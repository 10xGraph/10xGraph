"""Process-wide count of live graphs per shared resource.

Two compiled graphs can share one checkpointer, store or publisher. ``aclose()`` on one
must not close a resource the other still uses, so each graph registers the resources it
holds and releases one only when it is the last live graph holding it.

Graphs are tracked weakly: a graph that is garbage collected without ``aclose()`` stops
counting, so it cannot keep a resource open forever.
"""

import threading
import weakref
from typing import Any


_lock = threading.Lock()
# id(resource) -> graphs holding it. A graph holds its resource strongly, so the id
# cannot be reused while the set has a live member.
_holders: dict[int, "weakref.WeakSet[Any]"] = {}


def register(resource: object | None, holder: Any) -> None:
    """Record that ``holder`` (a compiled graph) uses ``resource``."""
    if resource is None:
        return
    with _lock:
        _holders.setdefault(id(resource), weakref.WeakSet()).add(holder)


def release(resource: object | None, holder: Any) -> bool:
    """Drop ``holder``'s claim on ``resource``.

    Returns True when no other live graph still holds it, meaning the caller should
    close it now.
    """
    if resource is None:
        return False
    with _lock:
        holders = _holders.get(id(resource))
        if holders is None:
            return True
        holders.discard(holder)
        if len(holders) == 0:
            del _holders[id(resource)]
            return True
        return False


def holder_count(resource: object | None) -> int:
    """Number of live graphs holding ``resource`` (for tests and diagnostics)."""
    if resource is None:
        return 0
    with _lock:
        return len(_holders.get(id(resource), ()))
