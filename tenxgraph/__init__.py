"""10xGraph: a graph-based orchestration framework for multi-agent LLM systems.

This module is intentionally light. Importing ``tenxgraph`` does NOT eagerly pull
in submodules -- import the subpackage you need:

    from tenxgraph.core.graph import StateGraph, Agent
    from tenxgraph.core.state import AgentState, Message
    from tenxgraph.storage.checkpointer import PgCheckpointer

``__version__`` is exposed eagerly. ``StateGraph``, ``Agent``, ``ToolNode``, ``AgentState``,
``Message``, ``START`` and ``END`` are re-exported lazily (PEP 562) so
``from tenxgraph import StateGraph`` works without importing the engine up front. It is resolved from the installed
distribution metadata rather than hardcoded, so there is a single source of truth
(``pyproject.toml``) and the reported version cannot drift from what is actually
installed -- which is exactly how the previous 0.8.0-vs-0.7.5.1 mismatch arose.
"""

import importlib as _importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version


try:
    __version__ = _dist_version("10xgraph")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"


_LAZY_EXPORTS = {
    "StateGraph": "tenxgraph.core.graph",
    "Agent": "tenxgraph.core.graph",
    "ToolNode": "tenxgraph.core.graph",
    "AgentState": "tenxgraph.core.state",
    "Message": "tenxgraph.core.state",
    "START": "tenxgraph.utils.constants",
    "END": "tenxgraph.utils.constants",
}


def __getattr__(name: str):
    """Resolve the convenience re-exports on first access (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "__version__",
    "StateGraph",
    "Agent",
    "ToolNode",
    "AgentState",
    "Message",
    "START",
    "END",
]
