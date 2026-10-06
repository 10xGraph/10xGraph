from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any as _Any

from .base_store import BaseStore
from .embedding import BaseEmbedding, GoogleEmbedding, OpenAIEmbedding
from .long_term_memory import (
    MemoryIntegration,
    ReadMode,
    create_memory_preload_node,
    get_agent_memory_system_prompt,
    get_memory_system_prompt,
)
from .mem0_store import (
    Mem0Store,
    create_mem0_store,
    create_mem0_store_with_qdrant,
)
from .memory_config import AgentMemoryConfig, MemoryConfig, UserMemoryConfig
from .qdrant_store import (
    DEFAULT_COLLECTION,
    QdrantStore,
    create_cloud_qdrant_store,
    create_local_qdrant_store,
    create_remote_qdrant_store,
)
from .store_schema import DistanceMetric, MemoryRecord, MemorySearchResult, MemoryType


_MEMORY_TOOL_EXPORTS = {
    "make_agent_memory_tool",
    "make_user_memory_tool",
    "memory_tool",
}


def __getattr__(name: str) -> _Any:
    """Resolve prebuilt memory tools only when callers ask for them."""
    if name in _MEMORY_TOOL_EXPORTS:
        memory_tools = _import_module("agentflow.prebuilt.tools.memory")
        value = getattr(memory_tools, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseEmbedding",
    "BaseStore",
    "DEFAULT_COLLECTION",
    "DistanceMetric",
    "GoogleEmbedding",
    "Mem0Store",
    "MemoryConfig",
    "MemoryIntegration",
    "MemoryRecord",
    "MemorySearchResult",
    "MemoryType",
    "OpenAIEmbedding",
    "QdrantStore",
    "ReadMode",
    "UserMemoryConfig",
    "AgentMemoryConfig",
    "create_cloud_qdrant_store",
    "create_local_qdrant_store",
    "create_mem0_store",
    "create_mem0_store_with_qdrant",
    "create_memory_preload_node",
    "create_remote_qdrant_store",
    "get_agent_memory_system_prompt",
    "get_memory_system_prompt",
    "make_agent_memory_tool",
    "make_user_memory_tool",
    "memory_tool",
]
