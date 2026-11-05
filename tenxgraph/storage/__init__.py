"""Storage components for Agentflow.

This package provides all persistence and media-handling infrastructure:

- ``tenxgraph.storage.checkpointer`` — agent state persistence (in-memory, Postgres, SQLite)
- ``tenxgraph.storage.store``        — vector/long-term memory stores (Qdrant, ...)
- ``tenxgraph.storage.media``        — multimodal media processing, storage, and resolution
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any as _Any

# Import media first to avoid circular dependency:
# storage → checkpointer → core.state → core.graph → utils → storage.checkpointer
from . import checkpointer, media, store

# --- Checkpointer ---
from .checkpointer import (
    BaseCheckpointer,
    InMemoryCheckpointer,
    PgCheckpointer,
    SqliteCheckpointer,
)

# --- Media ---
from .media import (
    GOOGLE_INLINE_THRESHOLD,
    BaseMediaStore,
    CloudMediaStore,
    DocumentHandling,
    ImageHandling,
    InMemoryMediaStore,
    LocalFileMediaStore,
    MediaOffloadPolicy,
    MediaProcessor,
    MediaRefResolver,
    MultimodalConfig,
    ProviderMediaCache,
    create_openai_file_attachment,
    create_openai_file_search_tool,
    enforce_file_size,
    ensure_media_offloaded,
    sanitize_filename,
    should_use_google_file_api,
    validate_magic_bytes,
)

# --- Store (vector / long-term memory) ---
from .store import (
    DEFAULT_COLLECTION,
    SYSTEM_USER_PREFIX,
    AgentMemoryConfig,
    BaseEmbedding,
    BaseStore,
    DistanceMetric,
    GoogleEmbedding,
    MemoryConfig,
    MemoryRecord,
    MemorySearchResult,
    MemoryType,
    OpenAIEmbedding,
    PgVectorStore,
    QdrantStore,
    ReadMode,
    UserMemoryConfig,
    create_cloud_qdrant_store,
    create_local_qdrant_store,
    create_remote_qdrant_store,
    get_agent_memory_system_prompt,
    is_system_user_id,
    system_user_id,
)


_MEMORY_TOOL_EXPORTS = {
    "make_agent_memory_tool",
    "make_user_memory_tool",
}


def __getattr__(name: str) -> _Any:
    """Keep prebuilt memory tools lazy at the storage package boundary."""
    if name in _MEMORY_TOOL_EXPORTS:
        memory_tools = _import_module("tenxgraph.prebuilt.tools.memory")
        value = getattr(memory_tools, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Store
    "DEFAULT_COLLECTION",
    # Media
    "GOOGLE_INLINE_THRESHOLD",
    "AgentMemoryConfig",
    # Checkpointer
    "BaseCheckpointer",
    "BaseEmbedding",
    "BaseMediaStore",
    "BaseStore",
    "CloudMediaStore",
    "DistanceMetric",
    "DocumentHandling",
    "GoogleEmbedding",
    "ImageHandling",
    "InMemoryCheckpointer",
    "InMemoryMediaStore",
    "LocalFileMediaStore",
    "MediaOffloadPolicy",
    "MediaProcessor",
    "MediaRefResolver",
    "MemoryConfig",
    "MemoryRecord",
    "MemorySearchResult",
    "MemoryType",
    "MultimodalConfig",
    "OpenAIEmbedding",
    "PgCheckpointer",
    "PgVectorStore",
    "ProviderMediaCache",
    "SYSTEM_USER_PREFIX",
    "is_system_user_id",
    "system_user_id",
    "SqliteCheckpointer",
    "QdrantStore",
    "ReadMode",
    "UserMemoryConfig",
    # Submodules
    "checkpointer",
    "create_cloud_qdrant_store",
    "create_local_qdrant_store",
    "create_openai_file_attachment",
    "create_openai_file_search_tool",
    "create_remote_qdrant_store",
    "enforce_file_size",
    "ensure_media_offloaded",
    "get_agent_memory_system_prompt",
    "make_agent_memory_tool",
    "make_user_memory_tool",
    "media",
    "sanitize_filename",
    "should_use_google_file_api",
    "store",
    "validate_magic_bytes",
]
