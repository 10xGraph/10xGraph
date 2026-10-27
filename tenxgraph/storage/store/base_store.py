"""
Simplified Async-First Base Store for TAF Framework

This module provides a clean, modern interface for memory stores with:
- Async-first design for better performance
- Core CRUD operations only
- Message-specific convenience methods
- Extensible for different backends (vector stores, managed services, etc.)
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import Any
from uuid import uuid4

from injectq import InjectQ

from tenxgraph.core.state import Message
from tenxgraph.utils import run_coroutine

from .store_schema import DistanceMetric, MemorySearchResult, MemoryType, RetrievalStrategy


logger = logging.getLogger("tenxgraph.store")

SYSTEM_USER_PREFIX = "__system__"
"""Reserved prefix for owners of shared, ownerless data (agent/app memory, RAG knowledge bases).

Stores refuse an empty ``user_id``, so data that belongs to no person is written and read under
``system_user_id(name)``. Each name is its own owner, so shared data sets stay separate. The API
refuses any authenticated identity that starts with this prefix, so no real user can act as one.
"""

_USER_ID_REQUIRED = (
    "user_id is required for memory operations; use system_user_id(name) for shared data"
)


def system_user_id(name: str) -> str:
    """Return the reserved owner id ``"__system__:<name>"`` for shared data called ``name``."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("system_user_id name must be a non-empty string")
    return f"{SYSTEM_USER_PREFIX}:{name.strip()}"


def is_system_user_id(user_id: Any) -> bool:
    """True if ``user_id`` is a string that starts with the reserved ``__system__`` prefix."""
    return isinstance(user_id, str) and user_id.startswith(SYSTEM_USER_PREFIX)


class BaseStore(ABC):
    """
    Simplified async-first base class for memory stores in TAF.

    This class provides a clean interface that supports:
    - Vector stores (Qdrant, Pinecone, Chroma, etc.)
    - Managed memory services (mem0, Zep, etc.)
    - Graph databases (Neo4j, etc.)

    Key Design Principles:
    - Async-first for better performance
    - Core CRUD operations only
    - User and agent-centric operations
    - Extensible filtering and metadata
    """

    def setup(self) -> Any:
        """
        Synchronous setup method for checkpointer.

        Returns:
            Any: Implementation-defined setup result.
        """
        return run_coroutine(self.asetup())

    async def asetup(self) -> Any:
        """
        Asynchronous setup method for checkpointer.

        Returns:
            Any: Implementation-defined setup result.
        """
        raise NotImplementedError

    def _scope_user_id(self, config: dict[str, Any], user_id: str | int | None) -> str | int | None:
        """Return the user_id to scope memory operations by, honoring the authz policy.

        Mirrors the checkpointer's isolation rule so one policy governs both:

        - ``config["authz"].scope == "none"`` -> ``None`` (do not scope; see everything).
        - otherwise (``"owner"`` or no policy) -> the given ``user_id`` (owner-only).

        The safe default (no policy) keeps scoping by ``user_id`` -- memory is personal, so
        it stays isolated unless a developer explicitly runs ``allow_all``/``scope=none``.
        """
        from tenxgraph.core.authz import SCOPE_NONE, isolation_scope

        if isolation_scope(config) == SCOPE_NONE:
            return None
        return user_id

    @staticmethod
    def _valid_user_id(user_id: Any) -> bool:
        """True for a non-empty string (after stripping) or an int."""
        if isinstance(user_id, bool):
            return False
        if isinstance(user_id, int):
            return True
        return isinstance(user_id, str) and bool(user_id.strip())

    def _require_user_id(
        self, config: dict[str, Any], user_id: str | int | None = None
    ) -> str | int | None:
        """Return the user_id to filter reads/updates/deletes by, or raise if it is missing.

        ``user_id`` defaults to ``config["user_id"]``. A missing, ``None``, empty or blank id
        raises ``ValueError`` (a store must never fall back to "every user"). The only
        exception is the explicit authz policy ``scope="none"`` (admin / ``allow_all``), where
        ``None`` is returned and callers skip the user filter.
        """
        if user_id is None:
            user_id = config.get("user_id")
        if not self._valid_user_id(user_id):
            from tenxgraph.core.authz import SCOPE_NONE, isolation_scope

            if isolation_scope(config) == SCOPE_NONE:
                return None  # scope="none": deliberately no user filter
            raise ValueError(_USER_ID_REQUIRED)
        return self._scope_user_id(config, user_id)

    def _require_owner_id(
        self, config: dict[str, Any], user_id: str | int | None = None
    ) -> str | int:
        """Return the concrete user_id a record is written under, or raise if it is missing.

        Used by writes and by ``aforget_memory``. Unlike ``_require_user_id`` this ignores
        ``scope="none"``: a stored record always has an owner (use ``system_user_id(name)`` for
        shared data), and a bulk forget always names whose memories to delete.
        """
        if user_id is None:
            user_id = config.get("user_id")
        if not self._valid_user_id(user_id):
            raise ValueError(_USER_ID_REQUIRED)
        return user_id  # type: ignore[return-value]

    async def generate_framework_id(self) -> str:
        """Generate a unique ID using the framework's DI-registered ID generator.

        Uses the ``generated_id`` factory bound in the InjectQ container
        (registered by ``StateGraph.compile()`` via ``BaseIDGenerator``). Falls
        back to a ``uuid4`` value when the DI context is not
        available (e.g. in unit tests that don't compile a graph).
        """
        generated_id = InjectQ.get_instance().try_get("generated_id", None)
        if not generated_id:
            return str(uuid4())
        if isinstance(generated_id, Awaitable):
            generated_id = await generated_id
        return str(generated_id)

    # --- Core Memory Operations ---

    @abstractmethod
    async def astore(
        self,
        config: dict[str, Any],
        content: str | Message,
        memory_type: MemoryType = MemoryType.EPISODIC,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """
        Add a new memory.

        Args:
            content: The memory content
            user_id: User identifier
            agent_id: Agent identifier
            memory_type: Type of memory (episodic, semantic, etc.)
            category: Memory category for organization
            metadata: Additional metadata
            **kwargs: Store-specific parameters

        Returns:
            Memory ID
        """
        raise NotImplementedError

    # --- Sync wrappers ---
    def store(
        self,
        config: dict[str, Any],
        content: str | Message,
        memory_type: MemoryType = MemoryType.EPISODIC,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Synchronous wrapper for `astore` that runs the async implementation."""
        return run_coroutine(
            self.astore(
                config,
                content,
                memory_type=memory_type,
                category=category,
                metadata=metadata,
                **kwargs,
            )
        )

    @abstractmethod
    async def asearch(
        self,
        config: dict[str, Any],
        query: str,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        max_tokens: int = 4000,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """
        Search memories by content similarity.

        Args:
            query: Search query
            user_id: Filter by user
            agent_id: Filter by agent
            memory_type: Filter by memory type
            category: Filter by category
            limit: Maximum results
            score_threshold: Minimum similarity score
            filters: Additional filters
            **kwargs: Store-specific parameters

        Returns:
            List of matching memories
        """
        raise NotImplementedError

    def search(
        self,
        config: dict[str, Any],
        query: str,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        max_tokens: int = 4000,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """Synchronous wrapper for `asearch` that runs the async implementation."""
        return run_coroutine(
            self.asearch(
                config,
                query,
                memory_type=memory_type,
                category=category,
                limit=limit,
                score_threshold=score_threshold,
                filters=filters,
                retrieval_strategy=retrieval_strategy,
                distance_metric=distance_metric,
                max_tokens=max_tokens,
                **kwargs,
            )
        )

    @abstractmethod
    async def aget(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs,
    ) -> MemorySearchResult | None:
        """Get a specific memory by ID."""
        raise NotImplementedError

    @abstractmethod
    async def aget_all(
        self,
        config: dict[str, Any],
        limit: int = 100,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """Get a specific memory by user_id."""
        raise NotImplementedError

    def get(self, config: dict[str, Any], memory_id: str, **kwargs) -> MemorySearchResult | None:
        """Synchronous wrapper for `aget` that runs the async implementation."""
        return run_coroutine(self.aget(config, memory_id, **kwargs))

    def get_all(
        self,
        config: dict[str, Any],
        limit: int = 100,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """Synchronous wrapper for `aget` that runs the async implementation."""
        return run_coroutine(self.aget_all(config, limit=limit, **kwargs))

    @abstractmethod
    async def aupdate(
        self,
        config: dict[str, Any],
        memory_id: str,
        content: str | Message,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> Any:
        """
        Update an existing memory.

        Args:
            memory_id: ID of memory to update
            content: New content (optional)
            metadata: New/additional metadata (optional)
            **kwargs: Store-specific parameters
        """
        raise NotImplementedError

    def update(
        self,
        config: dict[str, Any],
        memory_id: str,
        content: str | Message,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> Any:
        """Synchronous wrapper for `aupdate` that runs the async implementation."""
        return run_coroutine(self.aupdate(config, memory_id, content, metadata=metadata, **kwargs))

    @abstractmethod
    async def adelete(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs,
    ) -> Any:
        """Delete a memory by ID."""
        raise NotImplementedError

    def delete(self, config: dict[str, Any], memory_id: str, **kwargs) -> None:
        """Synchronous wrapper for `adelete` that runs the async implementation."""
        return run_coroutine(self.adelete(config, memory_id, **kwargs))

    @abstractmethod
    async def aforget_memory(
        self,
        config: dict[str, Any],
        **kwargs,
    ) -> Any:
        """Delete a memory by for a user or agent."""
        raise NotImplementedError

    def forget_memory(
        self,
        config: dict[str, Any],
        **kwargs,
    ) -> Any:
        """Delete a memory by for a user or agent."""
        return run_coroutine(self.aforget_memory(config, **kwargs))

    # --- Cleanup and Resource Management ---

    async def arelease(self) -> None:
        """Clean up any resources used by the store (override in subclasses if needed)."""
        raise NotImplementedError

    def release(self) -> None:
        """Clean up any resources used by the store (override in subclasses if needed)."""
        return run_coroutine(self.arelease())
