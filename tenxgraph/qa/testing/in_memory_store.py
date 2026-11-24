"""In-memory store for testing - no external dependencies.

This module provides an InMemoryStore class similar to InMemoryCheckpointer,
offering a simple in-memory implementation for testing without requiring
external databases or embedding services.
"""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from tenxgraph.core.state import Message
from tenxgraph.storage.sql_filter import isolation_filter_error
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.storage.store.store_schema import (
    DistanceMetric,
    MemorySearchResult,
    MemoryType,
    RetrievalStrategy,
    check_origins,
    is_live,
)


logger = logging.getLogger("tenxgraph.testing")

# Keys the store owns: they are never taken from caller metadata.
_SYSTEM_KEYS = frozenset(
    {
        "content",
        "user_id",
        "thread_id",
        "memory_type",
        "category",
        "timestamp",
        "memory_key",
        "origin",
        "expires_at",
    }
)


class InMemoryStore(BaseStore):
    """In-memory store for testing - no external dependencies.

    Like InMemoryCheckpointer, this provides a simple in-memory
    implementation for testing without requiring databases or embeddings.

    Features:
    - Store and retrieve memories by ID
    - Simple text-based search
    - Pre-configurable search results for testing
    - No external dependencies

    Example:
        ```python
        store = InMemoryStore()

        # Pre-configure search results for testing
        store.set_search_results([MemorySearchResult(id="1", content="Relevant memory", score=0.9)])

        graph = StateGraph()
        compiled = graph.compile(store=store)
        ```
    """

    supports_provenance = True

    def __init__(self):
        """Initialize an in-memory store."""
        self.collection = "memories"
        self.memories: dict[str, MemorySearchResult] = {}
        self._collection_of: dict[str, str] = {}
        self._search_results: list[MemorySearchResult] = []
        logger.debug("InMemoryStore initialized")

    async def asetup(self) -> Any:
        """Setup the store (no-op for in-memory)."""
        logger.debug("InMemoryStore setup called (no-op)")
        return None

    def set_search_results(self, results: list[MemorySearchResult]) -> None:
        """Pre-configure search results for testing.

        When set, asearch() will return these results instead of performing
        actual search. This is useful for testing retrieval-dependent behavior.

        Args:
            results: List of MemorySearchResult to return from searches
        """
        self._search_results = results
        logger.debug("Pre-configured %d search results", len(results))

    def _visible(
        self,
        config: dict[str, Any],
        memory_type: MemoryType | None = None,
        origins: Any = None,
    ) -> list[MemorySearchResult]:
        """Memories the caller may read, newest first (same scope rules as PgVectorStore).

        Expired memories are left out; ``origins`` (see ``check_origins``) narrows by origin.
        """
        _user, _thread, collection = self._extract_config_values(config)
        scoped = self._require_user_id(config)
        wanted = check_origins(origins)
        now = datetime.now(UTC)
        rows = [
            m
            for m in self.memories.values()
            if self._collection_of[m.id] == collection
            and (scoped is None or m.user_id == scoped)
            and (memory_type is None or m.memory_type == memory_type)
            and is_live(m.expires_at, now)
            and (
                wanted is None
                or (m.origin.value in wanted[0] if m.origin is not None else wanted[1])
            )
        ]
        # Ascending then reversed: equal timestamps keep newest-stored first.
        return sorted(rows, key=lambda m: m.timestamp or datetime.min.replace(tzinfo=UTC))[::-1]

    def _owned(
        self, config: dict[str, Any], memory_id: str, owner: str | int | None
    ) -> MemorySearchResult:
        """The memory in the config's collection owned by ``owner`` (None: any owner).

        Missing and foreign memories look the same: ``ValueError``.
        """
        _user, _thread, collection = self._extract_config_values(config)
        mem = self.memories.get(memory_id)
        if mem is None or self._collection_of[memory_id] != collection:
            raise ValueError(f"Memory {memory_id} not found")
        if owner is not None and mem.user_id != owner:
            raise ValueError(f"Memory {memory_id} not found")
        return mem

    @staticmethod
    def _metadata(
        base: dict[str, Any], category: Any, memory_type: MemoryType, memory_key: Any
    ) -> dict[str, Any]:
        """Caller metadata plus category, memory_type and memory_key."""
        meta = {k: v for k, v in base.items() if k not in _SYSTEM_KEYS}
        meta["category"] = category
        meta["memory_type"] = memory_type.value
        if memory_key is not None:
            meta["memory_key"] = memory_key
        return meta

    async def astore(
        self,
        config: dict[str, Any],
        content: str | Message,
        memory_type: MemoryType = MemoryType.EPISODIC,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Store a memory under the caller's own user_id (reach does not change the owner).

        Raises ``ValueError`` for a missing or blank user_id or collection, for text with NUL
        or a lone surrogate, and for a ``memory_id`` (keyword argument) of another user or
        collection (not found, as on the real stores; that memory is left as it is).
        """
        self._refuse_unstorable(
            content=content,
            metadata=metadata,
            category=category,
            memory_key=kwargs.get("memory_key"),
        )
        origin, expires_at = self._provenance(kwargs)
        user_id, thread_id, collection = self._extract_config_values(config)
        owner = self._require_owner_id(config, user_id)
        mem_id = str(kwargs.get("memory_id") or uuid4())
        if mem_id in self.memories:
            self._owned(config, mem_id, str(owner))
        content_str = content if isinstance(content, str) else content.text()
        meta = metadata or {}
        memory_key = kwargs.get("memory_key", meta.get("memory_key"))

        self.memories[mem_id] = MemorySearchResult(
            id=mem_id,
            content=content_str,
            score=1.0,
            memory_type=memory_type,
            metadata=self._metadata(meta, category or "general", memory_type, memory_key),
            user_id=str(owner),
            thread_id=thread_id,
            timestamp=datetime.now(UTC),
            origin=origin,
            expires_at=expires_at,
        )
        self._collection_of[mem_id] = collection

        cut_ratio = 50
        logger.debug(
            "Stored memory %s: %s...",
            mem_id,
            content_str[:cut_ratio] if len(content_str) > cut_ratio else content_str,
        )
        return mem_id

    async def asearch(
        self,
        config: dict[str, Any],
        query: str,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        retrieval_strategy: RetrievalStrategy | None = None,
        distance_metric: DistanceMetric | None = None,
        max_tokens: int = 4000,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        """Search memories - returns pre-configured results or a text match.

        If results were pre-configured via set_search_results(), returns those. Otherwise
        a case-insensitive substring match over the memories the caller may read, filtered
        by memory_type, category and ``filters`` (exact match on metadata keys; a ``user_id`` key
        raises ``FilterError``).
        """
        self._check_filters(filters)
        if self._search_results:
            logger.debug("Returning %d pre-configured search results", len(self._search_results))
            return self._search_results[:limit]

        query_lower = query.lower() if isinstance(query, str) else ""
        results = [
            m
            for m in self._visible(config, memory_type, kwargs.get("origins"))
            if query_lower in m.content.lower()
            and (category is None or m.metadata.get("category") == category)
            and self._matches(m, filters)
        ]
        logger.debug("Text search found %d results for query: %s", len(results), query[:30])
        return results[:limit]

    @staticmethod
    def _check_filters(filters: dict[str, Any] | None) -> None:
        """Refuse a filter on ``user_id``: the owner comes from the identity only."""
        if filters and "user_id" in filters:
            raise isolation_filter_error("user_id")

    @staticmethod
    def _matches(mem: MemorySearchResult, filters: dict[str, Any] | None) -> bool:
        return all(mem.metadata.get(k) == v for k, v in (filters or {}).items())

    async def aget(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs: Any,
    ) -> MemorySearchResult | None:
        """Get a memory by ID, or None when it is missing or outside the caller's scope."""
        result = next((m for m in self._visible(config) if m.id == memory_id), None)
        logger.debug("Get memory %s: %s", memory_id, "found" if result else "not found")
        return result

    async def aget_all(
        self,
        config: dict[str, Any],
        limit: int = 100,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        """Get the caller's memories, newest first, narrowed by ``filters`` (metadata keys)."""
        filters = kwargs.get("filters")
        self._check_filters(filters)
        visible = self._visible(config, origins=kwargs.get("origins"))
        results = [m for m in visible if self._matches(m, filters)][:limit]
        logger.debug("Get all memories: returning %d", len(results))
        return results

    async def aupdate(
        self,
        config: dict[str, Any],
        memory_id: str,
        content: str | Message,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Update the caller's own memory (reach never lets you edit another user's).

        Metadata is merged unless ``replace_metadata=True``. A ``category`` or ``memory_key``
        in ``metadata`` replaces that value, otherwise it is kept. ``origin`` and ``expires_at``
        (keyword arguments) replace the stored values; left out, they are kept. An expired
        memory can be updated. A memory that is missing or belongs to another user raises
        ``ValueError``.
        """
        self._refuse_unstorable(content=content, metadata=metadata)
        origin, expires_at = self._provenance(kwargs)
        user_id, _thread, _collection = self._extract_config_values(config)
        mem = self._owned(config, memory_id, self._require_owner_id(config, user_id))
        meta = metadata or {}
        base = {} if kwargs.get("replace_metadata") else mem.metadata
        merged = {**base, **{k: v for k, v in meta.items() if k not in _SYSTEM_KEYS}}
        category = meta.get("category") or mem.metadata.get("category")
        memory_key = meta.get("memory_key", mem.metadata.get("memory_key"))

        self.memories[memory_id] = mem.model_copy(
            update={
                "content": content if isinstance(content, str) else content.text(),
                "metadata": self._metadata(merged, category, mem.memory_type, memory_key),
                "timestamp": datetime.now(UTC),
                "origin": origin or mem.origin,
                "expires_at": expires_at or mem.expires_at,
            }
        )
        logger.debug("Updated memory %s", memory_id)
        return True

    async def adelete(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs: Any,
    ) -> bool:
        """Delete a memory in the caller's scope; ``ValueError`` when missing or foreign."""
        self._owned(config, memory_id, self._require_user_id(config))
        del self.memories[memory_id]
        del self._collection_of[memory_id]
        logger.debug("Deleted memory %s", memory_id)
        return True

    async def aforget_memory(
        self,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> int:
        """Delete the caller's own memories (a user_id is required, also under reach all).

        ``thread_id`` (from config), ``memory_type``, ``category`` and ``filters`` narrow it.

        Returns:
            Number of memories deleted
        """
        user_id, thread_id, collection = self._extract_config_values(config)
        owner = str(self._require_owner_id(config, user_id))
        memory_type = kwargs.get("memory_type")
        category = kwargs.get("category")
        self._check_filters(kwargs.get("filters"))

        to_delete = [
            m.id
            for m in self.memories.values()
            if m.user_id == owner
            and self._collection_of[m.id] == collection
            and (thread_id in (None, "") or m.thread_id == thread_id)
            and (memory_type is None or m.memory_type == memory_type)
            and (category is None or m.metadata.get("category") == category)
            and self._matches(m, kwargs.get("filters"))
        ]
        for mem_id in to_delete:
            del self.memories[mem_id]
            del self._collection_of[mem_id]

        logger.debug("Forgot %d memories for user=%s, thread=%s", len(to_delete), owner, thread_id)
        return len(to_delete)

    async def adelete_expired(self, batch_size: int = 1000, collection: str | None = None) -> int:
        """Delete every expired memory (in ``collection`` when given); return the count."""
        self._check_batch_size(batch_size)
        now = datetime.now(UTC)
        expired = [
            m.id
            for m in self.memories.values()
            if not is_live(m.expires_at, now)
            and (collection is None or self._collection_of[m.id] == collection)
        ]
        for mem_id in expired:
            del self.memories[mem_id]
            del self._collection_of[mem_id]
        return len(expired)

    async def aping(self) -> None:
        """Nothing to reach: the store lives in this process."""
        return

    async def arelease(self) -> None:
        """Clean up resources (clears all data)."""
        count = len(self.memories)
        self.clear()
        logger.debug("Released InMemoryStore, cleared %d memories", count)

    def clear(self) -> None:
        """Clear all memories and pre-configured results.

        Use this between tests to reset state.
        """
        self.memories.clear()
        self._collection_of.clear()
        self._search_results.clear()
        logger.debug("InMemoryStore cleared")
