"""In-memory store for testing - no external dependencies.

This module provides an InMemoryStore class similar to InMemoryCheckpointer,
offering a simple in-memory implementation for testing without requiring
external databases or embedding services.
"""

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from agentflow.core.state import Message
from agentflow.storage.store.base_store import BaseStore
from agentflow.storage.store.store_schema import (
    DistanceMetric,
    MemorySearchResult,
    MemoryType,
    RetrievalStrategy,
)


logger = logging.getLogger("agentflow.testing")


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

    def __init__(self):
        """Initialize an in-memory store."""
        self.memories: dict[str, MemorySearchResult] = {}
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

    async def astore(
        self,
        config: dict[str, Any],
        content: str | Message,
        memory_type: MemoryType = MemoryType.EPISODIC,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Store a memory.

        Args:
            config: Configuration dict (may contain user_id, thread_id)
            content: Memory content (string or Message)
            memory_type: Type of memory
            category: Memory category
            metadata: Additional metadata
            **kwargs: Additional store parameters

        Returns:
            Generated memory ID
        """
        mem_id = str(uuid4())
        content_str = content if isinstance(content, str) else content.text()

        self.memories[mem_id] = MemorySearchResult(
            id=mem_id,
            content=content_str,
            score=1.0,
            memory_type=memory_type,
            metadata=metadata or {},
            user_id=config.get("user_id"),
            thread_id=config.get("thread_id"),
            timestamp=datetime.now(),
        )

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
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        max_tokens: int = 4000,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        """Search memories - returns pre-configured results or text match.

        If search results were pre-configured via set_search_results(),
        returns those. Otherwise, performs simple case-insensitive text
        matching on stored memories.

        Args:
            config: Configuration dict
            query: Search query string
            memory_type: Filter by memory type
            category: Filter by category (not used in simple search)
            limit: Maximum results to return
            score_threshold: Minimum score threshold (not used in simple search)
            filters: Additional filters (not used in simple search)
            retrieval_strategy: Strategy (not used in simple search)
            distance_metric: Distance metric (not used in simple search)
            max_tokens: Max tokens (not used in simple search)
            **kwargs: Additional parameters

        Returns:
            List of matching MemorySearchResult
        """
        # Return pre-configured results if set
        if self._search_results:
            logger.debug("Returning %d pre-configured search results", len(self._search_results))
            return self._search_results[:limit]

        # Simple text search fallback
        results = []
        query_lower = query.lower() if isinstance(query, str) else ""

        for mem in self.memories.values():
            # Apply memory_type filter
            if memory_type and mem.memory_type != memory_type:
                continue

            # Simple text matching
            if query_lower in mem.content.lower():
                results.append(mem)

        logger.debug("Text search found %d results for query: %s", len(results), query[:30])
        return results[:limit]

    async def aget(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs: Any,
    ) -> MemorySearchResult | None:
        """Get a specific memory by ID.

        Args:
            config: Configuration dict
            memory_id: ID of memory to retrieve
            **kwargs: Additional parameters

        Returns:
            MemorySearchResult if found, None otherwise
        """
        result = self.memories.get(memory_id)
        logger.debug("Get memory %s: %s", memory_id, "found" if result else "not found")
        return result

    async def aget_all(
        self,
        config: dict[str, Any],
        limit: int = 100,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        """Get all memories.

        Args:
            config: Configuration dict
            limit: Maximum results to return
            **kwargs: Additional parameters

        Returns:
            List of all stored memories (up to limit)
        """
        results = list(self.memories.values())[:limit]
        logger.debug("Get all memories: returning %d of %d", len(results), len(self.memories))
        return results

    async def aupdate(
        self,
        config: dict[str, Any],
        memory_id: str,
        content: str | Message,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Update a memory.

        Args:
            config: Configuration dict
            memory_id: ID of memory to update
            content: New content
            metadata: New/additional metadata
            **kwargs: Additional parameters

        Returns:
            True if updated, False if not found
        """
        if memory_id not in self.memories:
            logger.debug("Update failed: memory %s not found", memory_id)
            return False

        mem = self.memories[memory_id]
        content_str = content if isinstance(content, str) else content.text()

        # Create updated memory
        self.memories[memory_id] = MemorySearchResult(
            id=memory_id,
            content=content_str,
            score=mem.score,
            memory_type=mem.memory_type,
            metadata={**mem.metadata, **(metadata or {})},
            user_id=mem.user_id,
            thread_id=mem.thread_id,
            timestamp=datetime.now(),
        )

        logger.debug("Updated memory %s", memory_id)
        return True

    async def adelete(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs: Any,
    ) -> bool:
        """Delete a memory.

        Args:
            config: Configuration dict
            memory_id: ID of memory to delete
            **kwargs: Additional parameters

        Returns:
            True if deleted, False if not found
        """
        if memory_id in self.memories:
            del self.memories[memory_id]
            logger.debug("Deleted memory %s", memory_id)
            return True

        logger.debug("Delete failed: memory %s not found", memory_id)
        return False

    async def aforget_memory(
        self,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> int:
        """Delete all memories for a user/agent.

        Args:
            config: Configuration dict (may contain user_id, thread_id)
            **kwargs: Additional parameters

        Returns:
            Number of memories deleted
        """
        user_id = config.get("user_id")
        thread_id = config.get("thread_id")

        to_delete = []
        for mem_id, mem in self.memories.items():
            if user_id and mem.user_id != user_id:
                continue
            if thread_id and mem.thread_id != thread_id:
                continue
            to_delete.append(mem_id)

        for mem_id in to_delete:
            del self.memories[mem_id]

        logger.debug(
            "Forgot %d memories for user=%s, thread=%s", len(to_delete), user_id, thread_id
        )
        return len(to_delete)

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
        self._search_results.clear()
        logger.debug("InMemoryStore cleared")
