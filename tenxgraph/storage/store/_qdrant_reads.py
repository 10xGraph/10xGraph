"""Read methods of ``QdrantStore``: search, get by id, list."""

import logging
from typing import Any

from ._qdrant_support import _ORDER_FIELD, QdrantHelpersMixin, payload_matches
from .base_store import limit_to_token_budget
from .store_schema import (
    DistanceMetric,
    MemorySearchResult,
    MemoryType,
    RetrievalStrategy,
)


logger = logging.getLogger("tenxgraph.store")


class QdrantReadsMixin(QdrantHelpersMixin):
    """``asearch``, ``aget`` and ``aget_all``. Every read is scoped by the caller's reach:
    ``user_id`` (reach ``own``) and the tenant conditions it picks."""

    _default_retrieval_strategy: RetrievalStrategy

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
        **kwargs,
    ) -> list[MemorySearchResult]:
        """Search memories by content similarity.

        Long-term memory is cross-thread — thread_id is intentionally excluded
        from the search filter so results span all conversations for the user.
        """
        strategy = RetrievalStrategy(
            retrieval_strategy
            if retrieval_strategy is not None
            else self._default_retrieval_strategy
        )
        self._check_strategy(strategy)
        if distance_metric is not None and distance_metric != self._default_distance_metric:
            raise ValueError(
                f"distance_metric {str(distance_metric)!r} differs from the store's "
                f"{str(self._default_distance_metric)!r}. A Qdrant collection's distance is "
                "fixed at collection creation; set it on QdrantStore(distance_metric=...)."
            )

        user_id, _thread_id, collection = self._extract_config_values(config)
        scoped_user_id = self._require_user_id(config, user_id)
        ctx = self._isolation(config)

        await self._ensure_collection_exists(collection)

        # Duck-typed embedding services may not define aembed_query.
        embed_query = getattr(self.embedding, "aembed_query", None) or self.embedding.aembed
        query_vector = await embed_query(query)
        if not query_vector or len(query_vector) != self.embedding.dimension:
            raise ValueError("Embedding service returned invalid vector")

        # Search by owner and tenant scope only; thread_id is excluded. The user filter is
        # dropped when the caller's reach is "tenant" or "all".
        search_filter = self._build_qdrant_filter(
            user_id=scoped_user_id,
            memory_type=memory_type,
            category=category,
            filters=filters,
            tenant=ctx.conditions,
        )

        # (query_points replaced search in qdrant-client 1.10; payloads are returned by default)
        response = await self.client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold,
        )

        results = limit_to_token_budget(
            [self._point_to_search_result(point) for point in response.points], max_tokens
        )

        logger.debug(f"Found {len(results)} memories for query in collection {collection}")
        return results

    async def aget(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs,
    ) -> MemorySearchResult | None:
        """Get a memory by id; ``None`` when it does not exist or is outside the caller's
        scope (another user's on reach ``own``, another tenant's). Backend errors raise."""
        user_id, _thread_id, collection = self._extract_config_values(config)
        scoped_user_id = self._require_user_id(config, user_id)
        ctx = self._isolation(config)

        await self._ensure_collection_exists(collection)
        point = await self._retrieve_point(collection, memory_id)
        # thread_id is intentionally not checked: long-term memory is cross-thread.
        if point is None or not payload_matches(
            getattr(point, "payload", None) or {}, scoped_user_id, ctx.conditions
        ):
            return None
        return self._point_to_search_result(point)

    async def aget_all(
        self,
        config: dict[str, Any],
        limit: int = 100,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """List the caller's memories in the collection, newest first.

        ``filters`` (keyword argument) narrows the list, same rules as ``asearch``. Ordered by
        the ``timestamp`` payload field (set on every write and update), which has a datetime
        payload index for remote Qdrant.
        """
        from qdrant_client.http.models import Direction, OrderBy

        user_id, _, collection = self._extract_config_values(config)
        scoped_user_id = self._require_user_id(config, user_id)
        ctx = self._isolation(config)

        await self._ensure_collection_exists(collection)

        search_filter = self._build_qdrant_filter(
            user_id=scoped_user_id, filters=kwargs.get("filters"), tenant=ctx.conditions
        )

        # Listing needs no vector, so scroll through the filtered points (payloads are
        # returned by default; scroll points carry no score).
        points, _next_offset = await self.client.scroll(
            collection_name=collection,
            scroll_filter=search_filter,
            limit=limit,
            order_by=OrderBy(key=_ORDER_FIELD, direction=Direction.DESC),
        )

        results = [self._point_to_search_result(point) for point in points]

        logger.debug(f"Found {len(results)} memories for query in collection {collection}")
        return results
