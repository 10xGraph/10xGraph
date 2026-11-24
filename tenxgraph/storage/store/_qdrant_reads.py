"""Read and delete methods of ``QdrantStore``: search, get by id, list, delete, forget."""

import logging
from datetime import UTC, datetime
from typing import Any

from ._qdrant_support import (
    _EXPIRES_FIELD,
    _ORDER_FIELD,
    QdrantHelpersMixin,
    payload_live,
    payload_matches,
)
from .base_store import limit_to_token_budget
from .store_schema import (
    DistanceMetric,
    MemorySearchResult,
    MemoryType,
    RetrievalStrategy,
)


logger = logging.getLogger("tenxgraph.store")


class QdrantReadsMixin(QdrantHelpersMixin):
    """``asearch``, ``aget``, ``aget_all``, ``adelete`` and ``aforget_memory``. Every call is
    scoped by the caller's reach: ``user_id`` (reach ``own``) and the tenant conditions it picks."""

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
        from the search filter so results span all conversations for the user. Expired
        memories are never returned; ``origins`` (keyword argument) narrows by who wrote them.
        Both are part of the query filter, so ``limit`` counts only points that pass them.
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
            live=True,
            origins=kwargs.get("origins"),
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
        """Get a memory by id; ``None`` when it does not exist, has expired or is outside the
        caller's scope (another user's on reach ``own``, another tenant's). Backend errors
        raise."""
        point = await self._scoped_point(config, memory_id)
        if point is None or not payload_live(getattr(point, "payload", None) or {}):
            return None
        return self._point_to_search_result(point)

    async def _scoped_point(self, config: dict[str, Any], memory_id: str) -> Any | None:
        """The point with ``memory_id`` when it is inside the caller's scope (expired or not)."""
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
        return point

    async def aget_all(
        self,
        config: dict[str, Any],
        limit: int = 100,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """List the caller's memories in the collection, newest first.

        ``filters`` and ``origins`` (keyword arguments) narrow the list, same rules as
        ``asearch``; expired memories are left out. Ordered by
        the ``timestamp`` payload field (set on every write and update), which has a datetime
        payload index for remote Qdrant.
        """
        from qdrant_client.http.models import Direction, OrderBy

        user_id, _, collection = self._extract_config_values(config)
        scoped_user_id = self._require_user_id(config, user_id)
        ctx = self._isolation(config)

        await self._ensure_collection_exists(collection)

        search_filter = self._build_qdrant_filter(
            user_id=scoped_user_id,
            filters=kwargs.get("filters"),
            tenant=ctx.conditions,
            live=True,
            origins=kwargs.get("origins"),
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

    async def adelete(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs,
    ) -> None:
        """Delete a memory by id; ``ValueError`` when it does not exist or is outside the
        caller's scope (another user's on reach ``own``, another tenant's)."""
        from qdrant_client.http import models

        _user_id, _thread_id, collection = self._extract_config_values(config)

        # The caller's scope applies, so a memory outside it is "not found" here. Expired
        # memories can still be deleted.
        existing = await self._scoped_point(config, memory_id)
        if existing is None:
            raise ValueError(f"Memory {memory_id} not found")

        await self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=[memory_id]),
        )

        logger.debug(f"Deleted memory {memory_id} from collection {collection}")

    async def adelete_expired(self, batch_size: int = 1000, collection: str | None = None) -> int:
        """Delete every expired point in a collection, over all owners; return how many.

        ``collection`` defaults to the store's collection name; with a function default it must
        be given. One server-side delete by filter (``batch_size`` is checked, not needed).
        """
        from qdrant_client.http import models

        self._check_batch_size(batch_size)
        name = collection if collection is not None else self.collection
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "adelete_expired needs a collection name when the default is a function"
            )
        await self._ensure_collection_exists(name)
        expired = models.Filter(
            must=[
                models.FieldCondition(
                    key=_EXPIRES_FIELD,
                    range=models.DatetimeRange(lte=datetime.now(UTC).isoformat()),
                )
            ]
        )
        count = (await self.client.count(name, count_filter=expired, exact=True)).count
        if count:
            await self.client.delete(
                collection_name=name, points_selector=models.FilterSelector(filter=expired)
            )
        logger.info("QdrantStore deleted %d expired memories from %s", count, name)
        return count

    async def aforget_memory(
        self,
        config: dict[str, Any],
        **kwargs,
    ) -> None:
        """Delete the memories of one user, optionally narrowed.

        ``user_id`` is required (also under reach ``all``): forgetting is always for a named
        owner, never the whole collection. The tenant conditions of the caller's reach apply
        too. With no narrowing every memory of that user in the
        collection goes. ``thread_id`` (from config), ``memory_type``, ``category`` and
        ``filters`` (keyword arguments) narrow the delete, all combined with AND. A ``filters``
        key or value that cannot be matched exactly raises ``FilterError`` instead of being
        skipped, because skipping it would delete more than the caller asked for.
        """
        from qdrant_client.http import models  # noqa: PLC0415

        user_id, thread_id, collection = self._extract_config_values(config)
        user_id = self._require_owner_id(config, user_id)
        ctx = self._isolation(config)

        memory_type = kwargs.get("memory_type")
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        category = kwargs.get("category")
        filters = kwargs.get("filters")

        delete_filter = self._build_qdrant_filter(
            user_id=user_id,
            thread_id=thread_id,
            memory_type=memory_type,
            category=category,
            filters=filters,
            tenant=ctx.conditions,
        )
        await self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=delete_filter or models.Filter(must=[])),
        )

        logger.info(
            f"Deleted memories for user_id={user_id}, thread_id={thread_id}, "
            f"memory_type={memory_type}, category={category}, filters={filters} "
            f"in collection {collection}"
        )
