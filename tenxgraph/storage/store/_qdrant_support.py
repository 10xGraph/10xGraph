"""Helpers of ``QdrantStore``: payload keys, filters, point conversion, collection setup.

``qdrant_store.py`` keeps the public names; this module holds the parts the read and write
methods share.
"""

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from tenxgraph.core.state import Message
from tenxgraph.storage.columns import ColumnSet, ColumnSpec, Isolation, resolve_isolation
from tenxgraph.utils.log_safety import describe_error

from .base_store import BaseStore
from .store_schema import (
    DistanceMetric,
    MemoryRecord,
    MemorySearchResult,
    MemoryType,
    RetrievalStrategy,
)


logger = logging.getLogger("tenxgraph.store")

# Payload fields the store owns. Caller metadata is stored alongside them but can never
# replace them -- in particular ``user_id``, which every read filters on. Tenant column names
# are added per store (``QdrantStore._reserved_keys``).
_SYSTEM_PAYLOAD_KEYS = frozenset(
    {"content", "user_id", "thread_id", "memory_type", "category", "timestamp"}
)

# Payload fields left out of ``MemorySearchResult.metadata`` (they have their own result
# fields). Tenant column names are added per store.
_HIDDEN_RESULT_KEYS = frozenset({"content", "user_id", "thread_id", "timestamp"})

# Names a tenant column cannot take: the store's payload fields.
_RESERVED_COLUMN_NAMES = _SYSTEM_PAYLOAD_KEYS | {"memory_key", "agent_id", "collection", "id"}

# Keyword payload indexes every collection gets (remote Qdrant refuses unindexed filters).
_KEYWORD_INDEXES = ("user_id", "memory_type", "category", "memory_key")

# ``aget_all`` orders by this payload field (newest first). ``timestamp`` is set on every
# write and moved on every update, so it is the store's ``updated_at``.
_ORDER_FIELD = "timestamp"


def payload_matches(
    payload: dict[str, Any], user_id: Any, conditions: Sequence[tuple[ColumnSpec, Any]]
) -> bool:
    """Whether a point's payload passes the scope: ``user_id`` (when not None) and every
    tenant condition. A ``None`` condition value (a writer without an optional value) matches
    only a point without that value, like ``IS NOT DISTINCT FROM`` on ``PgVectorStore``."""
    if user_id is not None and payload.get("user_id") != user_id:
        return False
    for spec, value in conditions:
        if payload.get(spec.name) != (None if value is None else str(value)):
            return False
    return True


class QdrantHelpersMixin(BaseStore):
    """Shared helpers of ``QdrantStore``. ``QdrantStore.__init__`` sets ``client``,
    ``embedding``, ``columns``, ``tenant_column_names`` and the collection cache and lock."""

    client: Any
    embedding: Any
    columns: ColumnSet
    tenant_column_names: tuple[str, ...]
    _collection_cache: set[str]
    _collection_lock: asyncio.Lock
    _default_distance_metric: DistanceMetric

    @staticmethod
    def _check_strategy(strategy: RetrievalStrategy) -> None:
        if strategy != RetrievalStrategy.SIMILARITY:
            raise ValueError(
                f"retrieval_strategy {str(strategy)!r} is not supported by QdrantStore; "
                "only 'similarity' is available."
            )

    @property
    def _reserved_keys(self) -> frozenset[str]:
        """Payload keys caller metadata cannot set: the store's fields and tenant columns."""
        return _SYSTEM_PAYLOAD_KEYS | frozenset(self.tenant_column_names)

    def _isolation(self, config: dict[str, Any], *, write: bool = False) -> Isolation:
        """Tenant values and the tenant conditions of the caller's reach for one call.

        Same rule as ``PgVectorStore`` (``tenxgraph.storage.columns.resolve_isolation``):
        values only from ``config["user"]``, ``write=True`` pins reach ``own`` and raises when a
        required column has no value, a read raises when a picked column has no value, and
        reach ``tenant`` raises when the store declares no tenant columns.
        """
        return resolve_isolation(self.columns, config, write=write, row_level_security=False)

    def _tenant_payload(self, ctx: Isolation) -> dict[str, str]:
        """The tenant values a write stores, as text. A missing optional value is left out."""
        return {
            name: str(ctx.values[name])
            for name in self.tenant_column_names
            if ctx.values.get(name) is not None
        }

    def _caller_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        """The caller's metadata kept in a payload (no store fields, no tenant values)."""
        return {k: v for k, v in payload.items() if k not in self._reserved_keys}

    def _distance_metric_to_qdrant(self, metric: DistanceMetric):
        """Convert framework distance metric to Qdrant distance."""
        from qdrant_client.http.models import Distance

        mapping = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLIDEAN: Distance.EUCLID,
            DistanceMetric.DOT_PRODUCT: Distance.DOT,
            DistanceMetric.MANHATTAN: Distance.MANHATTAN,
        }
        return mapping.get(metric, Distance.COSINE)

    def _point_to_search_result(self, point) -> MemorySearchResult:
        """Convert a Qdrant point to a MemorySearchResult.

        ``metadata`` is the caller's metadata plus ``category``, ``memory_type`` and
        ``memory_key``; content, owner, thread, timestamp and tenant values have their own
        fields or stay internal.
        """
        payload = getattr(point, "payload", {}) or {}
        hidden = _HIDDEN_RESULT_KEYS | frozenset(self.tenant_column_names)

        try:
            memory_type = MemoryType(payload.get("memory_type", "episodic"))
        except ValueError:
            memory_type = MemoryType.EPISODIC

        timestamp = None
        timestamp_str = payload.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError):
                timestamp = None

        return MemorySearchResult(
            id=str(point.id),
            content=payload.get("content", ""),
            score=float(getattr(point, "score", 1.0) or 0.0),
            memory_type=memory_type,
            metadata={k: v for k, v in payload.items() if k not in hidden},
            vector=getattr(point, "vector", None),
            user_id=payload.get("user_id"),
            thread_id=payload.get("thread_id")
            or payload.get("agent_id"),  # Support both thread_id and agent_id
            timestamp=timestamp,
        )

    def _build_qdrant_filter(
        self,
        user_id: str | int | None = None,
        thread_id: str | None = None,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        filters: dict[str, Any] | None = None,
        tenant: Sequence[tuple[ColumnSpec, Any]] = (),
    ) -> Any | None:
        """Build a Qdrant filter (AND of every condition) from the parameters.

        ``tenant`` is the scope's tenant conditions (``Isolation.conditions``): a value
        matches as text; a ``None`` value (a writer without an optional value) matches only
        points without that value.
        """
        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            IsEmptyCondition,
            MatchValue,
            PayloadField,
        )

        conditions: list[Any] = []
        if user_id is not None:
            conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        for spec, value in tenant:
            if value is None:
                conditions.append(IsEmptyCondition(is_empty=PayloadField(key=spec.name)))
            else:
                conditions.append(FieldCondition(key=spec.name, match=MatchValue(value=str(value))))
        if thread_id is not None and thread_id != "":
            conditions.append(FieldCondition(key="thread_id", match=MatchValue(value=thread_id)))
        if memory_type:
            conditions.append(
                FieldCondition(key="memory_type", match=MatchValue(value=memory_type.value))
            )
        if category:
            conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))

        if filters:
            for key, value in filters.items():
                if isinstance(value, str | int | bool):
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        return Filter(must=conditions) if conditions else None

    async def _retrieve_point(self, collection: str, memory_id: str) -> Any | None:
        """The point with ``memory_id``, or None. Backend errors propagate."""
        points = await self.client.retrieve(collection_name=collection, ids=[memory_id])
        return points[0] if points else None

    async def _ensure_can_write_id(
        self,
        collection: str,
        user_id: str | int,
        memory_id: str,
        conditions: Sequence[tuple[ColumnSpec, Any]] = (),
    ) -> None:
        """Refuse a caller-chosen ``memory_id`` that already belongs to another owner.

        ``astore`` upserts, so without this check a chosen id would silently replace another
        user's (or another tenant's) memory. Reusing one of your own ids (an intentional
        overwrite) is still allowed. Checked on every reach: writes stay reach ``own``, and
        ``conditions`` are the write's tenant conditions.
        """
        point = await self._retrieve_point(collection, memory_id)
        if point is None:
            return
        if not payload_matches(getattr(point, "payload", None) or {}, user_id, conditions):
            raise PermissionError(f"memory_id '{memory_id}' belongs to another user")

    async def _ensure_collection_exists(self, collection: str) -> None:
        """Ensure collection exists, create if not.

        The `_collection_cache` fast path means the network round-trip happens once
        per collection, not on every store/search/get. The lock closes the cold-start
        race: without it, N concurrent first-requests all missed the cache together
        and each issued its own get_collections()/create_collection() call.
        """
        if collection in self._collection_cache:
            return

        from qdrant_client.http.models import PayloadSchemaType, VectorParams

        async with self._collection_lock:
            # Re-check under the lock: another coroutine may have ensured it while
            # we were waiting.
            if collection in self._collection_cache:
                return

            await self._ensure_collection_exists_locked(collection, PayloadSchemaType, VectorParams)

    async def _ensure_collection_exists_locked(
        self,
        collection: str,
        PayloadSchemaType,
        VectorParams,
    ) -> None:
        """Create the collection and its payload indexes. Caller holds the lock."""
        from qdrant_client.http.models import KeywordIndexParams

        try:
            collections = await self.client.get_collections()
            existing_names = {col.name for col in collections.collections}

            if collection not in existing_names:
                await self.client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=self.embedding.dimension,
                        distance=self._distance_metric_to_qdrant(
                            self._default_distance_metric,
                        ),
                    ),
                )
                logger.info(f"Created collection: {collection}")

            # Payload indexes for filterable fields, required for remote/cloud Qdrant where
            # full scan is disabled. create_payload_index is idempotent, so this is safe for
            # new and pre-existing collections. Tenant columns use Qdrant's multitenancy
            # index (is_tenant), the timestamp a datetime index (aget_all orders by it).
            indexes: list[tuple[str, Any]] = [
                (name, PayloadSchemaType.KEYWORD) for name in _KEYWORD_INDEXES
            ]
            indexes += [
                (name, KeywordIndexParams(type="keyword", is_tenant=True))
                for name in self.tenant_column_names
            ]
            indexes.append((_ORDER_FIELD, PayloadSchemaType.DATETIME))
            for field_name, schema in indexes:
                try:
                    await self.client.create_payload_index(
                        collection_name=collection,
                        field_name=field_name,
                        field_schema=schema,
                    )
                except Exception:
                    # Best-effort: log and continue so one failing index
                    # doesn't prevent the collection from being usable.
                    logger.debug(
                        "Payload index for '%s' on '%s' may already exist or failed",
                        field_name,
                        collection,
                    )
            logger.info(f"Ensured payload indexes for collection: {collection}")

            self._collection_cache.add(collection)
        except Exception as e:
            logger.error(f"Error ensuring collection {collection} exists: {describe_error(e)}")
            raise

    def _prepare_content(self, content: str | Message) -> str:
        """Extract text content from string or Message."""
        if isinstance(content, Message):
            return content.text()
        return content

    def _create_memory_record(
        self,
        content: str | Message,
        user_id: str | None = None,
        thread_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Create a memory record from parameters."""
        text_content = self._prepare_content(content)

        if isinstance(content, Message):
            return MemoryRecord.from_message(
                content,
                user_id=user_id,
                thread_id=thread_id,
                additional_metadata=metadata,
            )

        return MemoryRecord(
            content=text_content,
            user_id=user_id,
            thread_id=thread_id,
            memory_type=memory_type,
            metadata=metadata or {},
            category=category,
        )
