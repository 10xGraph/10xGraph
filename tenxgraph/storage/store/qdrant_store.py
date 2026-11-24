"""
Qdrant Vector Store Implementation for TAF Framework

This module provides a modern, async-first implementation of BaseStore using Qdrant
as the backend vector database. Supports both local and cloud Qdrant deployments
with configurable embedding services.

Helpers live in ``_qdrant_support.py``, the read methods in ``_qdrant_reads.py`` and the
deletes in ``_qdrant_deletes.py``.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from tenxgraph.core.state import Message
from tenxgraph.storage.columns import parse_columns
from tenxgraph.storage.media.url_policy import redact_url

from ._qdrant_deletes import QdrantDeletesMixin
from ._qdrant_support import (
    _RESERVED_COLUMN_NAMES,
    _SYSTEM_PAYLOAD_KEYS,  # noqa: F401 - kept importable from here
    payload_matches,
)
from .embedding.base_embedding import BaseEmbedding
from .store_schema import (
    DistanceMetric,
    MemoryType,
    RetrievalStrategy,
)


HAS_QDRANT = False

try:
    import qdrant_client  # noqa: F401

    HAS_QDRANT = True
except ImportError:
    pass

logger = logging.getLogger("tenxgraph.store")

# Default collection name used when none is specified via config or constructor.
# Centralised here so factory functions and the class share the same value
# without duplicating the magic string.
DEFAULT_COLLECTION = "agentflow_memories"


class QdrantStore(QdrantDeletesMixin):
    """
    Modern async-first Qdrant-based vector store implementation.

    Features:
    - Async-only operations for better performance
    - Local and cloud Qdrant deployment support
    - Configurable embedding services
    - Efficient vector similarity search with multiple strategies
    - Collection management with automatic creation
    - Rich metadata filtering capabilities
    - User, tenant and agent-scoped operations

    Example:
        ```python
        # Local Qdrant with OpenAI embeddings
        store = QdrantStore(path="./qdrant_data", embedding_service=OpenAIEmbeddingService())

        # Remote Qdrant
        store = QdrantStore(host="localhost", port=6333, embedding_service=OpenAIEmbeddingService())

        # Cloud Qdrant
        store = QdrantStore(
            url="https://xyz.qdrant.io",
            api_key="your-api-key",
            embedding_service=OpenAIEmbeddingService(),
        )
        ```
    """

    supported_retrieval_strategies = frozenset({RetrievalStrategy.SIMILARITY})

    def __init__(
        self,
        embedding: BaseEmbedding,
        path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | Callable[[dict[str, Any]], str] | None = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY,
        tenant_columns: Any = None,
        **kwargs: Any,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            embedding: Service for generating embeddings
            path: Path for local Qdrant (file-based storage)
            host: Host for remote Qdrant server
            port: Port for remote Qdrant server
            url: URL for Qdrant cloud
            api_key: API key for Qdrant cloud
            collection: Default collection: a name, or a function ``(config) -> name`` that
                receives the config passed to each operation. Used when the config has no
                ``collection`` (defaults to DEFAULT_COLLECTION). A function must read the
                authenticated identity (``config["user"]`` / ``config["user_id"]``), never
                state. Precedence: scope ``collection`` (memory config), then a ``collection``
                in the config, then this default.
            distance_metric: Distance metric of the collection (fixed at collection creation)
            retrieval_strategy: Default search strategy when a call passes ``None``. Only
                ``SIMILARITY`` is supported.
            tenant_columns: Payload fields that isolate data, such as
                ``{"org_id": "uuid", "project_id": "text"}`` (or a list of names for ``text``
                fields), same declaration as ``PgVectorStore(tenant_columns=...)``. Each gets
                a keyword payload index marked ``is_tenant`` (Qdrant's multitenancy index).
                Every write stores the caller's values as text; caller metadata cannot set
                them. Every read, update, delete, forget and search is filtered by the tenant
                columns the caller's reach picks (all of them when there is no identity, see
                ``tenxgraph.core.authz``). A picked column whose value is missing raises, it
                is never dropped from the filter. Required by default;
                ``{"type": ..., "required": False}`` makes a column optional for writes.
                Types are ``text``, ``bigint``, ``integer`` or ``uuid`` (checked like
                ``PgVectorStore``). Another tenant's memory behaves like a missing one.
                Shared data (``system_user_id(name)``) is shared only inside the tenant values
                the scope picks. Points written before a column was declared have no value
                and are invisible to scopes that pick it until you backfill them (reach
                ``all`` still sees them). Without tenant columns reach ``tenant`` raises.
            **kwargs: Additional client parameters

        Trusted-source rule for ``tenant_columns``: values come only from ``config["user"]``
        (an ``Identity`` or a dict that validates into one, tenant values under ``tenant``),
        set by your auth class. Never read them from request config, metadata, state or model
        output.
        """
        if not HAS_QDRANT:
            raise ImportError(
                "qdrant-client package is required for QdrantStore. "
                "Install with `pip install '10xgraph[qdrant]'`."
            )
        self.embedding = embedding
        self.columns = parse_columns(
            tenant_columns, None, reserved=_RESERVED_COLUMN_NAMES, table="QdrantStore"
        )
        #: Names of the declared tenant columns (read by the authz helpers and compile()).
        self.tenant_column_names: tuple[str, ...] = self.columns.tenant_names

        # Initialize async client
        from qdrant_client import AsyncQdrantClient

        if path:
            self.client = AsyncQdrantClient(path=path, **kwargs)
        elif url:
            self.client = AsyncQdrantClient(url=url, api_key=api_key, **kwargs)
        else:
            host = host or "localhost"
            port = port or 6333
            self.client = AsyncQdrantClient(host=host, port=port, api_key=api_key, **kwargs)

        # Cache for collection existence checks, so the network round-trip happens
        # once per collection rather than on every store/search/get.
        self._collection_cache = set()
        # Serializes the cold-start miss: without it, N concurrent first-requests
        # all miss the cache together and each issues its own create_collection.
        self._collection_lock = asyncio.Lock()
        self._setup_lock = asyncio.Lock()

        self.collection: str | Callable[[dict[str, Any]], str] = collection or DEFAULT_COLLECTION
        self._default_distance_metric = distance_metric
        self._default_retrieval_strategy = RetrievalStrategy(retrieval_strategy)
        self._check_strategy(self._default_retrieval_strategy)

        # A Qdrant URL can carry credentials, so the log shows it without them.
        shown_url = redact_url(url) if url else url
        logger.info(
            f"Initialized QdrantStore with config: path={path}, host={host}, url={shown_url}"
        )

    async def asetup(self) -> Any:
        """Set up the store and ensure the default collection exists.

        With a function default there is no fixed name, so nothing is created here;
        collections are created lazily by the first operation that uses them.
        """
        async with self._setup_lock:
            if isinstance(self.collection, str):
                await self._ensure_collection_exists(self.collection)
        return True

    # --- BaseStore abstract method implementations (reads and deletes in mixins) ---

    async def astore(
        self,
        config: dict[str, Any],
        content: str | Message,
        memory_type: MemoryType = MemoryType.EPISODIC,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Store a new memory.

        A record always has an owner: an empty ``user_id`` raises, even under
        reach ``all``. Use ``system_user_id(name)`` for shared data. Tenant values come from
        ``config["user"]``; a required one that is missing raises. A ``memory_id`` (keyword
        argument) that belongs to another user or tenant raises ``PermissionError``.
        """
        user_id, thread_id, collection = self._extract_config_values(config)
        user_id = self._require_owner_id(config, user_id)
        ctx = self._isolation(config, write=True)

        from qdrant_client.http.models import PointStruct

        await self._ensure_collection_exists(collection)

        record = self._create_memory_record(
            content=content,
            user_id=user_id,
            thread_id=thread_id,
            memory_type=memory_type,
            category=category,
            metadata=metadata,
        )
        requested_id = kwargs.get("memory_id")
        if requested_id:
            await self._ensure_can_write_id(collection, user_id, requested_id, ctx.conditions)
        record.id = requested_id or await self.generate_framework_id()

        text_content = self._prepare_content(content)
        vector = await self.embedding.aembed(text_content)
        if not vector or len(vector) != self.embedding.dimension:
            raise ValueError("Embedding service returned invalid vector")

        reserved = self._reserved_keys
        payload = {
            **{k: v for k, v in record.metadata.items() if k not in reserved},
            **self._tenant_payload(ctx),
            "content": record.content,
            "user_id": record.user_id,
            "thread_id": record.thread_id,
            "memory_type": record.memory_type.value,
            "category": record.category,
            "timestamp": record.timestamp.isoformat() if record.timestamp else None,
        }

        point = PointStruct(id=record.id, vector=vector, payload=payload)
        await self.client.upsert(collection_name=collection, points=[point])

        logger.debug(f"Stored memory {record.id} in collection {collection}")
        return record.id

    async def aupdate(
        self,
        config: dict[str, Any],
        memory_id: str,
        content: str | Message,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Update a memory's content (re-embedded) and metadata.

        Metadata is merged into the stored metadata unless ``replace_metadata=True``. A
        ``category`` or ``memory_key`` in ``metadata`` updates it; otherwise it is kept (also
        with ``replace_metadata=True``). Writes stay reach ``own``: a memory that does not
        exist or belongs to another user or tenant raises ``ValueError`` (not found), on
        every reach, as on ``PgVectorStore``. Tenant values never change on update.
        """
        from qdrant_client.http.models import PointStruct

        user_id, _thread_id, collection = self._extract_config_values(config)
        owner = self._require_owner_id(config, user_id)  # writes stay reach own
        ctx = self._isolation(config, write=True)

        await self._ensure_collection_exists(collection)
        point = await self._retrieve_point(collection, memory_id)
        payload = (getattr(point, "payload", None) or {}) if point is not None else {}
        # thread_id is not an access-control boundary for long-term memory.
        if point is None or not payload_matches(payload, owner, ctx.conditions):
            raise ValueError(f"Memory {memory_id} not found")

        text_content = self._prepare_content(content)
        new_vector = await self.embedding.aembed(text_content)
        if not new_vector or len(new_vector) != self.embedding.dimension:
            raise ValueError("Embedding service returned invalid vector")

        new_metadata = dict(metadata or {})
        if kwargs.get("replace_metadata", False):
            updated_metadata = dict(new_metadata)
            if "memory_key" not in updated_metadata and payload.get("memory_key") is not None:
                updated_metadata["memory_key"] = payload["memory_key"]
        else:
            updated_metadata = {**self._caller_metadata(payload), **new_metadata}

        reserved = self._reserved_keys
        updated_payload = {
            **{k: v for k, v in updated_metadata.items() if k not in reserved},
            **{name: payload[name] for name in self.tenant_column_names if name in payload},
            "content": text_content,
            "user_id": payload.get("user_id"),
            "thread_id": payload.get("thread_id"),
            "memory_type": payload.get("memory_type", MemoryType.EPISODIC.value),
            "category": new_metadata.get("category") or payload.get("category") or "general",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        new_point = PointStruct(id=memory_id, vector=new_vector, payload=updated_payload)
        await self.client.upsert(collection_name=collection, points=[new_point])

        logger.debug(f"Updated memory {memory_id} in collection {collection}")

    async def aping(self) -> None:
        """Check the Qdrant server is reachable; raise when it is not.

        Lists collections, a read-only call that needs no collection to exist and neither creates
        collections nor embeds anything. It is bounded by the client's configured ``timeout``.
        """
        await self.client.get_collections()

    async def arelease(self) -> None:
        """Clean up resources."""
        if hasattr(self.client, "close"):
            await self.client.close()
        logger.info("QdrantStore resources released")


# Convenience factory functions


def create_local_qdrant_store(
    path: str,
    embedding: BaseEmbedding,
    collection: str | Callable[[dict[str, Any]], str] = DEFAULT_COLLECTION,
    **kwargs,
) -> QdrantStore:
    """Create a local Qdrant store."""
    return QdrantStore(
        embedding=embedding,
        path=path,
        collection=collection,
        **kwargs,
    )


def create_remote_qdrant_store(
    host: str,
    port: int,
    embedding: BaseEmbedding,
    collection: str | Callable[[dict[str, Any]], str] = DEFAULT_COLLECTION,
    **kwargs,
) -> QdrantStore:
    """Create a remote Qdrant store."""
    return QdrantStore(
        embedding=embedding,
        host=host,
        port=port,
        collection=collection,
        **kwargs,
    )


def create_cloud_qdrant_store(
    url: str,
    api_key: str,
    embedding: BaseEmbedding,
    collection: str | Callable[[dict[str, Any]], str] = DEFAULT_COLLECTION,
    **kwargs,
) -> QdrantStore:
    """Create a cloud Qdrant store."""
    return QdrantStore(
        embedding=embedding,
        url=url,
        api_key=api_key,
        collection=collection,
        **kwargs,
    )
