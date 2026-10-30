"""Mem0 Long-Term Memory Store

Async-first implementation of :class:`BaseStore` that uses the `mem0` library
as a managed long-term memory layer. In TAF we treat the *graph state* as
short-term (ephemeral per run / session) memory and a store implementation as
long-term, durable memory. This module wires Mem0 so that:

* ``astore`` / ``asearch`` / etc. map to Mem0's `add`, `search`, `get_all`, `update`, `delete`.
* We maintain a generated UUID (framework memory id) separate from the Mem0 internal id.
* Metadata is enriched to retain memory type, category, timestamps and app scoping.
* The public async methods satisfy the :class:`BaseStore` contract (``astore``, ``abatch_store``,
  ``asearch``, ``aget``, ``aupdate``, ``adelete``, ``aforget_memory`` and ``arelease``).

Design notes:
--------------
Mem0 (>= 0.2.x / 2025 spec) still exposes synchronous Python APIs. We off-load
blocking calls to a thread executor to keep the interface awaitable. Where Mem0
does not support an operation directly (e.g. fetch by custom memory id) we
fallback to scanning ``get_all`` for the user. For batch insertion we parallelise
Add operations with gather while bounding concurrency (simple semaphore) to
avoid thread explosion.

The store interprets the supplied ``config`` mapping passed to every method as:
``{"user_id": str | None, "app_id": str | None}``. Conversation-level scoping is
intentionally ignored so long-term memory retrieval and writes work across threads.

Prerequisite: install mem0.
```
pip install mem0ai
```
Optional vector DB / embedder / llm configuration should be supplied through
Mem0's native configuration structure (see upstream docs - memory configuration,
vector store configuration). You can also use helper factory function
``create_mem0_store_with_qdrant`` for quick Qdrant backing.
"""

import logging
import secrets
from collections.abc import Awaitable, Iterable
from datetime import UTC, datetime
from typing import Any

from injectq import InjectQ

from tenxgraph.core.state import Message

from .base_store import BaseStore, limit_to_token_budget
from .qdrant_store import DEFAULT_COLLECTION
from .store_schema import MemorySearchResult, MemoryType


HAS_MEM0 = False

try:
    from mem0 import AsyncMemory
    from mem0.configs.base import MemoryConfig

    HAS_MEM0 = True
except ImportError:
    MemoryConfig = None
    AsyncMemory = None

logger = logging.getLogger("tenxgraph.store")

# Keys that choose whose memories a Mem0 call touches. The store sets them from the
# authenticated config; a caller-supplied ``filters`` dict may never carry them.
_RESERVED_FILTER_KEYS = frozenset({"user_id", "agent_id", "app_id", "run_id"})
_LOGICAL_FILTER_KEYS = frozenset({"AND", "OR", "NOT"})


def _strip_reserved_filters(filters: Any) -> Any:
    """Drop reserved owner keys from a filter dict, also inside AND / OR / NOT groups."""
    if not isinstance(filters, dict):
        return filters
    clean: dict[str, Any] = {}
    for key, value in filters.items():
        if key in _RESERVED_FILTER_KEYS:
            continue
        if key in _LOGICAL_FILTER_KEYS and isinstance(value, list):
            groups = [_strip_reserved_filters(item) for item in value]
            groups = [g for g in groups if g not in ({}, None)]
            if groups:
                clean[key] = groups
            continue
        clean[key] = value
    return clean


def _keys(result: Any) -> list[str]:
    """Top-level keys of a Mem0 response, for logs; never the stored memory text."""
    return sorted(str(k) for k in result) if isinstance(result, dict) else []


class Mem0Store(BaseStore):
    """Mem0 implementation of long-term memory.

    Primary responsibilities:
    * Persist memories (episodic by default) across graph invocations
    * Retrieve semantically similar memories to augment state
    * Provide CRUD lifecycle aligned with ``BaseStore`` async API

    Unlike in-memory state, these memories survive process restarts as they are
    managed by Mem0's configured vector / persistence layer.
    """

    def __init__(
        self,
        config: MemoryConfig | dict,  # type: ignore
        app_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not HAS_MEM0:
            raise ImportError(
                "mem0 package is required for Mem0Store. Install with `pip install mem0ai`."
            )

        super().__init__()
        self.config = config
        self.app_id = app_id or "agentflow_app"
        self._client = None  # Lazy initialization

        logger.info(
            "Initialized Mem0Store (long-term) app=%s",
            self.app_id,
        )

    async def _get_client(self) -> AsyncMemory:  # type: ignore
        """Lazy initialization of AsyncMemory client."""
        if self._client is None:
            try:
                # Prefer explicit config via Memory.from_config when supplied; fallback to defaults
                if isinstance(self.config, dict):
                    self._client = await AsyncMemory.from_config(self.config)  # type: ignore
                elif isinstance(self.config, MemoryConfig):  # type: ignore
                    self._client = AsyncMemory(config=self.config)  # type: ignore
                else:
                    self._client = AsyncMemory()  # type: ignore
            except Exception as e:  # pragma: no cover - defensive
                logger.error("Failed to initialize Mem0 client (%s)", type(e).__name__)
                raise
        return self._client

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _extract_ids(self, config: dict[str, Any]) -> tuple[str, str | None]:
        """Extract user_id and app_id from per-call config with fallbacks."""
        # Mem0 always scopes by a concrete user (it ignores authz scope="none").
        user_id = str(self._require_owner_id(config))
        app_id = config.get("app_id") or self.app_id

        return user_id, app_id

    def _create_result(
        self,
        raw: dict[str, Any],
        user_id: str,
    ) -> MemorySearchResult:
        # check user id belongs to the user
        if raw.get("user_id") != user_id:
            raise ValueError("Memory user_id does not match the requested user_id")

        metadata = raw.get("metadata", {}) or {}
        # Ensure memory_type enum mapping
        memory_type_val = metadata.get("memory_type", MemoryType.EPISODIC.value)
        try:
            memory_type = MemoryType(memory_type_val)
        except ValueError:
            memory_type = MemoryType.EPISODIC

        _raw_id = raw.get("id")
        if _raw_id is None:
            _inj_id = InjectQ.get_instance().try_get("generated_id", None)
            _raw_id = (
                str(_inj_id)
                if _inj_id is not None and not isinstance(_inj_id, Awaitable)
                else secrets.token_hex(16)
            )
        return MemorySearchResult(
            id=metadata.get("memory_id", str(_raw_id)),
            content=raw.get("memory") or raw.get("data", ""),
            score=float(raw.get("score", 0.0) or 0.0),
            memory_type=memory_type,
            metadata=metadata,
            user_id=user_id,
        )

    def _iter_results(self, response: Any) -> Iterable[dict[str, Any]]:
        if isinstance(response, list):
            for item in response:
                if isinstance(item, dict):
                    yield item
        elif isinstance(response, dict) and "results" in response:
            for item in response["results"]:
                if isinstance(item, dict):
                    yield item
        else:  # pragma: no cover
            logger.debug("Unexpected Mem0 response type: %s", type(response))

    # ------------------------------------------------------------------
    # BaseStore required async operations
    # ------------------------------------------------------------------

    async def astore(
        self,
        config: dict[str, Any],
        content: str | Message,
        memory_type: MemoryType = MemoryType.EPISODIC,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        text = content.text() if isinstance(content, Message) else str(content)
        if not text.strip():
            raise ValueError("Content cannot be empty")

        user_id, app_id = self._extract_ids(config)

        # memory_type and category come from the arguments; caller metadata cannot replace them.
        mem_meta = {
            "created_at": datetime.now(UTC).isoformat(),
            **(metadata or {}),
            "memory_type": memory_type.value,
            "category": category,
        }

        infer = kwargs.get("infer", True)

        client = await self._get_client()
        result = await client.add(  # type: ignore
            messages=[{"role": "user", "content": text}],
            user_id=user_id,
            agent_id=app_id,
            metadata=mem_meta,
            infer=infer,
        )

        logger.debug("Stored memory for user=%s app=%s id=%s", user_id, app_id, result)

        return result

    async def asearch(
        self,
        config: dict[str, Any],
        query: str,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        retrieval_strategy=None,  # Unused for Mem0; kept for signature parity
        distance_metric=None,  # Unused
        max_tokens: int = 4000,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        user_id, app_id = self._extract_ids(config)

        # mem0ai 2.x takes the owner inside ``filters``. The scope is applied last, so
        # nothing in the caller's filters can replace it.
        scoped_filters = {
            **(_strip_reserved_filters(filters) or {}),
            "user_id": user_id,
            "agent_id": app_id,
        }
        search_kwargs: dict[str, Any] = {"top_k": limit, "filters": scoped_filters}
        if score_threshold is not None:
            search_kwargs["threshold"] = score_threshold

        client = await self._get_client()
        result = await client.search(query=query, **search_kwargs)  # type: ignore

        if "results" not in result:
            logger.warning("Mem0 search response missing 'results'; keys=%s", _keys(result))
            return []

        if "relations" in result:
            logger.warning(
                "Mem0 search response contains 'relations', which is not supported yet",
            )

        out: list[MemorySearchResult] = limit_to_token_budget(
            [self._create_result(raw, user_id) for raw in result["results"]], max_tokens
        )

        logger.debug(
            "Searched memories for user=%s app=%s query=%s found=%d",
            user_id,
            app_id,
            query,
            len(out),
        )
        return out

    async def aget(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs: Any,
    ) -> MemorySearchResult | None:
        user_id, _ = self._extract_ids(config)
        # If we stored mapping use that user id instead (authoritative)

        client = await self._get_client()
        result = await client.get(  # type: ignore
            memory_id=memory_id,
        )

        return self._create_result(result, user_id) if result else None

    async def aget_all(
        self,
        config: dict[str, Any],
        limit: int = 100,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        user_id, app_id = self._extract_ids(config)

        client = await self._get_client()
        result = await client.get_all(  # type: ignore
            filters={"user_id": user_id, "agent_id": app_id},
            top_k=limit,
        )

        if "results" not in result:
            logger.warning("Mem0 get_all response missing 'results'; keys=%s", _keys(result))
            return []

        if "relations" in result:
            logger.warning(
                "Mem0 get_all response contains 'relations', which is not supported yet",
            )

        out: list[MemorySearchResult] = [
            self._create_result(raw, user_id) for raw in result["results"]
        ]

        logger.debug(
            "Fetched all memories for user=%s app=%s count=%d",
            user_id,
            app_id,
            len(out),
        )
        return out

    async def aupdate(
        self,
        config: dict[str, Any],
        memory_id: str,
        content: str | Message,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        existing = await self.aget(config, memory_id)
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")

        # user_id obtained for potential permission checks (not used by Mem0 update directly)

        new_text = content.text() if isinstance(content, Message) else str(content)
        updated_meta = {**(existing.metadata or {}), **(metadata or {})}
        updated_meta["updated_at"] = datetime.now(UTC).isoformat()

        client = await self._get_client()
        res = await client.update(  # type: ignore
            memory_id=existing.id,
            data=new_text,
        )

        logger.debug("Updated memory %s via recreate", memory_id)
        return res

    async def adelete(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs: Any,
    ) -> Any:
        user_id, _ = self._extract_ids(config)
        existing = await self.aget(config, memory_id)
        if not existing:
            logger.warning("Memory %s not found for deletion", memory_id)
            return {
                "deleted": False,
                "reason": "not_found",
            }

        if existing.user_id != user_id:
            raise ValueError("Cannot delete memory belonging to a different user")

        client = await self._get_client()
        res = await client.delete(  # type: ignore
            memory_id=existing.id,
        )

        logger.debug("Deleted memory %s for user %s", memory_id, user_id)
        return res

    async def aforget_memory(
        self,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        # Mem0's delete_all only takes entity ids (user, agent, run), not memory_type, category
        # or metadata filters. Deleting everything for a narrow request would lose data, so refuse.
        narrowing = [
            name for name in ("memory_type", "category", "filters") if kwargs.get(name) is not None
        ]
        if narrowing:
            raise ValueError(
                f"Mem0Store.aforget_memory cannot narrow by {', '.join(narrowing)}: Mem0 deletes "
                "by user only. Remove the narrowing to delete every memory of the user, or "
                "delete single memories with adelete."
            )
        # Delete all memories for a user
        user_id, _ = self._extract_ids(config)
        client = await self._get_client()
        res = await client.delete_all(user_id=user_id)  # type: ignore
        logger.debug("Forgot all memories for user %s", user_id)
        return res

    async def arelease(self) -> None:
        logger.info("Mem0Store released resources")


# Convenience factory functions


def create_mem0_store(
    config: dict[str, Any],
    user_id: str = "default_user",
    app_id: str = "agentflow_app",
) -> Mem0Store:
    """Factory for a basic Mem0 long-term store."""
    return Mem0Store(
        config=config,
        default_user_id=user_id,
        app_id=app_id,
    )


def create_mem0_store_with_qdrant(
    qdrant_url: str,
    qdrant_api_key: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = "text-embedding-ada-002",
    llm_model: str = "gpt-4o-mini",
    app_id: str = "agentflow_app",
    **kwargs: Any,
) -> Mem0Store:
    """Factory producing a Mem0Store configured for Qdrant backing."""
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "url": qdrant_url,
                "api_key": qdrant_api_key,
                **kwargs.get("vector_store_config", {}),
            },
        },
        "embedder": {
            "provider": kwargs.get("embedder_provider", "openai"),
            "config": {"model": embedding_model, **kwargs.get("embedder_config", {})},
        },
        "llm": {
            "provider": kwargs.get("llm_provider", "openai"),
            "config": {"model": llm_model, **kwargs.get("llm_config", {})},
        },
    }
    return create_mem0_store(
        config=config,
        app_id=app_id,
    )
