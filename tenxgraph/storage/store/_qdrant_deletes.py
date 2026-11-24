"""Delete methods of ``QdrantStore``: delete by id and forget."""

import logging
from typing import Any

from ._qdrant_reads import QdrantReadsMixin
from .store_schema import MemoryType


logger = logging.getLogger("tenxgraph.store")


class QdrantDeletesMixin(QdrantReadsMixin):
    """``adelete`` and ``aforget_memory``, scoped like the reads."""

    async def adelete(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs,
    ) -> None:
        """Delete a memory by id; ``ValueError`` when it does not exist or is outside the
        caller's scope (another user's on reach ``own``, another tenant's)."""
        from qdrant_client.http import models

        user_id, _thread_id, collection = self._extract_config_values(config)
        self._require_user_id(config, user_id)

        # ``aget`` applies the caller's scope, so a memory outside it is "not found" here.
        existing = await self.aget(config, memory_id)
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")

        await self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=[memory_id]),
        )

        logger.debug(f"Deleted memory {memory_id} from collection {collection}")

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
        value that cannot be matched exactly raises instead of being skipped, because skipping
        it would delete more than the caller asked for.
        """
        from qdrant_client.http import models

        user_id, thread_id, collection = self._extract_config_values(config)
        user_id = self._require_owner_id(config, user_id)
        ctx = self._isolation(config)

        memory_type = kwargs.get("memory_type")
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        category = kwargs.get("category")
        filters = kwargs.get("filters")
        for key, value in (filters or {}).items():
            if not isinstance(value, str | int | bool):
                raise ValueError(
                    f"Unsupported forget filter {key!r}: only str, int and bool values can be "
                    "matched exactly"
                )

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
            points_selector=models.FilterSelector(filter=delete_filter),
        )

        logger.info(
            f"Deleted memories for user_id={user_id}, thread_id={thread_id}, "
            f"memory_type={memory_type}, category={category}, filters={filters} "
            f"in collection {collection}"
        )
