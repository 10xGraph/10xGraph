"""Writes: store, update, delete, forget and embedding backfill."""

import json
import logging
from typing import Any

from tenxgraph.core.state import Message
from tenxgraph.storage.columns import (
    ColumnSpec,
    column_value_errors,
    rls_all_rows_settings,
    rows_affected,
)

from ._pgv_host import PgvHostBase
from ._pgv_support import (
    _SYSTEM_METADATA_KEYS,
    asyncpg,
)
from .store_schema import (
    DistanceMetric,
    MemoryType,
)


logger = logging.getLogger("tenxgraph.store.pgvector")


class _PgvWritesMixin(PgvHostBase):
    """Writes: store, update, delete, forget and embedding backfill."""

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def astore(
        self,
        config: dict[str, Any],
        content: str | Message,
        memory_type: MemoryType = MemoryType.EPISODIC,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Store a memory and return its id, in one statement.

        A record always has an owner: an empty ``user_id`` raises, even under
        reach ``all``. Use ``system_user_id(name)`` for shared data.

        Without a ``memory_key`` it is an upsert on ``(collection, id)``: a ``memory_id`` you
        pass that belongs to another user or tenant raises ``ValueError`` (not found, as
        ``aupdate`` and ``adelete``; the update's ``WHERE`` matches no row) and leaves the row
        as it is, on every reach: writes stay reach ``own``. With a ``memory_key``
        (keyword argument or ``metadata["memory_key"]``) it is an atomic upsert on
        ``(collection, user_id, memory_type, memory_key)``: the existing memory with that key is
        updated (metadata merged) and its id returned.

        Keyword Args:
            memory_id: Id for a new memory (generated when absent).
            memory_key: Stable topic key for the memory (see above).
            origin: Who wrote it (``MemoryOrigin``); ``None`` stores unknown.
            expires_at: Timezone-aware time after which the memory is hidden from reads and
                removed by ``adelete_expired``; ``None`` never expires. A naive datetime raises.
                A keyed upsert sets both from this write, so remembering a key again revives
                an expired memory.
        """
        self._refuse_unstorable(
            content=content,
            metadata=metadata,
            category=category,
            memory_key=kwargs.get("memory_key"),
        )
        origin, expires_at = self._provenance(kwargs)
        user_id, thread_id, collection = self._extract_config_values(config)
        owner = str(self._require_owner_id(config, user_id))
        text, memory_type, clean_metadata = self._prepare(content, memory_type, metadata)
        memory_key = self._memory_key(kwargs.get("memory_key"), metadata)
        requested_id = kwargs.get("memory_id")
        memory_id = str(requested_id) if requested_id else await self.generate_framework_id()
        ctx = self._tenant_context(config, write=True)
        check_owner = ctx.owner_scoped
        vector = await self._embed(text, query=False, metric=self.distance_metric)

        params: list[Any] = [
            memory_id,
            collection,
            owner,
            str(thread_id) if thread_id not in (None, "") else None,
            memory_key,
            memory_type.value,
            str(category or "general"),
            text,
            json.dumps(clean_metadata, default=str),
        ]
        columns = [
            "id",
            "collection",
            "user_id",
            "thread_id",
            "memory_key",
            "memory_type",
            "category",
            "content",
            "metadata",
        ]
        values = [f"${i}" for i in range(1, len(params) + 1)]
        values[8] = "$9::text::jsonb"
        if self.embedding is not None:
            params.append(vector)
            columns.append("embedding")
            values.append(f"${len(params)}::text::vector")
        column_params: dict[int, ColumnSpec] = {}
        for spec in self.columns.all:
            params.append(ctx.values.get(spec.name))
            columns.append(spec.quoted)
            values.append(spec.placeholder(f"${len(params)}"))
            column_params[len(params)] = spec
        params.extend([origin.value if origin is not None else None, expires_at])
        columns.extend(["origin", "expires_at"])
        values.extend([f"${len(params) - 1}", f"${len(params)}"])

        insert = (
            f"INSERT INTO {self._qualified_table} AS m ({', '.join(columns)}) "  # noqa: S608
            f"VALUES ({', '.join(values)}) "
        )
        set_embedding = ", embedding = EXCLUDED.embedding" if self.embedding is not None else ""
        set_embedding += ", origin = EXCLUDED.origin, expires_at = EXCLUDED.expires_at"
        if memory_key is not None:
            # The key index includes the tenant columns, so a conflict is always a row of this
            # same owner in this same tenant. Extra values present in this write replace the
            # stored ones; missing ones keep them.
            set_extra = "".join(
                f", {spec.quoted} = COALESCE(EXCLUDED.{spec.quoted}, m.{spec.quoted})"
                for spec in self.columns.extra
            )
            sql = (
                insert + f"ON CONFLICT ({', '.join(self._key_columns())}) "
                "WHERE memory_key IS NOT NULL DO UPDATE SET "
                "content = EXCLUDED.content, category = EXCLUDED.category, "
                f"metadata = m.metadata || EXCLUDED.metadata{set_embedding}{set_extra}, "
                "updated_at = now() RETURNING id"
            )
        else:
            # The record is replaced, so its tenant and extra values become the writer's, like
            # user_id. The WHERE keeps the replace inside the caller's scope: the stored row
            # must match on user_id (when scoped) and on every tenant column the scope picks.
            set_columns = "".join(
                f", {spec.quoted} = EXCLUDED.{spec.quoted}" for spec in self.columns.all
            )
            guards = ["m.user_id = EXCLUDED.user_id"] if check_owner else []
            guards.extend(
                f"m.{spec.quoted} IS NOT DISTINCT FROM EXCLUDED.{spec.quoted}"
                for spec, _ in ctx.conditions
            )
            sql = (
                insert + "ON CONFLICT (collection, id) DO UPDATE SET "
                "user_id = EXCLUDED.user_id, thread_id = EXCLUDED.thread_id, "
                "memory_key = EXCLUDED.memory_key, memory_type = EXCLUDED.memory_type, "
                "category = EXCLUDED.category, content = EXCLUDED.content, "
                f"metadata = EXCLUDED.metadata{set_embedding}{set_columns}, updated_at = now()"
                + (f" WHERE {' AND '.join(guards)}" if guards else "")
                + " RETURNING id"
            )

        async def _write(conn: Any) -> Any:
            try:
                with column_value_errors(column_params):
                    return await conn.fetchrow(sql, *params)
            except asyncpg.UniqueViolationError as e:  # type: ignore[union-attr]
                raise self._store_conflict_error(e, memory_id, memory_key) from e
            except asyncpg.InsufficientPrivilegeError as e:  # type: ignore[union-attr]
                # Under RLS an upsert that hits another tenant's row fails the policy instead
                # of matching no row; report it the same way.
                if self.row_level_security and "row-level security" in str(e):
                    raise ValueError(f"Memory {memory_id} not found") from e
                raise

        row = await self._run(_write, settings=ctx.settings)
        if row is None:
            raise ValueError(f"Memory {memory_id} not found")
        stored_id = str(row["id"])
        logger.debug("Stored memory %s in collection %s", stored_id, collection)
        return stored_id

    def _store_conflict_error(
        self, error: Exception, memory_id: str, memory_key: str | None
    ) -> ValueError:
        """The ``ValueError`` for a unique violation in ``astore``.

        A keyed write that hits the key index of an earlier declaration (kept during a
        rollout, see ``adrop_legacy_key_index``) gets its own message.
        """
        constraint = str(getattr(error, "constraint_name", "") or "")
        if memory_key is not None and self._is_key_index_name(constraint):
            return ValueError(
                f"memory_key '{memory_key}' is taken in another tenant under the key index "
                f"{constraint} of an earlier declaration. Drop it with "
                "adrop_legacy_key_index() once every process runs this version."
            )
        return ValueError(f"memory_id '{memory_id}' is already used by another memory")

    async def aupdate(
        self,
        config: dict[str, Any],
        memory_id: str,
        content: str | Message,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Update a memory's content (re-embedded) and metadata, in one statement.

        Metadata is merged into the stored metadata unless ``replace_metadata=True``. A
        ``category`` or ``memory_key`` in ``metadata`` updates that column; otherwise it is kept.
        A memory that does not exist or belongs to another user raises ``ValueError`` (not
        found), as on ``QdrantStore``. ``origin`` and ``expires_at`` (keyword arguments)
        replace the stored values; left out, the stored ones are kept. An expired memory can
        be updated (for example given a new ``expires_at``).
        """
        self._refuse_unstorable(content=content, metadata=metadata)
        origin, expires_at = self._provenance(kwargs)
        user_id, _thread_id, collection = self._extract_config_values(config)
        scoped_user_id = self._require_owner_id(config, user_id)  # writes stay reach own
        text = content.text() if isinstance(content, Message) else content
        if not isinstance(text, str):
            raise ValueError("memory content must be a string or a Message")
        meta = dict(metadata or {})
        memory_key = self._memory_key(None, meta)
        category = meta.get("category")
        clean = {k: v for k, v in meta.items() if k not in _SYSTEM_METADATA_KEYS}
        replace = bool(kwargs.get("replace_metadata", False))
        ctx = self._tenant_context(config, write=True)
        vector = await self._embed(text, query=False, metric=self.distance_metric)

        params: list[Any] = []
        where = self._scope_where(params, collection, scoped_user_id, tenant=ctx.conditions)
        where.eq("id", str(memory_id))
        content_p = where.param(text)
        category_p = where.param(str(category) if category else None)
        key_p = where.param(memory_key)
        replace_p = where.param(replace)
        meta_p = where.param(json.dumps(clean, default=str))
        set_embedding = ""
        if self.embedding is not None:
            set_embedding = f", embedding = {where.param(vector)}::text::vector"
        # Tenant values never change on update; extra values present in config["user"] do.
        column_params: dict[int, ColumnSpec] = {}
        set_extra = ""
        for spec in self.columns.extra:
            if ctx.values.get(spec.name) is None:
                continue
            placeholder = where.param(ctx.values[spec.name])
            column_params[int(placeholder[1:])] = spec
            set_extra += f", {spec.quoted} = {spec.placeholder(placeholder)}"
        origin_p = where.param(origin.value if origin is not None else None)
        expires_p = where.param(expires_at)
        sql = (
            f"UPDATE {self._qualified_table} SET "  # noqa: S608
            f"content = {content_p}, "
            f"category = COALESCE({category_p}::text, category), "
            f"memory_key = COALESCE({key_p}::text, memory_key), "
            f"metadata = CASE WHEN {replace_p}::bool THEN {meta_p}::text::jsonb "
            f"ELSE metadata || {meta_p}::text::jsonb END{set_embedding}{set_extra}, "
            f"origin = COALESCE({origin_p}::text, origin), "
            f"expires_at = COALESCE({expires_p}::timestamptz, expires_at), "
            f"updated_at = now() WHERE {where.sql()} RETURNING id"
        )

        async def _write(conn: Any) -> Any:
            try:
                with column_value_errors(column_params):
                    return await conn.fetchrow(sql, *params)
            except asyncpg.UniqueViolationError as e:  # type: ignore[union-attr]
                raise ValueError(
                    f"memory_key '{memory_key}' is already used by another memory"
                ) from e

        row = await self._run(_write, settings=ctx.settings)
        if row is None:
            raise ValueError(f"Memory {memory_id} not found")
        logger.debug("Updated memory %s in collection %s", memory_id, collection)

    async def adelete(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs,
    ) -> None:
        """Delete a memory by id; ``ValueError`` when it does not exist or is another user's."""
        user_id, _thread_id, collection = self._extract_config_values(config)
        scoped_user_id = self._require_user_id(config, user_id)
        ctx = self._tenant_context(config)
        params: list[Any] = []
        where = self._scope_where(params, collection, scoped_user_id, tenant=ctx.conditions)
        where.eq("id", str(memory_id))
        sql = f"DELETE FROM {self._qualified_table} WHERE {where.sql()} RETURNING id"  # noqa: S608

        row = await self._run(lambda conn: conn.fetchrow(sql, *params), settings=ctx.settings)
        if row is None:
            raise ValueError(f"Memory {memory_id} not found")
        logger.debug("Deleted memory %s from collection %s", memory_id, collection)

    async def aforget_memory(
        self,
        config: dict[str, Any],
        **kwargs,
    ) -> None:
        """Delete the memories of one user, optionally narrowed (same rules as ``QdrantStore``).

        ``user_id`` is required, also under reach ``all``. ``thread_id`` (from config),
        ``memory_type``, ``category`` and ``filters`` (keyword arguments) narrow the delete, all
        combined with AND. ``filters`` follows the same rules as ``asearch`` (column operators
        through ``tenxgraph.storage.sql_filter``, exact match on metadata keys).
        """
        user_id, thread_id, collection = self._extract_config_values(config)
        owner = self._require_owner_id(config, user_id)
        ctx = self._tenant_context(config)
        params: list[Any] = []
        where = self._scope_where(params, collection, owner, tenant=ctx.conditions)
        if thread_id not in (None, ""):
            where.eq("thread_id", str(thread_id))
        self._add_filters(
            where,
            memory_type=kwargs.get("memory_type"),
            category=kwargs.get("category"),
            filters=kwargs.get("filters"),
        )
        sql = f"DELETE FROM {self._qualified_table} WHERE {where.sql()}"  # noqa: S608
        await self._run(lambda conn: conn.execute(sql, *params), settings=ctx.settings)
        logger.info(
            "Deleted memories for user_id=%s, thread_id=%s in collection %s",
            owner,
            thread_id,
            collection,
        )

    async def adelete_expired(self, batch_size: int = 1000, collection: str | None = None) -> int:
        """Delete every expired memory, over all owners and tenants; return how many.

        Runs one short ``DELETE`` per batch of ``batch_size`` rows (``FOR UPDATE SKIP LOCKED``,
        row locks only, no long transaction), so it is safe while the store is in use and two
        runs at once do not block each other. ``collection`` narrows it to one collection.
        """
        batch_size = self._check_batch_size(batch_size)
        table = self._qualified_table
        params: list[Any] = [batch_size]
        narrow = ""
        if collection is not None:
            params.append(str(collection))
            narrow = " AND collection = $2"
        sql = (
            f"DELETE FROM {table} WHERE (collection, id) IN ("  # noqa: S608
            f"SELECT collection, id FROM {table} WHERE expires_at <= now(){narrow} "
            "ORDER BY expires_at LIMIT $1 FOR UPDATE SKIP LOCKED)"
        )
        # Maintenance over every tenant, as in abackfill_embeddings.
        settings = rls_all_rows_settings(self.columns.tenant) if self.row_level_security else None
        deleted = 0
        while True:
            status = await self._run(lambda conn: conn.execute(sql, *params), settings=settings)
            count = rows_affected(status)
            deleted += count
            if count < batch_size:
                break
        logger.info("PgVectorStore deleted %d expired memories", deleted)
        return deleted

    async def abackfill_embeddings(self, batch_size: int = 100) -> int:
        """Embed every memory saved without an embedding; return how many were filled.

        Walks the table once in primary-key order, ``batch_size`` rows at a time. Each batch is
        embedded with ``aembed_batch`` outside any transaction, then written with one
        ``UPDATE`` that skips rows whose content changed meanwhile. Embedding errors raise
        here (no fallback). Safe to run while the store is in use and to run again.
        """
        if self.embedding is None:
            raise ValueError("abackfill_embeddings needs an embedding; this store is text-only")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be an integer >= 1")
        table = self._qualified_table
        select_sql = (
            f"SELECT collection, id, content FROM {table} "  # noqa: S608
            "WHERE embedding IS NULL AND (collection, id) > ($1, $2) "
            "ORDER BY collection, id LIMIT $3"
        )
        update_sql = (
            f"UPDATE {table} AS m SET embedding = d.embedding::vector, updated_at = now() "  # noqa: S608
            "FROM unnest($1::text[], $2::text[], $3::text[], $4::text[]) "
            "AS d(collection, id, content, embedding) "
            "WHERE m.collection = d.collection AND m.id = d.id "
            "AND m.embedding IS NULL AND m.content = d.content"
        )
        filled = 0
        last_collection, last_id = "", ""
        # Maintenance over every tenant: under RLS the calls set a scope that picks no tenant
        # column, which the policy reads as "all rows".
        settings = rls_all_rows_settings(self.columns.tenant) if self.row_level_security else None
        while True:

            async def _select(conn: Any, c: str = last_collection, i: str = last_id) -> Any:
                return await conn.fetch(select_sql, c, i, batch_size)

            rows = await self._run(_select, settings=settings)
            if not rows:
                break
            texts = [row["content"] for row in rows]
            vectors = await self.embedding.aembed_batch(texts)
            if len(vectors) != len(rows):
                raise ValueError("Embedding service returned a different number of vectors")
            cosine = self.distance_metric == DistanceMetric.COSINE
            literals = [self._vector_literal(v, cosine=cosine) for v in vectors]
            args = (
                [row["collection"] for row in rows],
                [row["id"] for row in rows],
                texts,
                literals,
            )

            async def _update(conn: Any, a: tuple[list[Any], ...] = args) -> Any:
                return await conn.execute(update_sql, *a)

            status = await self._run(_update, settings=settings)
            filled += rows_affected(status)
            last_collection, last_id = rows[-1]["collection"], rows[-1]["id"]
            if len(rows) < batch_size:
                break
        logger.info("PgVectorStore backfilled %d embeddings", filled)
        return filled
