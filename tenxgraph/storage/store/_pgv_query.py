"""Query helpers: scoping, filters, vector literals, embedding and row conversion."""

import json
import logging
import math
from collections.abc import Sequence
from typing import Any

from tenxgraph.core.state import Message
from tenxgraph.storage.columns import (
    ColumnSpec,
    resolve_isolation,
)
from tenxgraph.storage.sql_filter import (
    MAX_CONDITIONS,
    FilterError,
    compile_filters,
    validate_filters,
)

from ._pgv_host import PgvHostBase
from ._pgv_support import (
    _ITERATIVE_SCAN_SETTING,
    _SELECT_COLUMNS,
    _SYSTEM_METADATA_KEYS,
    _loads_jsonb,
    _quote_ident,
    _TenantContext,
    _Where,
)
from .store_schema import (
    DistanceMetric,
    MemoryRecord,
    MemorySearchResult,
    MemoryType,
)


logger = logging.getLogger("tenxgraph.store.pgvector")


class _PgvQueryMixin(PgvHostBase):
    """Query helpers: scoping, filters, vector literals, embedding and row conversion."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tenant_context(self, config: dict[str, Any], *, write: bool = False) -> _TenantContext:
        """Resolve the declared columns' values and the tenant isolation for one call.

        The rule is shared with PgCheckpointer: ``tenxgraph.storage.columns.resolve_isolation``.
        ``write=True`` pins reach ``own`` (writes never widen, A13). Reach ``tenant`` raises
        when the store declares no tenant columns.
        """
        return resolve_isolation(
            self.columns, config, write=write, row_level_security=self.row_level_security
        )

    def _search_settings(self, ctx: _TenantContext, vector_search: bool) -> list[tuple[str, str]]:
        """Settings for a search: RLS values plus ``hnsw.iterative_scan`` for vector scans."""
        settings = list(ctx.settings or [])
        if vector_search and self._iterative_scan and self._index_metric is not None:
            settings.append(_ITERATIVE_SCAN_SETTING)
        return settings

    def _scope_where(
        self,
        params: list[Any],
        collection: str,
        user_id: Any,
        alias: str | None = None,
        tenant: Sequence[tuple[ColumnSpec, Any]] = (),
    ) -> _Where:
        """The isolation part of every WHERE clause: collection, owner (when scoped), then the
        tenant columns the scope picks."""
        where = _Where(params, alias)
        where.eq("collection", collection)
        if user_id is not None:
            where.eq("user_id", str(user_id))
        for spec, value in tenant:
            where.eq(spec.name, value)
        return where

    def _add_filters(  # noqa: PLR0912
        self,
        where: _Where,
        memory_type: MemoryType | str | None = None,
        category: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> None:
        """Narrow ``where`` by memory type, category and ``filters`` (AND only).

        ``filters`` keys that name a filterable column (``memory_type``, ``category``,
        ``thread_id``, ``memory_key``, ``created_at``, ``updated_at`` and the declared extra
        columns) go through ``tenxgraph.storage.sql_filter``: a plain value means equality, an
        object such as ``{"gte": 5}`` applies operators (see that module for the table). Tenant
        columns and ``user_id`` are rejected. Any other key matches a top-level JSONB metadata
        key exactly (``metadata @> {"key": value}``, the key travels inside the parameter,
        never as SQL) and takes only ``str``, ``int``, ``float`` or ``bool`` values. At most
        ``MAX_CONDITIONS`` conditions in all. Problems raise ``FilterError`` (a ``ValueError``).
        """
        if memory_type:
            where.eq("memory_type", MemoryType(memory_type).value)
        if category:
            where.eq("category", str(category))
        if not filters:
            return
        if not isinstance(filters, dict):
            raise FilterError("filters must be an object mapping column names to conditions")
        if len(filters) > MAX_CONDITIONS:
            raise FilterError(f"Too many filter conditions: at most {MAX_CONDITIONS}")
        schema = self.filter_schema
        column_filters: dict[str, Any] = {}
        metadata_filters: list[tuple[str, Any]] = []
        for key, value in filters.items():
            if not isinstance(key, str) or not key:
                raise FilterError("Unsupported filter key: keys must be non-empty strings")
            if schema.get(key) is not None or schema.is_isolation(key):
                column_filters[key] = value
                continue
            if value is None or not isinstance(value, str | int | float | bool):
                raise FilterError(
                    "Unsupported filter on a metadata key: PgVectorStore matches metadata "
                    "exactly and takes only str, int, float or bool values"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise FilterError("Unsupported filter on a metadata key: value must be finite")
            metadata_filters.append((key, value))
        conditions = validate_filters(column_filters, schema)
        if len(conditions) + len(metadata_filters) > MAX_CONDITIONS:
            raise FilterError(f"Too many filter conditions: at most {MAX_CONDITIONS}")
        column_sql = compile_filters(conditions, where.params, where.alias)
        if column_sql:
            where.add(column_sql)
        for key, value in metadata_filters:
            where.add(
                f"{where.column('metadata')} @> {where.param(json.dumps({key: value}))}"
                "::text::jsonb"
            )

    def _vector_literal(self, vector: Any, *, cosine: bool = False) -> str:
        """Validate an embedding and render it as a pgvector text literal ``[x,y,...]``.

        With ``cosine=True`` a zero vector raises ``ValueError``: its cosine distance to
        anything is undefined (pgvector returns NaN), which would sort and score arbitrarily.
        """
        dimension = self._dimension()
        if vector is None or len(vector) != dimension:
            raise ValueError("Embedding service returned invalid vector")
        values = []
        for item in vector:
            number = float(item)
            if not math.isfinite(number):
                raise ValueError("Embedding service returned invalid vector")
            values.append(repr(number))
        if cosine and not any(float(item) != 0.0 for item in vector):
            raise ValueError(
                "Embedding service returned a zero vector: cosine distance is undefined for it"
            )
        return "[" + ",".join(values) + "]"

    async def _embed(
        self, text: str, *, query: bool, metric: DistanceMetric | None = None
    ) -> str | None:
        """Embed ``text`` and return a vector literal, or ``None`` after a fallback.

        An embedding service error is caught under ``on_embedding_error="fallback"`` (a warning
        is logged). A vector of the wrong size always raises: it is a configuration error, not
        an outage. Under ``metric=COSINE`` a zero vector is refused: a write raises
        ``ValueError``; a query raises under ``"raise"`` and falls back to ``TEXT`` under
        ``"fallback"``.
        """
        if self.embedding is None:
            return None
        if query:
            embed = getattr(self.embedding, "aembed_query", None) or self.embedding.aembed
        else:
            embed = self.embedding.aembed
        try:
            vector = await embed(text)
        except Exception as e:
            if self.on_embedding_error == "raise":
                raise
            logger.warning(
                "PgVectorStore: embedding failed (%s: %s); %s",
                type(e).__name__,
                e,
                "searching by text instead" if query else "saving the memory without an embedding",
            )
            return None
        cosine = metric == DistanceMetric.COSINE
        if query and cosine and self.on_embedding_error == "fallback":
            try:
                return self._vector_literal(vector, cosine=True)
            except ValueError as e:
                if "zero vector" not in str(e):
                    raise
                logger.warning(
                    "PgVectorStore: the query embedding is a zero vector, which has no cosine "
                    "distance; searching by text instead"
                )
                return None
        return self._vector_literal(vector, cosine=cosine)

    @staticmethod
    def _prepare(
        content: str | Message,
        memory_type: MemoryType | str,
        metadata: dict[str, Any] | None,
    ) -> tuple[str, MemoryType, dict[str, Any]]:
        """Return ``(text, memory_type, metadata)`` for a write, as ``QdrantStore`` builds them."""
        if isinstance(content, Message):
            record = MemoryRecord.from_message(content, additional_metadata=metadata)
            text, memory_type, meta = record.content, record.memory_type, record.metadata
        else:
            text, meta = content, dict(metadata or {})
            memory_type = MemoryType(memory_type)
        if not isinstance(text, str):
            raise ValueError("memory content must be a string or a Message")
        clean = {k: v for k, v in meta.items() if k not in _SYSTEM_METADATA_KEYS}
        return text, memory_type, clean

    @staticmethod
    def _memory_key(explicit: Any, metadata: dict[str, Any] | None) -> str | None:
        key = explicit if explicit is not None else (metadata or {}).get("memory_key")
        if key is None or key == "":
            return None
        if not isinstance(key, str) or not key.strip():
            raise ValueError("memory_key must be a non-empty string")
        return key

    @staticmethod
    def _row_to_result(row: Any, score: float) -> MemorySearchResult:
        score = float(score or 0.0)
        try:
            memory_type = MemoryType(row["memory_type"])
        except ValueError:
            memory_type = MemoryType.EPISODIC
        metadata = _loads_jsonb(row["metadata"])
        metadata["category"] = row["category"]
        metadata["memory_type"] = memory_type.value
        if row["memory_key"] is not None:
            metadata["memory_key"] = row["memory_key"]
        return MemorySearchResult(
            id=str(row["id"]),
            content=row["content"] or "",
            score=max(0.0, score) if math.isfinite(score) else 0.0,
            memory_type=memory_type,
            metadata=metadata,
            vector=None,
            user_id=row["user_id"],
            thread_id=row["thread_id"],
            timestamp=row["updated_at"],
        )

    def _columns(self, alias: str | None = None) -> str:
        prefix = f"{alias}." if alias else ""
        return ", ".join(f"{prefix}{_quote_ident(c)}" for c in _SELECT_COLUMNS)

    def _warn_metric(self, metric: DistanceMetric) -> None:
        if metric == self._index_metric or metric in self._warned_metrics:
            return
        self._warned_metrics.add(metric)
        logger.warning(
            "PgVectorStore: searching with distance %s, but the vector index uses %s; this "
            "search runs without the index (exact scan).",
            metric,
            self._index_metric or "no index",
        )
