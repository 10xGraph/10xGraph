"""Reads: get, list and search (vector, text and hybrid)."""

import logging
import math
from typing import Any

from ._pgv_host import PgvHostBase
from ._pgv_support import (
    _METRIC_SQL,
    _similarity_sql,
    _Where,
)
from .base_store import limit_to_token_budget
from .store_schema import (
    DistanceMetric,
    MemorySearchResult,
    MemoryType,
    RetrievalStrategy,
)


logger = logging.getLogger("tenxgraph.store.pgvector")


class _PgvReadsMixin(PgvHostBase):
    """Reads: get, list and search (vector, text and hybrid)."""

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def aget(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs,
    ) -> MemorySearchResult | None:
        """Get a memory by id; ``None`` when it does not exist or is another user's."""
        user_id, _thread_id, collection = self._extract_config_values(config)
        scoped_user_id = self._require_user_id(config, user_id)
        ctx = self._tenant_context(config)
        params: list[Any] = []
        where = self._scope_where(params, collection, scoped_user_id, tenant=ctx.conditions)
        where.eq("id", str(memory_id))
        self._add_provenance(where)
        sql = f"SELECT {self._columns()} FROM {self._qualified_table} WHERE {where.sql()}"  # noqa: S608
        row = await self._run(lambda conn: conn.fetchrow(sql, *params), settings=ctx.settings)
        return self._row_to_result(row, 1.0) if row is not None else None

    async def aget_all(
        self,
        config: dict[str, Any],
        limit: int = 100,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """List a user's memories in the collection, newest first.

        ``filters`` and ``origins`` (keyword arguments) narrow the list, same rules as
        ``asearch``. Expired memories are left out.
        """
        user_id, _thread_id, collection = self._extract_config_values(config)
        scoped_user_id = self._require_user_id(config, user_id)
        ctx = self._tenant_context(config)
        params: list[Any] = []
        where = self._scope_where(params, collection, scoped_user_id, tenant=ctx.conditions)
        self._add_filters(where, filters=kwargs.get("filters"))
        self._add_provenance(where, kwargs.get("origins"))
        limit_p = where.param(self._check_limit(limit))
        sql = (
            f"SELECT {self._columns()} FROM {self._qualified_table} "  # noqa: S608
            f"WHERE {where.sql()} ORDER BY updated_at DESC, id LIMIT {limit_p}"
        )
        rows = await self._run(lambda conn: conn.fetch(sql, *params), settings=ctx.settings)
        return [self._row_to_result(row, 1.0) for row in rows]

    @staticmethod
    def _check_limit(limit: Any) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be an integer >= 1")
        return limit

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
        """Search a user's memories across all threads.

        ``retrieval_strategy`` and ``distance_metric`` default to the store's settings.
        ``SIMILARITY`` orders by vector distance, ``TEXT`` by ``ts_rank`` on the generated
        ``tsv`` column (``websearch_to_tsquery`` syntax), ``HYBRID`` fuses both lists with
        Reciprocal Rank Fusion. When the query cannot be embedded and ``on_embedding_error`` is
        ``"fallback"``, ``SIMILARITY`` and ``HYBRID`` run as ``TEXT``. See the class docstring
        for what scores and ``score_threshold`` mean in each strategy. ``filters`` narrows
        every strategy (also both candidate lists of ``HYBRID``); see ``_add_filters``.

        Expired memories are never returned, and ``origins`` (keyword argument, a list of
        ``MemoryOrigin``; ``None`` in it keeps unknown origin) narrows by who wrote them. Both
        are part of the query, so ``limit`` counts only rows that pass them.
        """
        strategy = self._check_strategy(
            RetrievalStrategy(
                retrieval_strategy if retrieval_strategy is not None else self.retrieval_strategy
            )
        )
        metric = DistanceMetric(
            distance_metric if distance_metric is not None else self.distance_metric
        )
        limit = self._check_limit(limit)
        user_id, _thread_id, collection = self._extract_config_values(config)
        scoped_user_id = self._require_user_id(config, user_id)
        ctx = self._tenant_context(config)
        await self._ensure_setup()

        vector = None
        if strategy != RetrievalStrategy.TEXT:
            vector = await self._embed(query, query=True, metric=metric)
            if vector is None:
                strategy = RetrievalStrategy.TEXT
            else:
                self._warn_metric(metric)

        # Every column in the search SQL is qualified with the table alias ``m`` (and the
        # query with its own alias), so a declared column can never collide with a name the
        # SQL introduces (the tsquery, ranks, scores).
        params: list[Any] = []
        where = self._scope_where(
            params, collection, scoped_user_id, alias="m", tenant=ctx.conditions
        )
        self._add_filters(where, memory_type=memory_type, category=category, filters=filters)
        self._add_provenance(where, kwargs.get("origins"))

        if strategy == RetrievalStrategy.TEXT or vector is None:
            sql = self._text_sql(where, query, limit)
        elif strategy == RetrievalStrategy.SIMILARITY:
            sql = self._similarity_sql(where, metric, vector, limit)
        else:
            sql = self._hybrid_sql(where, collection, metric, vector, query, limit)

        settings = self._search_settings(ctx, vector_search=strategy != RetrievalStrategy.TEXT)
        rows = await self._run(lambda conn: conn.fetch(sql, *params), settings=settings)

        results: list[MemorySearchResult] = []
        best = (sum(self.hybrid_weights)) / (self.rrf_k + 1)
        for row in rows:
            # A non-finite score (NaN from a zero vector stored before zero vectors were
            # refused) never passes a threshold: NaN < x is False, so it would otherwise.
            if strategy == RetrievalStrategy.HYBRID:
                similarity = row["similarity"]
                if score_threshold is not None and (
                    similarity is None
                    or not math.isfinite(float(similarity))
                    or float(similarity) < score_threshold
                ):
                    continue
                score = float(row["score"]) / best
            else:
                score = float(row["score"])
                if score_threshold is not None and (
                    not math.isfinite(score) or score < score_threshold
                ):
                    continue
            results.append(self._row_to_result(row, score))

        results = limit_to_token_budget(results, max_tokens)
        logger.debug("Found %d memories (%s) in collection %s", len(results), strategy, collection)
        return results

    def _similarity_sql(
        self, where: _Where, metric: DistanceMetric, vector: str, limit: int
    ) -> str:
        """``SIMILARITY`` search. ``where`` is built with the table alias ``m``."""
        operator, _opclass = _METRIC_SQL[metric]
        vector_p = where.param(vector)
        distance = f"m.embedding {operator} {vector_p}::text::vector"
        where.add("m.embedding IS NOT NULL")
        limit_p = where.param(limit)
        return (
            f"SELECT {self._columns('m')}, {_similarity_sql(metric, distance)} AS score "  # noqa: S608
            f"FROM {self._qualified_table} AS m WHERE {where.sql()} "
            f"ORDER BY {distance} LIMIT {limit_p}"
        )

    def _text_sql(self, where: _Where, query: str, limit: int) -> str:
        """``TEXT`` search. ``where`` is built with the table alias ``m``.

        The tsquery is the column ``q`` of the function alias ``tsq_input``; every reference
        is qualified, so a declared column named ``tsq`` or ``q`` cannot shadow it.
        """
        config_p = where.param(self.text_search_config)
        query_p = where.param(query)
        rank = "ts_rank(m.tsv, tsq_input.q, 32)"
        where.add("m.tsv @@ tsq_input.q")
        limit_p = where.param(limit)
        return (
            f"SELECT {self._columns('m')}, {rank} AS score "  # noqa: S608
            f"FROM {self._qualified_table} AS m, "
            f"websearch_to_tsquery({config_p}::text::regconfig, {query_p}) AS tsq_input(q) "
            f"WHERE {where.sql()} ORDER BY {rank} DESC, m.id LIMIT {limit_p}"
        )

    def _hybrid_sql(
        self,
        where: _Where,
        collection: str,
        metric: DistanceMetric,
        vector: str,
        query: str,
        limit: int,
    ) -> str:
        operator, _opclass = _METRIC_SQL[metric]
        scope = where.sql()
        vector_p = where.param(vector)
        config_p = where.param(self.text_search_config)
        query_p = where.param(query)
        candidates_p = where.param(self.hybrid_candidates or max(4 * limit, 20))
        vector_weight_p = where.param(self.hybrid_weights[0])
        text_weight_p = where.param(self.hybrid_weights[1])
        k_p = where.param(self.rrf_k)
        collection_p = where.param(collection)
        limit_p = where.param(limit)
        table = self._qualified_table
        # ``scope`` is built with the table alias ``m``; each CTE aliases the table as ``m``
        # and the tsquery as ``tsq_input(q)``, so every reference is qualified (see
        # ``_text_sql``).
        m_distance = f"m.embedding {operator} {vector_p}::text::vector"
        rank = "ts_rank(m.tsv, tsq_input.q, 32)"
        return (
            "WITH vec AS ("  # noqa: S608
            f"SELECT m.id, ROW_NUMBER() OVER (ORDER BY {m_distance}) AS rnk FROM {table} AS m "
            f"WHERE {scope} AND m.embedding IS NOT NULL ORDER BY {m_distance} "
            f"LIMIT {candidates_p}"
            "), txt AS ("
            f"SELECT m.id, ROW_NUMBER() OVER (ORDER BY {rank} DESC) AS rnk "
            f"FROM {table} AS m, "
            f"websearch_to_tsquery({config_p}::text::regconfig, {query_p}) AS tsq_input(q) "
            f"WHERE {scope} AND m.tsv @@ tsq_input.q "
            f"ORDER BY {rank} DESC LIMIT {candidates_p}"
            "), fused AS ("
            "SELECT COALESCE(vec.id, txt.id) AS id, "
            f"COALESCE({vector_weight_p}::float8 / ({k_p}::int + vec.rnk), 0) + "
            f"COALESCE({text_weight_p}::float8 / ({k_p}::int + txt.rnk), 0) AS rrf "
            "FROM vec FULL OUTER JOIN txt ON vec.id = txt.id"
            ") "
            f"SELECT {self._columns('m')}, f.rrf AS score, "
            f"{_similarity_sql(metric, m_distance)} AS similarity "
            f"FROM fused f JOIN {table} m ON m.collection = {collection_p} AND m.id = f.id "
            f"ORDER BY f.rrf DESC, m.id LIMIT {limit_p}"
        )
