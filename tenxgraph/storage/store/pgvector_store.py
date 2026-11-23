"""Postgres + pgvector store for long-term memory and RAG.

``PgVectorStore`` keeps memories in one Postgres table, next to the ``PgCheckpointer`` tables if
you like, and searches them by vector similarity (pgvector), by full-text rank (``tsvector``), or
by both fused with Reciprocal Rank Fusion. It follows the same owner rules as ``QdrantStore``.

Requirements: Postgres 14+, the pgvector extension 0.7.0+ (``CREATE EXTENSION vector``), and the
``pgvector`` extra (``pip install '10xgraph[pgvector]'``). A text-only store (``embedding=None``)
does not need the extension.

Vectors are sent to Postgres as text literals cast to ``vector`` and the store never reads the
``embedding`` column back, so it needs no type codec on the connection. That is what lets it use
a pool you pass in as is, and open connections before the extension exists.
"""

# ruff: noqa: F401

import asyncio
import hashlib
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from tenxgraph.core.state import Message
from tenxgraph.storage._pg_common import (
    DEFAULT_ACQUIRE_TIMEOUT,
    DEFAULT_DDL_LOCK_TIMEOUT,
    DEFAULT_POOL_CONFIG,
    apply_migrations,
    connection_errors,
    loads_jsonb,
    recorded_schema_version,
    retry_on_connection_error,
    run_on_connection,
)
from tenxgraph.storage.columns import (
    ColumnSpec,
    Isolation,
    column_value_errors,
    ddl_lock_timeout_setting,
    ddl_session,
    drop_index,
    ensure_column_indexes,
    ensure_columns,
    ensure_index,
    ensure_rls,
    index_catalog,
    parse_columns,
    quote_identifier,
    resolve_isolation,
    rls_all_rows_settings,
    rls_policy_expression,
    rls_policy_name,
    rows_affected,
    warn_if_rls_bypassed,
)
from tenxgraph.storage.sql_filter import (
    MAX_CONDITIONS,
    FilterError,
    FilterSchema,
    compile_filters,
    validate_filters,
)
from tenxgraph.utils.log_safety import describe_error

from ._pgv_query import _PgvQueryMixin
from ._pgv_reads import _PgvReadsMixin
from ._pgv_schema import _PgvSchemaMixin
from ._pgv_support import (
    _ITERATIVE_SCAN_SETTING,
    _METRIC_SQL,
    _MIGRATIONS,
    _NULLS_NOT_DISTINCT_MIN_SERVER,
    _OPCLASS_TO_METRIC,
    _SELECT_COLUMNS,
    _SUPPORTED_STRATEGIES,
    _SYSTEM_METADATA_KEYS,
    _TABLE_RE,
    _TS_CONFIG_RE,
    CURRENT_SCHEMA_VERSION,
    DEFAULT_TABLE,
    HAS_ASYNCPG,
    HNSW_MAX_DIMENSIONS,
    ITERATIVE_SCAN_MIN_VERSION,
    MIN_PGVECTOR_VERSION,
    STORE_BUILTIN_COLUMNS,
    _loads_jsonb,
    _parse_version,
    _quote_ident,
    _similarity_sql,
    _TenantContext,
    _Where,
    asyncpg,
)
from ._pgv_writes import _PgvWritesMixin
from .base_store import BaseStore, limit_to_token_budget
from .embedding.base_embedding import BaseEmbedding
from .qdrant_store import DEFAULT_COLLECTION
from .store_schema import (
    DistanceMetric,
    MemoryRecord,
    MemorySearchResult,
    MemoryType,
    RetrievalStrategy,
)


logger = logging.getLogger("tenxgraph.store.pgvector")


class PgVectorStore(
    _PgvSchemaMixin,
    _PgvQueryMixin,
    _PgvWritesMixin,
    _PgvReadsMixin,
    BaseStore,
):
    """Long-term memory and RAG in Postgres with pgvector.

    One table holds every collection (``collection`` is a column, part of the primary key).
    Owner rules match ``QdrantStore``: every operation needs a non-empty ``user_id`` (use
    ``system_user_id(name)`` for shared data); reads and deletes are filtered by the caller's
    reach; writes (``astore``, ``aupdate``) always stay reach ``own``, and ``aforget_memory``
    always names an owner.

    Example:
        ```python
        store = PgVectorStore(
            embedding=OpenAIEmbedding(),
            dsn="postgresql://user:pass@localhost:5432/app",
            retrieval_strategy=RetrievalStrategy.HYBRID,
        )
        await store.asetup()
        ```

    Scores are similarities, so ``score_threshold`` means what it means on ``QdrantStore``:
    cosine similarity for ``COSINE`` (the default), the inner product for ``DOT_PRODUCT``,
    ``1 / (1 + distance)`` for ``EUCLIDEAN`` and ``MANHATTAN`` (Qdrant reports the raw distance
    for those). ``TEXT`` scores are ``ts_rank`` normalized to ``[0, 1)``. ``HYBRID`` scores are
    the RRF score scaled so a row ranked first in both lists scores 1.0; in ``HYBRID`` the
    ``score_threshold`` applies to each row's vector similarity, so it keeps its meaning.
    Scores below zero are reported as 0.0.
    """

    #: ``astore`` with a ``memory_key`` is one atomic upsert on the unique key index, so the
    #: memory layer does not need its search-then-update dedup for keyed writes.
    supports_memory_key_upsert = True
    supported_retrieval_strategies = frozenset(_SUPPORTED_STRATEGIES)

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        embedding: BaseEmbedding | None,
        dsn: str | None = None,
        pool: Any | None = None,
        *,
        pool_config: dict[str, Any] | None = None,
        schema: str = "public",
        table: str = DEFAULT_TABLE,
        collection: str | Callable[[dict[str, Any]], str] | None = None,
        distance_metric: DistanceMetric | str = DistanceMetric.COSINE,
        retrieval_strategy: RetrievalStrategy | str | None = None,
        hybrid_weights: tuple[float, float] = (1.0, 1.0),
        rrf_k: int = 60,
        hybrid_candidates: int | None = None,
        text_search_config: str = "english",
        on_embedding_error: Literal["fallback", "raise"] = "fallback",
        acquire_timeout: float | None = DEFAULT_ACQUIRE_TIMEOUT,
        tenant_columns: Any = None,
        extra_columns: Any = None,
        row_level_security: bool = False,
        ddl_lock_timeout: float | None = DEFAULT_DDL_LOCK_TIMEOUT,
    ):
        """Create the store. Nothing touches the database until ``asetup()`` or the first call.

        Args:
            embedding: Embedding service, or ``None`` for a text-only store (``TEXT`` search
                only, no ``embedding`` column, no pgvector extension needed).
            dsn: Postgres connection string. The store creates a pool from it and closes that
                pool in ``arelease()``.
            pool: An existing asyncpg pool. The store never closes it. Pass ``dsn`` or ``pool``.
            pool_config: Extra ``asyncpg.create_pool`` arguments for a pool built from ``dsn``.
            schema: Postgres schema of the table.
            table: Table name (letters, digits, underscore; at most 48 characters).
            collection: Default collection: a name, or a function ``(config) -> name`` that
                receives the config passed to each operation. A ``collection`` in the config
                wins over it; a memory scope's ``collection`` wins over both. A function must
                read the authenticated identity (``config["user"]`` / ``config["user_id"]``),
                never state.
            distance_metric: Distance of the HNSW index and the default for searches. A call
                may pass another metric; it is computed exactly (no index) and logs a warning.
            retrieval_strategy: Default search strategy when a call passes ``None``:
                ``SIMILARITY``, ``TEXT`` or ``HYBRID``. ``None`` means ``SIMILARITY``, or
                ``TEXT`` for a text-only store.
            hybrid_weights: ``(vector_weight, text_weight)`` for RRF in ``HYBRID`` search.
            rrf_k: The RRF constant ``k`` (score of rank ``r`` is ``weight / (k + r)``).
            hybrid_candidates: Rows each list contributes to ``HYBRID`` fusion. ``None`` means
                ``max(4 * limit, 20)``.
            text_search_config: Postgres text search configuration (``english``, ``simple``,
                ...). It is fixed in the table's generated ``tsv`` column when the table is
                created.
            on_embedding_error: ``"fallback"`` (default): when the embedding service fails, a
                write saves the text with no embedding (fill it later with
                ``abackfill_embeddings()``) and a search falls back to ``TEXT``; a warning is
                logged. ``"raise"``: the error propagates.
            acquire_timeout: Seconds to wait for a free pooled connection (``None`` waits).
            tenant_columns: Columns that isolate data, such as
                ``{"org_id": "uuid", "project_id": "text"}`` (or a list of names for ``text``
                columns). Each becomes a real column with a B-tree index. Every write stores
                the caller's values; every read, update, delete, forget and search is filtered
                by the tenant columns the caller's reach picks (all of them when
                there is no identity, see ``tenxgraph.core.authz``). A picked column whose
                value is missing raises, it is never dropped from the filter. Required by
                default; ``{"type": ..., "required": False}`` makes a column optional for
                writes. Tenant columns are scalar ids: ``text``, ``bigint``, ``integer`` or
                ``uuid`` (anything else raises). Shared data (``system_user_id(name)``) is
                shared only inside the tenant values the scope picks. Rows written before a
                column existed hold NULL and are invisible to scopes that pick it until you
                backfill them.
            extra_columns: Data columns stored with each memory, same declaration format.
                Optional by default (missing = NULL); ``"required": True`` makes a write
                without a value raise. ``aupdate`` changes only the extra values present.
                Any Postgres type: ``text``, ``bigint``, ``integer``, ``boolean``, ``uuid``
                and ``timestamptz`` are checked in Python and filterable with the full
                operator table; any other type (``jsonb``, ``text[]``, ``numeric(10,2)``,
                ``date``, an enum) is syntax-checked when declared and resolved on the server
                at setup (unknown types fail setup; only the server's canonical name reaches
                DDL). ``json``/``jsonb`` values are serialised with ``json.dumps``; other
                values go to asyncpg as given, and a value it cannot encode raises
                ``ColumnValueError`` (column and type, never the value). Filters on those
                columns allow only ``is_null``. Example::

                    PgVectorStore(
                        ...,
                        tenant_columns=["org_id"],
                        extra_columns={
                            "public_id": "text",
                            "settings": "jsonb",
                            "tags": "text[]",
                            "seats": {"type": "integer", "required": True},
                        },
                    )
            row_level_security: Also enforce the tenant columns with Postgres native row-level
                security, on top of the WHERE filter: setup runs ``ENABLE`` + ``FORCE ROW
                LEVEL SECURITY`` and creates a policy comparing each tenant column with
                ``current_setting('tenxgraph.<col>', true)``; every call then runs in a
                transaction that sets those values with ``set_config(..., true)`` (``SET
                LOCAL``). Costs one transaction and one extra statement per call. Superusers
                and roles with ``BYPASSRLS`` skip RLS (a warning is logged), and the table
                owner is filtered only because of ``FORCE``. Needs tenant columns.
            ddl_lock_timeout: Seconds a setup DDL statement waits for a lock on the table
                (Postgres ``lock_timeout``) before setup fails with
                ``tenxgraph.storage.columns.SetupLockTimeoutError``; ``None`` waits forever.
                Without it an ``ALTER TABLE`` stuck behind a long write would queue every
                later read of the table behind itself. Default 10 seconds.

        Setup on a table that is in use: setup reads the catalog first and runs no DDL for
        anything already in place, so a restart takes no table lock. Missing indexes on an
        existing table are built with ``CREATE INDEX CONCURRENTLY`` (reads and writes go on);
        an index left invalid by a build that failed or was cancelled is dropped and built
        again on the next setup. Adding a declared column is a short ``ALTER TABLE`` under
        ``ddl_lock_timeout``. Setup is serialised across processes by a session-level
        advisory lock, which holds across those non-transactional statements.

        Rolling out tenant columns on a table that already has data (the ``memory_key``
        unique index gains the tenant columns):

        1. Deploy the new version with ``tenant_columns``. Its setup adds the columns and
           builds the new unique key index ``<table>_kuq_<hash>`` concurrently. The old key
           index ``<table>_key_uq`` is kept, so processes still running the previous version
           keep their keyed writes (their ``ON CONFLICT`` needs it). While both exist, a keyed
           write for the same ``(collection, user_id, memory_type, memory_key)`` in a second
           tenant fails with ``ValueError``; nothing else changes.
        2. When every process runs the new version, backfill the tenant values of old rows
           (they hold NULL and are invisible to scopes that pick those columns).
        3. Call ``await store.adrop_legacy_key_index()`` once. It drops the old index
           concurrently, and only when the new one is valid.

        Trusted-source rule for ``tenant_columns`` and ``extra_columns``: tenant and extra
        column values come only from ``config["user"]``. ``user`` is a reserved config key:
        clients cannot set it. Your auth class fills it (``BaseAuth.authenticate()`` result or
        JWT claims). Return ``org_id``, ``project_id``, ``public_id`` there. Never read them
        from request config, metadata, state or model output. For example::

            class MyAuth(BaseAuth):
                async def authenticate(self, request, response, credential):
                    claims = verify(credential)  # your token check
                    return {
                        "user_id": claims["sub"],
                        "org_id": claims["org_id"],
                        "project_id": claims["project_id"],
                    }
        """
        if not HAS_ASYNCPG:
            raise ImportError(
                "asyncpg is required for PgVectorStore. "
                "Install with `pip install '10xgraph[pgvector]'`."
            )
        if dsn and pool is not None:
            raise ValueError("Pass either dsn or pool to PgVectorStore, not both.")
        if not dsn and pool is None:
            raise ValueError("PgVectorStore needs a dsn or an asyncpg pool.")

        try:
            quote_identifier(schema, "schema name")
        except ValueError:
            raise ValueError(
                f"Invalid schema name {schema!r}: use letters, digits and underscores, "
                "starting with a letter or underscore."
            )
        if not isinstance(table, str) or not _TABLE_RE.match(table):
            raise ValueError(
                f"Invalid table name {table!r}: use letters, digits and underscores, starting "
                "with a letter or underscore, at most 48 characters."
            )
        if not isinstance(text_search_config, str) or not _TS_CONFIG_RE.match(text_search_config):
            raise ValueError(
                f"Invalid text_search_config {text_search_config!r}: use a configuration name "
                "such as 'english' or 'simple'."
            )
        if on_embedding_error not in ("fallback", "raise"):
            raise ValueError("on_embedding_error must be 'fallback' or 'raise'")
        self.columns = parse_columns(
            tenant_columns, extra_columns, reserved=STORE_BUILTIN_COLUMNS, table="PgVectorStore"
        )
        #: Names of the declared tenant columns (read by the authz helpers and compile()).
        self.tenant_column_names: tuple[str, ...] = self.columns.tenant_names
        #: Columns ``filters`` may name with operators (built-ins plus extra columns); pass it
        #: to ``tenxgraph.storage.sql_filter.validate_filters`` to check a request early.
        self.filter_schema = FilterSchema.for_memories(self.columns)
        if not isinstance(row_level_security, bool):
            raise ValueError("row_level_security must be True or False")
        if row_level_security and not self.columns.tenant:
            raise ValueError("row_level_security=True needs at least one tenant column")
        self.row_level_security = row_level_security
        ddl_lock_timeout_setting(ddl_lock_timeout)  # validates
        self.ddl_lock_timeout = ddl_lock_timeout

        self.embedding = embedding
        self.schema = schema
        self.table = table
        self.text_search_config = text_search_config
        self.on_embedding_error = on_embedding_error
        self.acquire_timeout = acquire_timeout
        self.collection: str | Callable[[dict[str, Any]], str] = collection or DEFAULT_COLLECTION

        self.distance_metric = DistanceMetric(distance_metric)
        if retrieval_strategy is None:
            retrieval_strategy = (
                RetrievalStrategy.SIMILARITY if embedding is not None else RetrievalStrategy.TEXT
            )
        self.retrieval_strategy = self._check_strategy(RetrievalStrategy(retrieval_strategy))

        self.hybrid_weights = self._check_weights(hybrid_weights)
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
            raise ValueError("rrf_k must be an integer >= 1")
        self.rrf_k = rrf_k
        if hybrid_candidates is not None and (
            isinstance(hybrid_candidates, bool)
            or not isinstance(hybrid_candidates, int)
            or hybrid_candidates < 1
        ):
            raise ValueError("hybrid_candidates must be None or an integer >= 1")
        self.hybrid_candidates = hybrid_candidates

        self._dsn = dsn
        self._pool_config = dict(pool_config or {})
        self._pool: Any | None = pool
        # Decided here, not when the pool is built (lazily, maybe never): we close only a pool
        # we created from the dsn, never one the caller passed in.
        self._owns_pool = pool is None
        self._pool_lock = asyncio.Lock()
        self._setup_lock = asyncio.Lock()
        self._setup_done = False
        # Metric of the HNSW index found or created by setup (None: no index).
        self._index_metric: DistanceMetric | None = None
        self._warned_metrics: set[DistanceMetric] = set()
        # Whether the server's pgvector supports ``hnsw.iterative_scan`` (found by setup).
        self._iterative_scan = False
        # ``server_version_num`` (read by setup only when tenant columns are declared).
        self._server_version = 0
        # Key indexes from an earlier declaration kept for older processes (found by setup).
        self._legacy_key_indexes: list[str] = []

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _check_strategy(self, strategy: RetrievalStrategy) -> RetrievalStrategy:
        if strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"retrieval_strategy {str(strategy)!r} is not supported by PgVectorStore; "
                "use 'similarity', 'text' or 'hybrid'."
            )
        if self.embedding is None and strategy != RetrievalStrategy.TEXT:
            raise ValueError(
                f"retrieval_strategy {str(strategy)!r} needs an embedding; this PgVectorStore "
                "is text-only (embedding=None), so only 'text' is available."
            )
        return strategy

    @staticmethod
    def _check_weights(weights: Any) -> tuple[float, float]:
        try:
            vector_weight, text_weight = (float(w) for w in weights)
        except (TypeError, ValueError) as e:
            raise ValueError("hybrid_weights must be two numbers (vector, text)") from e
        if not all(math.isfinite(w) and w >= 0 for w in (vector_weight, text_weight)):
            raise ValueError("hybrid_weights must be finite and >= 0")
        if vector_weight + text_weight <= 0:
            raise ValueError("hybrid_weights must not both be 0")
        return vector_weight, text_weight

    @property
    def _qualified_table(self) -> str:
        return f"{_quote_ident(self.schema)}.{_quote_ident(self.table)}"

    @property
    def _version_table(self) -> str:
        return f"{_quote_ident(self.schema)}.{_quote_ident(self.table + '_schema_version')}"

    def _key_columns(self) -> list[str]:
        """Columns of the unique ``memory_key`` index, quoted where they are declared columns.

        Tenant columns are part of the key: the same ``user_id`` in two tenants is two
        different people, so each keeps its own memory for a key. Without tenant columns
        this is ``(collection, user_id, memory_type, memory_key)`` as before.
        """
        return [
            "collection",
            "user_id",
            *(spec.quoted for spec in self.columns.tenant),
            "memory_type",
            "memory_key",
        ]

    def _dimension(self) -> int:
        dimension = getattr(self.embedding, "dimension", None)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError(f"embedding.dimension must be a positive integer, got {dimension!r}")
        return dimension

    def _setup_lock_key(self) -> int:
        """A stable 64-bit advisory-lock key for this table (same in every process)."""
        digest = hashlib.blake2b(
            f"tenxgraph.pgvector:{self.schema}.{self.table}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big", signed=True)

    # ------------------------------------------------------------------
    # Pool and query plumbing
    # ------------------------------------------------------------------

    async def _get_pool(self) -> Any:
        if self._pool is None:
            async with self._pool_lock:
                # Re-check: another coroutine may have built it while we waited.
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(  # type: ignore[union-attr]
                        dsn=self._dsn, **{**DEFAULT_POOL_CONFIG, **self._pool_config}
                    )
        return self._pool

    async def _retry_on_connection_error(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        """Run ``operation``, retrying on a lost connection (3 tries, backoff 1s then 2s).

        Shared with ``PgCheckpointer`` (``storage/_pg_common.py``). Every operation is one
        statement, or one transaction, and writes carry a fixed id, so a retry never writes
        twice.
        """
        return await retry_on_connection_error(operation, owner="PgVectorStore", logger=logger)

    async def _run(
        self,
        fn: Callable[[Any], Awaitable[Any]],
        *,
        setup: bool = True,
        settings: Sequence[tuple[str, str]] | None = None,
    ) -> Any:
        """Acquire a connection, run ``fn(conn)``, retrying on connection errors.

        With ``setup=True`` the first call creates or upgrades the table (see ``asetup``).
        With ``settings`` the call runs in a transaction that first sets each
        ``(name, value)`` with ``set_config(name, value, true)`` (row-level security values,
        ``hnsw.iterative_scan``); they end with the transaction.
        """
        if setup:
            await self._ensure_setup()

        async def _operation() -> Any:
            pool = await self._get_pool()
            return await run_on_connection(
                pool, fn, timeout=self.acquire_timeout, settings=settings
            )

        return await self._retry_on_connection_error(_operation)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aping(self) -> None:
        """Run ``SELECT 1`` on the pool; raise when Postgres cannot be reached.

        Read-only: never creates the table or the extension and never embeds anything.
        """
        await self._run(lambda conn: conn.fetchval("SELECT 1"), setup=False)

    async def arelease(self) -> None:
        """Close the pool if this store created it (from ``dsn``); a passed-in pool stays open."""
        if not self._owns_pool:
            logger.info("PgVectorStore: pool was passed in, leaving it open")
            return
        pool, self._pool = self._pool, None
        if pool is None:
            return
        try:
            if not pool.is_closing():
                await pool.close()
            logger.info("PgVectorStore resources released")
        except Exception as e:
            # Cleanup is best effort, as in PgCheckpointer.
            logger.error("Error closing PgVectorStore pool: %s", describe_error(e))
