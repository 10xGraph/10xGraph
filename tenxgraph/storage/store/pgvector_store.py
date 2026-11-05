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

import asyncio
import hashlib
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from tenxgraph.core.state import Message
from tenxgraph.storage.columns import (
    ColumnSpec,
    apply_settings,
    column_value_errors,
    ddl_lock_timeout_setting,
    ddl_session,
    drop_index,
    ensure_column_indexes,
    ensure_columns,
    ensure_index,
    ensure_rls,
    index_catalog,
    isolation_conditions,
    parse_columns,
    rls_all_rows_settings,
    rls_policy_expression,
    rls_policy_name,
    rls_settings,
    warn_if_rls_bypassed,
    write_conditions,
)
from tenxgraph.storage.sql_filter import (
    MAX_CONDITIONS,
    FilterError,
    FilterSchema,
    compile_filters,
    validate_filters,
)

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


HAS_ASYNCPG = False

try:
    import asyncpg

    HAS_ASYNCPG = True
except ImportError:  # pragma: no cover - exercised only without the extra
    asyncpg = None  # type: ignore[assignment]


logger = logging.getLogger("tenxgraph.store.pgvector")

# Schema version of the store table. Bump it and add an entry to ``_MIGRATIONS`` when the table
# changes, so existing databases are upgraded in place by ``asetup()``.
CURRENT_SCHEMA_VERSION = 2

# Version -> idempotent DDL that brings the table from ``version - 1`` to ``version``. Each
# statement is a format string with ``{table}`` (the quoted, schema-qualified table name).
# Version 1 is the table created by ``_build_create_sql``.
#
# Version 2 adds the developer's tenant and extra columns. Those depend on the store's
# declaration, not on the version, so they are added by ``_ensure_declared_columns`` on every
# setup (``ADD COLUMN IF NOT EXISTS`` + one B-tree index each, only for a column that is
# missing), which also covers a column declared after the table reached version 2. With
# nothing declared, version 2 changes nothing in the table.
_MIGRATIONS: dict[int, list[str]] = {2: []}

# Columns the store owns. Tenant and extra columns cannot reuse these names.
STORE_BUILTIN_COLUMNS = frozenset(
    {
        "id",
        "collection",
        "user_id",
        "thread_id",
        "memory_key",
        "memory_type",
        "category",
        "content",
        "tsv",
        "embedding",
        "metadata",
        "created_at",
        "updated_at",
    }
)

# pgvector 0.8.0 added iterative index scans: with a WHERE filter, an HNSW scan keeps going
# until ``LIMIT`` rows pass the filter instead of stopping after ``hnsw.ef_search`` candidates.
ITERATIVE_SCAN_MIN_VERSION = (0, 8, 0)
# ``strict_order`` keeps results in exact distance order (``relaxed_order`` may not), which the
# ``ORDER BY ... LIMIT`` of every vector search relies on.
_ITERATIVE_SCAN_SETTING = ("hnsw.iterative_scan", "strict_order")
# Postgres 15 added NULLS NOT DISTINCT for unique indexes.
_NULLS_NOT_DISTINCT_MIN_SERVER = 150000

DEFAULT_TABLE = "tenxgraph_memories"
MIN_PGVECTOR_VERSION = (0, 7, 0)
# pgvector cannot build an HNSW index on a ``vector`` column with more dimensions than this.
HNSW_MAX_DIMENSIONS = 2000

DEFAULT_ACQUIRE_TIMEOUT = 30.0  # seconds to wait for a free pooled connection
#: Seconds setup DDL waits for a table lock before it fails with ``SetupLockTimeoutError``.
DEFAULT_DDL_LOCK_TIMEOUT = 10.0
DEFAULT_POOL_CONFIG: dict[str, Any] = {
    "command_timeout": 60.0,  # per query
    "timeout": 30.0,  # per new connection
}

# Table names leave room for the index and version-table suffixes inside Postgres' 63-byte
# identifier limit (longer names would be truncated and could collide).
_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,47}$")
_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_TS_CONFIG_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

_SUPPORTED_STRATEGIES = (
    RetrievalStrategy.SIMILARITY,
    RetrievalStrategy.TEXT,
    RetrievalStrategy.HYBRID,
)

# Distance metric -> (pgvector distance operator, HNSW operator class).
_METRIC_SQL: dict[DistanceMetric, tuple[str, str]] = {
    DistanceMetric.COSINE: ("<=>", "vector_cosine_ops"),
    DistanceMetric.EUCLIDEAN: ("<->", "vector_l2_ops"),
    DistanceMetric.DOT_PRODUCT: ("<#>", "vector_ip_ops"),
    DistanceMetric.MANHATTAN: ("<+>", "vector_l1_ops"),
}
_OPCLASS_TO_METRIC = {opclass: metric for metric, (_op, opclass) in _METRIC_SQL.items()}

# Keys the store keeps in real columns. They are taken out of caller metadata before it is saved
# as JSONB, so metadata can never replace them.
_SYSTEM_METADATA_KEYS = frozenset(
    {"content", "user_id", "thread_id", "memory_type", "category", "timestamp", "memory_key"}
)

# ``filters`` keys that match a real column (built-ins, plus declared extra columns) are
# validated and compiled by ``tenxgraph.storage.sql_filter`` (``MEMORY_FILTER_COLUMNS``).

_SELECT_COLUMNS = (
    "id",
    "collection",
    "user_id",
    "thread_id",
    "memory_key",
    "memory_type",
    "category",
    "content",
    "metadata",
    "created_at",
    "updated_at",
)


def _quote_ident(name: str) -> str:
    """Return ``name`` as a quoted SQL identifier, after checking it against a strict pattern."""
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return f'"{name}"'


def _parse_version(text: str) -> tuple[int, ...]:
    """Turn an extension version such as ``"0.8.0"`` into ``(0, 8, 0)``."""
    parts = re.findall(r"\d+", str(text))[:3]
    return tuple(int(p) for p in parts) + (0,) * (3 - len(parts))


def _similarity_sql(metric: DistanceMetric, distance_sql: str) -> str:
    """SQL that turns a pgvector distance into a similarity (higher is more similar).

    Cosine: ``1 - distance`` (the cosine similarity Qdrant reports). Dot product: the inner
    product (``<#>`` returns its negative). Euclidean and Manhattan: ``1 / (1 + distance)``.
    """
    if metric == DistanceMetric.COSINE:
        return f"(1 - ({distance_sql}))"
    if metric == DistanceMetric.DOT_PRODUCT:
        return f"(({distance_sql}) * -1)"
    return f"(1 / (1 + ({distance_sql})))"


def _loads_jsonb(value: Any) -> dict[str, Any]:
    """Decode a JSONB column that a connection may return as text, bytes or an object."""
    if value is None:
        return {}
    if isinstance(value, bytes | bytearray):
        value = value.decode()
    if isinstance(value, str):
        value = json.loads(value)
    return dict(value) if isinstance(value, dict) else {}


class _Where:
    """AND-joined WHERE conditions. Columns are quoted identifiers; values are always params.

    Every read, update and delete builds its WHERE clause through one instance, so later
    isolation columns (tenant columns) are added in a single place.
    """

    def __init__(self, params: list[Any], alias: str | None = None):
        self.params = params
        self.alias = alias
        self.clauses: list[str] = []

    def column(self, name: str) -> str:
        quoted = _quote_ident(name)
        return f"{self.alias}.{quoted}" if self.alias else quoted

    def param(self, value: Any) -> str:
        self.params.append(value)
        return f"${len(self.params)}"

    def eq(self, column: str, value: Any) -> None:
        self.clauses.append(f"{self.column(column)} = {self.param(value)}")

    def add(self, clause: str) -> None:
        self.clauses.append(clause)

    def sql(self) -> str:
        return " AND ".join(self.clauses) if self.clauses else "TRUE"


@dataclass
class _TenantContext:
    """Tenant and extra column values for one call, and the isolation they imply.

    ``values`` holds every declared column's value from ``config["user"]`` (``None`` when
    missing); ``conditions`` the tenant columns the scope picks with their values; ``settings``
    the RLS ``set_config`` pairs (``None`` when RLS is off).
    """

    values: dict[str, Any] = field(default_factory=dict)
    conditions: list[tuple[ColumnSpec, Any]] = field(default_factory=list)
    settings: list[tuple[str, str]] | None = None


class PgVectorStore(BaseStore):
    """Long-term memory and RAG in Postgres with pgvector.

    One table holds every collection (``collection`` is a column, part of the primary key).
    Owner rules match ``QdrantStore``: every operation needs a non-empty ``user_id`` (use
    ``system_user_id(name)`` for shared data); reads, updates and deletes are filtered by it
    unless the authz policy is ``scope="none"``; writes and ``aforget_memory`` always name an
    owner.

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
                by the tenant columns the request's isolation scope picks (all of them when
                there is no authz policy, see ``tenxgraph.core.authz``). A picked column whose
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
                async def authenticate(self, request, credential):
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

        if not isinstance(schema, str) or not _SCHEMA_RE.match(schema):
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

    @staticmethod
    def _connection_errors() -> tuple[type[BaseException], ...]:
        errors: list[type[BaseException]] = [ConnectionError]
        if asyncpg is not None:
            errors.extend([asyncpg.PostgresConnectionError, asyncpg.InterfaceError])
        return tuple(errors)

    async def _retry_on_connection_error(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        """Run ``operation``, retrying on a lost connection (3 tries, backoff 1s then 2s).

        Same policy as ``PgCheckpointer``. Every operation is one statement, or one
        transaction, and writes carry a fixed id, so a retry never writes twice.
        """
        errors = self._connection_errors()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await operation()
            except errors as e:
                if attempt == max_retries - 1:
                    logger.error("PgVectorStore failed after %d attempts: %s", max_retries, e)
                    raise
                wait = 2**attempt
                logger.warning(
                    "PgVectorStore connection error (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1,
                    max_retries,
                    wait,
                    e,
                )
                await asyncio.sleep(wait)
        return None  # pragma: no cover - the loop always returns or raises

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
            async with pool.acquire(timeout=self.acquire_timeout) as conn:
                if settings:
                    async with conn.transaction():
                        await apply_settings(conn, settings)
                        return await fn(conn)
                return await fn(conn)

        return await self._retry_on_connection_error(_operation)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_create_sql(self) -> list[str]:
        """``CREATE TABLE`` for the memory table. Indexes: ``_ensure_indexes``."""
        table = self._qualified_table
        embedding_column = (
            f"embedding vector({self._dimension()}),\n" if self.embedding is not None else ""
        )
        # The text search configuration is checked against _TS_CONFIG_RE, so it cannot close
        # the string literal. A generated column needs a constant configuration.
        ts_config = f"'{self.text_search_config}'::regconfig"
        return [
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT NOT NULL,
                collection TEXT NOT NULL,
                user_id TEXT NOT NULL,
                thread_id TEXT,
                memory_key TEXT,
                memory_type TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                content TEXT NOT NULL,
                tsv tsvector GENERATED ALWAYS AS (to_tsvector({ts_config}, content)) STORED,
                {embedding_column}metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (collection, id)
            )
            """,
        ]

    def _build_version_table_sql(self) -> str:
        return f"""
            CREATE TABLE IF NOT EXISTS {self._version_table} (
                version INT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """

    def _base_indexes(self) -> list[tuple[str, str]]:
        """``(name, definition)`` of the plain indexes every table has."""
        return [
            (f"{self.table}_uid_idx", "(collection, user_id)"),
            (f"{self.table}_tsv_idx", "USING gin (tsv)"),
        ]

    def _key_nulls_clause(self) -> str:
        """``NULLS NOT DISTINCT`` for the key index when tenant columns may hold NULL.

        An optional tenant column (``required=False``) can be NULL; with the default "NULLs are
        distinct" two keyed writes by the same owner would then both insert. Postgres 15+
        treats those NULLs as equal; on Postgres 14 the clause is left out and such rows do
        not dedupe by ``memory_key`` (required tenant columns are never NULL on write).
        """
        if self.columns.tenant and self._server_version >= _NULLS_NOT_DISTINCT_MIN_SERVER:
            return " NULLS NOT DISTINCT"
        return ""

    def _key_index_definition(self) -> str:
        """The part after ``ON <table>`` of the ``memory_key`` unique index."""
        return (
            f"({', '.join(self._key_columns())}){self._key_nulls_clause()} "
            "WHERE memory_key IS NOT NULL"
        )

    def _key_index_names(self) -> tuple[str, str]:
        """``(plain, hashed)`` names for the key index of this declaration.

        Without tenant columns the index keeps its historical name ``<table>_key_uq``. With
        tenant columns it is ``<table>_kuq_<hash of its definition>``, so a new definition
        never reuses the name of the index older processes still rely on (see
        ``adrop_legacy_key_index``). The hashed name is also the fallback when the plain name
        is taken by an index with another definition.
        """
        digest = hashlib.blake2b(self._key_index_definition().encode(), digest_size=4).hexdigest()
        hashed = f"{self.table}_kuq_{digest}"
        plain = f"{self.table}_key_uq" if not self.columns.tenant else hashed
        return plain, hashed

    def _is_key_index_name(self, name: str) -> bool:
        return name == f"{self.table}_key_uq" or name.startswith(f"{self.table}_kuq_")

    async def _key_index_rows(self, conn: Any) -> list[Any]:
        """The unique partial indexes on the table that this store manages as key indexes."""
        rows = await conn.fetch(
            "SELECT c.relname, i.indisvalid, i.indexrelid::regclass::text AS qualified, "
            "pg_get_indexdef(i.indexrelid) AS definition, "
            "ARRAY(SELECT a.attname::text "
            "FROM unnest(i.indkey::int2[]) WITH ORDINALITY AS k(attnum, ord) "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum "
            "ORDER BY k.ord) AS columns "
            "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE i.indrelid = to_regclass($1) AND i.indisunique AND i.indpred IS NOT NULL",
            self._qualified_table,
        )
        return [row for row in rows or [] if self._is_key_index_name(str(row["relname"]))]

    def _matches_key_definition(self, row: Any) -> bool:
        wanted = [c.strip('"') for c in self._key_columns()]
        nulls = "NULLS NOT DISTINCT" in str(row["definition"] or "")
        return list(row["columns"] or []) == wanted and nulls == bool(self._key_nulls_clause())

    async def _ensure_key_index(
        self, conn: Any, catalog: dict[str, tuple[bool, str]], *, concurrently: bool
    ) -> None:
        """Make sure a valid ``memory_key`` unique index matches the store's declaration.

        An index of either name with the wanted columns (and ``NULLS NOT DISTINCT`` flag)
        satisfies it, so nothing runs on a restart. Otherwise the index is built under the
        name from ``_key_index_names``, ``CONCURRENTLY`` on an existing table, and an invalid
        leftover of that name is dropped and rebuilt. Key indexes of an earlier declaration
        are kept (older processes still need them for their ``ON CONFLICT``) and dropped by
        ``adrop_legacy_key_index()``. Adding tenant columns to the key only relaxes it, so
        the build cannot fail on existing rows; removing one can, and then setup raises.
        """
        rows = await self._key_index_rows(conn)
        current = next(
            (row for row in rows if row["indisvalid"] and self._matches_key_definition(row)),
            None,
        )
        if current is None:
            plain, hashed = self._key_index_names()
            taken = next((row for row in rows if str(row["relname"]) == plain), None)
            name = hashed if taken is not None and taken["indisvalid"] else plain
            logger.info(
                "PgVectorStore: building %s on %s.%s%s",
                name,
                self.schema,
                self.table,
                " concurrently" if concurrently else "",
            )
            try:
                await ensure_index(
                    conn,
                    catalog,
                    name=name,
                    table_sql=self._qualified_table,
                    definition=self._key_index_definition(),
                    concurrently=concurrently,
                    unique=True,
                )
            except asyncpg.UniqueViolationError as e:  # type: ignore[union-attr]
                if concurrently:  # a failed concurrent build leaves an invalid index behind
                    leftover = (await index_catalog(conn, self._qualified_table)).get(name)
                    if leftover is not None and not leftover[0]:
                        await drop_index(conn, leftover[1], concurrently=True)
                raise ValueError(
                    f"Cannot build the memory_key index {name} on {self.schema}.{self.table}: "
                    "existing rows share a key under the new declaration (a tenant column was "
                    "removed?). Deduplicate them or keep the column."
                ) from e
        else:
            name = str(current["relname"])
        self._legacy_key_indexes = sorted(
            str(row["relname"]) for row in rows if row["indisvalid"] and str(row["relname"]) != name
        )
        if self._legacy_key_indexes:
            logger.warning(
                "PgVectorStore: keeping key index(es) %s of an earlier declaration on %s.%s, "
                "so processes still running the previous version keep their keyed writes. "
                "Once every process runs this version, call adrop_legacy_key_index().",
                ", ".join(self._legacy_key_indexes),
                self.schema,
                self.table,
            )

    async def adrop_legacy_key_index(self) -> list[str]:
        """Drop the ``memory_key`` indexes of an earlier declaration; return their names.

        Call it once every process runs the current declaration (see the rollout steps in
        the constructor docstring). Each index is dropped with ``DROP INDEX CONCURRENTLY``
        under the setup lock and ``ddl_lock_timeout``, and only after checking that a valid
        index for the current declaration exists, so keyed writes never lose their
        ``ON CONFLICT`` target. Returns ``[]`` when there is nothing to drop.

        Raises:
            RuntimeError: No valid key index matches the current declaration (run
                ``asetup()`` first).
        """
        await self._ensure_setup()

        async def _drop(conn: Any) -> list[str]:
            async with ddl_session(
                conn,
                lock_key=self._setup_lock_key(),
                timeout=self.ddl_lock_timeout,
                owner="PgVectorStore",
            ):
                rows = await self._key_index_rows(conn)
                current = [
                    row for row in rows if row["indisvalid"] and self._matches_key_definition(row)
                ]
                if not current:
                    raise RuntimeError(
                        f"No valid memory_key index matches the declaration of "
                        f"{self.schema}.{self.table}; run asetup() first."
                    )
                keep = str(current[0]["relname"])
                dropped = []
                for row in rows:
                    if str(row["relname"]) == keep:
                        continue
                    await drop_index(conn, str(row["qualified"]), concurrently=True)
                    dropped.append(str(row["relname"]))
                return dropped

        dropped = await self._run(_drop, setup=False)
        self._legacy_key_indexes = []
        if dropped:
            logger.info("PgVectorStore: dropped legacy key index(es) %s", ", ".join(dropped))
        return dropped

    async def _ensure_declared_columns(self, conn: Any) -> None:
        """Add the declared tenant and extra columns and the RLS policy (idempotent).

        Runs inside the setup transaction; the columns' indexes are built by
        ``_ensure_indexes``. With nothing declared this does nothing at all (the table stays
        exactly as before). To turn RLS off, keep the tenant columns and set
        ``row_level_security=False``: the next setup drops the policy and disables RLS on the
        table.
        """
        if not self.columns:
            return
        await ensure_columns(
            conn, self._qualified_table, self.columns, f"{self.table}_c", indexes=False
        )
        await ensure_rls(
            conn,
            table_sql=self._qualified_table,
            schema=self.schema,
            table=self.table,
            enabled=self.row_level_security,
            policy=rls_policy_name(self.columns.tenant) if self.row_level_security else None,
            expression=(
                rls_policy_expression(self.columns.tenant) if self.row_level_security else None
            ),
        )
        if self.row_level_security:
            await warn_if_rls_bypassed(conn, logger, "PgVectorStore")

    async def _ensure_indexes(self, conn: Any, *, concurrently: bool) -> None:
        """Build every index the table should have that is missing or invalid.

        Reads the catalog once and skips what is in place, so a restart runs no DDL (a
        ``CREATE INDEX IF NOT EXISTS`` would still wait for a SHARE lock behind open writes).
        ``concurrently=True`` (an existing table) builds with ``CREATE INDEX CONCURRENTLY``
        on a connection with no open transaction; a new table gets plain builds inside the
        setup transaction, which nobody else can see yet.
        """
        table = self._qualified_table
        catalog = await index_catalog(conn, table)
        for name, definition in self._base_indexes():
            await ensure_index(
                conn,
                catalog,
                name=name,
                table_sql=table,
                definition=definition,
                concurrently=concurrently,
            )
        await self._ensure_vector_index(conn, catalog, concurrently=concurrently)
        await self._ensure_key_index(conn, catalog, concurrently=concurrently)
        if self.columns:
            await ensure_column_indexes(
                conn,
                table,
                self.columns,
                f"{self.table}_c",
                concurrently=concurrently,
                catalog=catalog,
            )

    async def _ensure_extension(self, conn: Any) -> None:
        """Create the pgvector extension if needed and check its version."""
        version = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        if version is None:
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except self._connection_errors():
                raise
            except Exception as e:
                raise RuntimeError(
                    "PgVectorStore could not create the pgvector extension ('vector'): "
                    f"{e}. Install pgvector >= 0.7.0 on the server and have a database admin "
                    "run `CREATE EXTENSION vector;`, or use PgVectorStore(embedding=None) for "
                    "a text-only store."
                ) from e
            version = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
        if version is None or _parse_version(version) < MIN_PGVECTOR_VERSION:
            needed = ".".join(str(p) for p in MIN_PGVECTOR_VERSION)
            raise RuntimeError(
                f"PgVectorStore needs pgvector >= {needed}; the database has {version}. "
                "Upgrade it with `ALTER EXTENSION vector UPDATE;`."
            )
        self._iterative_scan = _parse_version(version) >= ITERATIVE_SCAN_MIN_VERSION

    async def _check_embedding_column(self, conn: Any) -> None:
        """Raise when the table's embedding column does not match this store's embedding."""
        if self.embedding is None:
            return
        column_type = await conn.fetchval(
            "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
            "WHERE a.attrelid = to_regclass($1) AND a.attname = 'embedding' "
            "AND NOT a.attisdropped",
            self._qualified_table,
        )
        if column_type is None:
            raise ValueError(
                f"Table {self.schema}.{self.table} has no embedding column: it was created by "
                "a text-only PgVectorStore (embedding=None). Use another table for an "
                "embedding store."
            )
        match = re.fullmatch(r"vector\((\d+)\)", str(column_type))
        dimension = self._dimension()
        if not match or int(match.group(1)) != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: table {self.schema}.{self.table} stores "
                f"{column_type}, the embedding service returns vector({dimension}). Use "
                "another table, or migrate the existing one."
            )

    async def _ensure_vector_index(
        self, conn: Any, catalog: dict[str, tuple[bool, str]], *, concurrently: bool
    ) -> None:
        """Create the HNSW index, or read the metric of the one that exists."""
        if self.embedding is None:
            self._index_metric = None
            return
        if self._dimension() > HNSW_MAX_DIMENSIONS:
            logger.warning(
                "PgVectorStore: %d dimensions is above pgvector's HNSW limit of %d; "
                "similarity search runs without an index (exact scan).",
                self._dimension(),
                HNSW_MAX_DIMENSIONS,
            )
            self._index_metric = None
            return
        index_name = f"{self.table}_emb_idx"
        existing = await conn.fetchval(
            "SELECT opc.opcname FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indexrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_opclass opc ON opc.oid = i.indclass[0] "
            "WHERE n.nspname = $1 AND c.relname = $2",
            self.schema,
            index_name,
        )
        valid = catalog.get(index_name, (True, ""))[0]
        if existing is None or not valid:
            _op, opclass = _METRIC_SQL[self.distance_metric]
            await ensure_index(
                conn,
                catalog,
                name=index_name,
                table_sql=self._qualified_table,
                definition=f"USING hnsw (embedding {opclass})",
                concurrently=concurrently,
            )
            self._index_metric = self.distance_metric
            return
        self._index_metric = _OPCLASS_TO_METRIC.get(str(existing))
        if self._index_metric != self.distance_metric:
            logger.warning(
                "PgVectorStore: the HNSW index on %s.%s uses %s, the store is set to %s. "
                "Searches with %s run without the index. Drop the index %s to rebuild it.",
                self.schema,
                self.table,
                existing,
                self.distance_metric,
                self.distance_metric,
                index_name,
            )

    async def _apply_migrations(self, conn: Any) -> None:
        recorded = await conn.fetchval(f"SELECT max(version) FROM {self._version_table}")  # noqa: S608
        current = int(recorded or 0)
        if current >= CURRENT_SCHEMA_VERSION:
            return
        for version in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
            for statement in _MIGRATIONS.get(version, []):
                await conn.execute(statement.format(table=self._qualified_table))
        await conn.execute(
            f"INSERT INTO {self._version_table} (version) VALUES ($1) "  # noqa: S608
            "ON CONFLICT (version) DO NOTHING",
            CURRENT_SCHEMA_VERSION,
        )

    async def _setup_schema(self, conn: Any) -> bool:
        """Extension, table, version table, migrations, declared columns and RLS policy.

        Runs inside the setup transaction. Every step reads the catalog first and runs DDL
        only for what is missing. Returns whether the table was created by this call.
        """
        if self.columns.tenant:
            self._server_version = int(
                await conn.fetchval("SELECT current_setting('server_version_num')::int") or 0
            )
        if self.embedding is not None:
            await self._ensure_extension(conn)
        exists = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", self._qualified_table)
        if not exists:
            for statement in self._build_create_sql():
                await conn.execute(statement)
        if not await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", self._version_table):
            await conn.execute(self._build_version_table_sql())
        await self._check_embedding_column(conn)
        await self._apply_migrations(conn)
        await self._ensure_declared_columns(conn)
        return not exists

    async def _ensure_setup(self) -> None:
        if self._setup_done:
            return
        async with self._setup_lock:
            if self._setup_done:
                return

            async def _setup(conn: Any) -> None:
                # Concurrent setups (several workers starting at once) run one after another
                # under a session-level advisory lock, which also holds across the
                # non-transactional CREATE INDEX CONCURRENTLY below; every DDL statement waits
                # at most ddl_lock_timeout for a table lock.
                async with ddl_session(
                    conn,
                    lock_key=self._setup_lock_key(),
                    timeout=self.ddl_lock_timeout,
                    owner="PgVectorStore",
                ):
                    # Schema changes in one transaction, so a cancel rolls back cleanly.
                    async with conn.transaction():
                        created = await self._setup_schema(conn)
                        if created:
                            await self._ensure_indexes(conn, concurrently=False)
                    if not created:
                        # Missing indexes of a table in use are built without blocking it.
                        await self._ensure_indexes(conn, concurrently=True)

            await self._run(_setup, setup=False)
            self._setup_done = True
            logger.info("PgVectorStore ready on %s.%s", self.schema, self.table)

    async def asetup(self) -> Any:
        """Create the extension (when an embedding is set), the table and its indexes.

        Safe to call from many processes at once (Postgres advisory lock) and many coroutines
        (asyncio lock). Upgrades an older table to ``CURRENT_SCHEMA_VERSION``. Runs no DDL for
        anything already in place; missing indexes of an existing table are built
        concurrently (see the constructor docstring). Raises a clear ``RuntimeError`` when the
        pgvector extension cannot be created, ``ValueError`` when the table's embedding
        dimension differs from the embedding service's, and ``SetupLockTimeoutError`` when a
        DDL statement waits longer than ``ddl_lock_timeout`` for a table lock.
        """
        await self._ensure_setup()
        return True

    async def aget_schema_version(self) -> int:
        """The schema version recorded in the database; 0 when the table is missing."""

        async def _read(conn: Any) -> int:
            exists = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", self._version_table)
            if not exists:
                return 0
            recorded = await conn.fetchval(f"SELECT max(version) FROM {self._version_table}")  # noqa: S608
            return int(recorded or 0)

        return await self._run(_read, setup=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_config_values(self, config: dict[str, Any]) -> tuple[Any, Any, str]:
        """Return ``(user_id, thread_id, collection)`` from the config (same rules as Qdrant)."""
        user_id = config.get("user_id")
        thread_id = config.get("thread_id")
        collection = config.get("collection")
        if collection is None:
            collection = self.collection
            if callable(collection):
                collection = collection(config)
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError(f"collection must be a non-empty string, got {collection!r}")
        return user_id, thread_id, collection

    def _tenant_context(
        self, config: dict[str, Any], *, write: bool = False, strict: bool = True
    ) -> _TenantContext:
        """Resolve the declared columns' values and the tenant isolation for one call.

        Values come only from ``config["user"]``. The scope is the request's authz scope
        (``tenxgraph.core.authz.isolation_columns``); with no policy every tenant column
        applies. ``write=True`` raises when a required column has no value; any call raises
        when a column the scope picks has no value, except with ``strict=False`` (``astore``
        only), where a missing optional value is matched with ``IS NOT DISTINCT FROM``. A
        scope naming an undeclared column raises even when the store declares no columns.
        """
        picked = self._isolation_columns(config)
        if not self.columns:
            return _TenantContext()
        values = self.columns.resolve(config, write=write)
        specs = self.columns.scope_specs(picked)
        conditions = (
            isolation_conditions(specs, values) if strict else write_conditions(specs, values)
        )
        settings = (
            rls_settings(self.columns.tenant, specs, values) if self.row_level_security else None
        )
        return _TenantContext(values=values, conditions=conditions, settings=settings)

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
        ``scope="none"``. Use ``system_user_id(name)`` for shared data.

        Without a ``memory_key`` it is an upsert on ``(collection, id)``: a ``memory_id`` you
        pass that belongs to another user raises ``PermissionError`` (the update's ``WHERE``
        matches no row), unless isolation is off (``scope="none"``). With a ``memory_key``
        (keyword argument or ``metadata["memory_key"]``) it is an atomic upsert on
        ``(collection, user_id, memory_type, memory_key)``: the existing memory with that key is
        updated (metadata merged) and its id returned.

        Keyword Args:
            memory_id: Id for a new memory (generated when absent).
            memory_key: Stable topic key for the memory (see above).
        """
        user_id, thread_id, collection = self._extract_config_values(config)
        owner = str(self._require_owner_id(config, user_id))
        text, memory_type, clean_metadata = self._prepare(content, memory_type, metadata)
        memory_key = self._memory_key(kwargs.get("memory_key"), metadata)
        requested_id = kwargs.get("memory_id")
        memory_id = str(requested_id) if requested_id else await self.generate_framework_id()
        check_owner = self._scope_user_id(config, owner) is not None
        ctx = self._tenant_context(config, write=True, strict=False)
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

        insert = (
            f"INSERT INTO {self._qualified_table} AS m ({', '.join(columns)}) "  # noqa: S608
            f"VALUES ({', '.join(values)}) "
        )
        set_embedding = ", embedding = EXCLUDED.embedding" if self.embedding is not None else ""
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
                    raise PermissionError(f"memory_id '{memory_id}' belongs to another user") from e
                raise

        row = await self._run(_write, settings=ctx.settings)
        if row is None:
            raise PermissionError(f"memory_id '{memory_id}' belongs to another user")
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
        found), as on ``QdrantStore``.
        """
        user_id, _thread_id, collection = self._extract_config_values(config)
        scoped_user_id = self._require_user_id(config, user_id)
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
        sql = (
            f"UPDATE {self._qualified_table} SET "  # noqa: S608
            f"content = {content_p}, "
            f"category = COALESCE({category_p}::text, category), "
            f"memory_key = COALESCE({key_p}::text, memory_key), "
            f"metadata = CASE WHEN {replace_p}::bool THEN {meta_p}::text::jsonb "
            f"ELSE metadata || {meta_p}::text::jsonb END{set_embedding}{set_extra}, "
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

        ``user_id`` is required, also under ``scope="none"``. ``thread_id`` (from config),
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
            filled += self._rows_affected(status)
            last_collection, last_id = rows[-1]["collection"], rows[-1]["id"]
            if len(rows) < batch_size:
                break
        logger.info("PgVectorStore backfilled %d embeddings", filled)
        return filled

    @staticmethod
    def _rows_affected(status: Any) -> int:
        """Rows affected from an asyncpg command tag such as ``"UPDATE 3"``."""
        try:
            return int(str(status).rsplit(" ", 1)[-1])
        except (ValueError, IndexError):
            return 0

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

        ``filters`` (keyword argument) narrows the list, same rules as ``asearch``.
        """
        user_id, _thread_id, collection = self._extract_config_values(config)
        scoped_user_id = self._require_user_id(config, user_id)
        ctx = self._tenant_context(config)
        params: list[Any] = []
        where = self._scope_where(params, collection, scoped_user_id, tenant=ctx.conditions)
        self._add_filters(where, filters=kwargs.get("filters"))
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
            logger.error("Error closing PgVectorStore pool: %s", e)
