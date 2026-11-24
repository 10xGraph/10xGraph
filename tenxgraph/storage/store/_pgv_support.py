"""Shared imports, constants and small helpers of ``PgVectorStore``.

``pgvector_store.py`` re-exports everything here, so existing imports keep working.
"""

import logging
import re
from typing import Any

from tenxgraph.storage._pg_common import (
    loads_jsonb,
)
from tenxgraph.storage.columns import quote_identifier

from .store_schema import (
    DistanceMetric,
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

# DEFAULT_ACQUIRE_TIMEOUT, DEFAULT_DDL_LOCK_TIMEOUT and DEFAULT_POOL_CONFIG are shared with
# PgCheckpointer (storage/_pg_common.py) and stay importable from here.

# Table names leave room for the index and version-table suffixes inside Postgres' 63-byte
# identifier limit (longer names would be truncated and could collide).
_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,47}$")

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
    """Decode the ``metadata`` JSONB column; ``{}`` for NULL or anything but an object."""
    value = loads_jsonb(value)
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
        quoted = quote_identifier(name)
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
