"""Schema: table, indexes, extension, migrations and setup."""

import hashlib
import logging
import re
from typing import Any

from tenxgraph.storage._pg_common import (
    apply_migrations,
    connection_errors,
    recorded_schema_version,
)
from tenxgraph.storage.columns import (
    ddl_session,
    drop_index,
    ensure_column_indexes,
    ensure_columns,
    ensure_index,
    ensure_rls,
    index_catalog,
    rls_policy_expression,
    rls_policy_name,
    warn_if_rls_bypassed,
)

from ._pgv_support import (
    _METRIC_SQL,
    _MIGRATIONS,
    _NULLS_NOT_DISTINCT_MIN_SERVER,
    _OPCLASS_TO_METRIC,
    CURRENT_SCHEMA_VERSION,
    HNSW_MAX_DIMENSIONS,
    ITERATIVE_SCAN_MIN_VERSION,
    MIN_PGVECTOR_VERSION,
    _parse_version,
    asyncpg,
)


logger = logging.getLogger("tenxgraph.store.pgvector")


class _PgvSchemaMixin:
    """Schema: table, indexes, extension, migrations and setup."""

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

        Deprecated: this method and the legacy key index handling (keeping the pre-tenant
        ``<table>_key_uq`` index next to the new one) are transitional. They will be removed
        in the next breaking release (1.5 or 2.0), after which setup replaces the old index
        directly.

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
            except connection_errors():
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
        table = self._qualified_table
        await apply_migrations(
            conn,
            version_table=self._version_table,
            target=CURRENT_SCHEMA_VERSION,
            migrations={v: [s.format(table=table) for s in ss] for v, ss in _MIGRATIONS.items()},
            owner="PgVectorStore",
            logger=logger,
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
            return await recorded_schema_version(conn, self._version_table)

        return await self._run(_read, setup=False)
