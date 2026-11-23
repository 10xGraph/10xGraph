"""Schema: DDL, migrations, indexes and setup of the Postgres tables."""

import logging
from typing import Any

from tenxgraph.storage._pg_common import (
    apply_migrations,
    recorded_schema_version,
    run_on_connection,
)
from tenxgraph.storage.columns import (
    column_ddl,
    ddl_session,
    ensure_column_indexes,
    ensure_columns,
    ensure_index,
    ensure_rls,
    index_catalog,
    rls_policy_expression,
    rls_policy_name,
    warn_if_rls_bypassed,
)
from tenxgraph.utils.log_safety import describe_error

from ._pg_host import PgHostBase
from ._pg_support import (
    _THREAD_CHILD_TABLES,
    CURRENT_SCHEMA_VERSION,
    ID_TYPE_MAP,
    SCHEMA_INIT_LOCK_ID,
)


logger = logging.getLogger("tenxgraph.checkpointer.pg")


class _PgSchemaMixin(PgHostBase):
    """Schema: DDL, migrations, indexes and setup of the Postgres tables."""

    def _get_sql_type(self, type_name: str) -> str:
        """
        Get SQL type for given configuration type.

        Args:
            type_name (str): Type name ('string', 'int', 'bigint').

        Returns:
            str: Corresponding SQL type.
        """
        return ID_TYPE_MAP.get(type_name, "VARCHAR(255)")

    def _get_current_schema_version(self) -> int:
        """Return current expected schema version."""
        return CURRENT_SCHEMA_VERSION

    def _build_create_tables_sql(self) -> list[str]:
        """
        Build SQL statements for table creation with dynamic ID types.

        Returns:
            list[str]: List of SQL statements for table creation.
        """
        tables = self._build_table_statements()
        created = {table for table, _sql in tables}
        statements = [sql for _table, sql in tables]
        # ``tool_executions`` is made by migration v3, so its index is not part of this list.
        statements.extend(
            f"CREATE INDEX IF NOT EXISTS {name} ON {self._get_table_name(table)}{definition}"
            for name, table, definition in self._base_index_specs()
            if table in created
        )
        return statements

    @staticmethod
    def _base_index_specs() -> list[tuple[str, str, str]]:
        """``(index name, table, definition)`` of the plain indexes of a fresh schema.

        Version-dependent indexes on ``states`` are created by the schema migrations (see
        _build_migrations) so that an upgrade from a pre-version schema adds the column before
        indexing it.
        """
        return [
            ("idx_threads_user_id", "threads", "(user_id)"),
            ("idx_states_thread_id", "states", "(thread_id)"),
            ("idx_messages_thread_id", "messages", "(thread_id)"),
            # v7: aprune_tool_results deletes by age across all threads.
            ("idx_tool_executions_created_at", "tool_executions", "(created_at)"),
        ]

    def _build_table_statements(self) -> list[tuple[str | None, str]]:
        """``(table, sql)`` creating each table (``None`` for the enum type block)."""
        thread_id_type = self._get_sql_type(self.id_type)
        user_id_type = self._get_sql_type(self.user_id_type)
        message_id_type = self._get_sql_type(self.id_type)

        # For AUTO INCREMENT types, we need to handle primary key differently
        thread_pk = (
            "thread_id SERIAL PRIMARY KEY"
            if self.id_type == "int"
            else f"thread_id {thread_id_type} PRIMARY KEY"
        )
        # Message ids are unique per thread, not globally (see migration v4), so the key
        # is declared as a table constraint on (thread_id, message_id).
        message_col = (
            "message_id SERIAL" if self.id_type == "int" else f"message_id {message_id_type}"
        )

        return [
            # Schema version tracking table
            (
                "schema_version",
                f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name("schema_version")} (
                version INT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            ),
            # Create message role enum (safe for older Postgres versions)
            (
                None,
                (
                    "DO $$\n"
                    "BEGIN\n"
                    "    CREATE TYPE message_role AS ENUM "
                    "('user', 'assistant', 'system', 'tool');\n"
                    "EXCEPTION\n"
                    "    WHEN duplicate_object THEN NULL;\n"
                    "END$$;"
                ),
            ),
            # Create threads table
            (
                "threads",
                f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name("threads")} (
                {thread_pk},
                thread_name VARCHAR(255),
                user_id {user_id_type} NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                meta JSONB DEFAULT '{{}}'::jsonb
            )
            """,
            ),
            # Create states table.
            # ``version`` is a per-thread monotonically increasing counter used
            # for optimistic concurrency control (see aput_state/aput_checkpoint).
            # The UNIQUE (thread_id, version) constraint is what makes concurrent
            # appends collide instead of silently duplicating a version.
            (
                "states",
                f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name("states")} (
                state_id SERIAL PRIMARY KEY,
                thread_id {thread_id_type} NOT NULL
                    REFERENCES {self._get_table_name("threads")}(thread_id)
                    ON DELETE CASCADE,
                version BIGINT NOT NULL DEFAULT 0,
                state_data JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                meta JSONB DEFAULT '{{}}'::jsonb
            )
            """,
            ),
            # Create messages table
            (
                "messages",
                f"""
            CREATE TABLE IF NOT EXISTS {self._get_table_name("messages")} (
                {message_col} NOT NULL,
                thread_id {thread_id_type} NOT NULL
                    REFERENCES {self._get_table_name("threads")}(thread_id)
                    ON DELETE CASCADE,
                role message_role NOT NULL,
                content TEXT NOT NULL,
                tool_calls JSONB,
                tool_call_id VARCHAR(255),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                total_tokens INT DEFAULT 0,
                usages JSONB DEFAULT '{{}}'::jsonb,
                meta JSONB DEFAULT '{{}}'::jsonb,
                PRIMARY KEY (thread_id, message_id)
            )
            """,
            ),
        ]

    def _build_migrations(self) -> dict[int, list[str]]:
        """Return ordered DDL steps keyed by the schema version they produce.

        Every statement must be idempotent (``IF NOT EXISTS`` / guarded), so a
        migration can run safely both on a freshly created schema (where the
        target objects already exist and each step is a no-op) and on an older
        database being upgraded in place. Migrations are applied in ascending
        version order inside a single transaction (see
        ``_check_and_apply_schema_version``).
        """
        states = self._get_table_name("states")
        messages = self._get_table_name("messages")
        return {
            # v2: per-thread state versioning for optimistic concurrency control.
            2: [
                f"ALTER TABLE {states} ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 0",
                # Backfill deterministic per-thread versions for any pre-existing
                # rows so the unique index below can be created. Ordered by
                # created_at then state_id to match the "latest wins" read order.
                f"""
                UPDATE {states} AS s
                SET version = sub.rn
                FROM (
                    SELECT state_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY thread_id
                               ORDER BY created_at, state_id
                           ) AS rn
                    FROM {states}
                ) AS sub
                WHERE s.state_id = sub.state_id
                  AND s.version = 0
                """,  # noqa: S608
                f"CREATE INDEX IF NOT EXISTS idx_states_thread_version ON "
                f"{states}(thread_id, version DESC)",
                f"CREATE UNIQUE INDEX IF NOT EXISTS uq_states_thread_version ON "
                f"{states}(thread_id, version)",
            ],
            # v3: durable tool-execution ledger, so a node replayed after a crash
            # does not re-fire tool calls that already completed (audit B2).
            # The (thread_id, tool_call_id) primary key IS the idempotency key.
            3: [
                f"""
                CREATE TABLE IF NOT EXISTS {self._get_table_name("tool_executions")} (
                    thread_id {self._get_sql_type(self.id_type)} NOT NULL
                        REFERENCES {self._get_table_name("threads")}(thread_id)
                        ON DELETE CASCADE,
                    tool_call_id VARCHAR(255) NOT NULL,
                    result JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (thread_id, tool_call_id)
                )
                """,
            ],
            # v4: message ids are unique per thread. The old key was message_id alone,
            # so one user could claim an id another thread later needed and make that
            # thread's checkpoint fail. Guarded: a schema created at v4 or later
            # already has the two-column key, so the block does nothing there.
            4: [
                f"""
                DO $$
                DECLARE pk_name text;
                BEGIN
                    SELECT conname INTO pk_name FROM pg_constraint
                    WHERE conrelid = '{messages}'::regclass
                      AND contype = 'p'
                      AND array_length(conkey, 1) = 1;
                    IF pk_name IS NOT NULL THEN
                        EXECUTE format(
                            'ALTER TABLE %s DROP CONSTRAINT %I',
                            '{messages}'::regclass, pk_name
                        );
                        EXECUTE format(
                            'ALTER TABLE %s ADD PRIMARY KEY (thread_id, message_id)',
                            '{messages}'::regclass
                        );
                    END IF;
                END$$;
                """,  # noqa: S608
            ],
            # v5: reasoning lives only in the ReasoningBlocks inside ``content``, so
            # the separate text column is gone. A fresh schema never has it.
            5: [
                f"ALTER TABLE {messages} DROP COLUMN IF EXISTS reasoning",
            ],
            # v6: the developer's tenant and extra columns on ``threads`` (ADD COLUMN IF NOT
            # EXISTS). Nothing declared: no change. A column declared later is added by
            # _ensure_declared_columns, which runs on every setup. Extra columns of other
            # types (jsonb, text[], ...) are added there too, after the server resolves the
            # declared type: their text never reaches DDL. Their B-tree indexes are built by
            # _ensure_indexes, concurrently on an existing table.
            6: column_ddl(
                self._get_table_name("threads"),
                self.columns.builtin_only(),
                "idx_threads",
                indexes=False,
            ),
            # v7: an index on tool_executions(created_at) for aprune_tool_results. No DDL
            # here: _ensure_indexes builds it after the migrations, CONCURRENTLY when the
            # table already exists, so the upgrade does not block ledger writes.
            7: [],
        }

    async def _ensure_declared_columns(self, conn) -> None:
        """Add declared columns that are missing and manage the native RLS policies.

        Runs on every schema init, under the init advisory lock, after the migrations. Reads
        the catalog first, so nothing is altered when everything is in place. With nothing
        declared it does nothing. To turn RLS off, keep the tenant columns and set
        ``row_level_security=False``: the next setup drops the policies and disables RLS.
        The columns' indexes are built by ``_ensure_indexes``.
        """
        if not self.columns:
            return
        threads = self._get_table_name("threads")
        await ensure_columns(conn, threads, self.columns, "idx_threads", indexes=False)
        tenant = self.columns.tenant
        enabled = self.row_level_security
        await ensure_rls(
            conn,
            table_sql=threads,
            schema=self.schema,
            table="threads",
            enabled=enabled,
            policy=rls_policy_name(tenant) if enabled else None,
            expression=rls_policy_expression(tenant) if enabled else None,
        )
        for child in _THREAD_CHILD_TABLES:
            table_sql = self._get_table_name(child)
            # The subquery on threads is itself filtered by the threads policy, so a child
            # row is visible exactly when its thread is.
            expression = (
                f"EXISTS (SELECT 1 FROM {threads} AS t "  # noqa: S608
                f"WHERE t.thread_id = {table_sql}.thread_id)"
            )
            await ensure_rls(
                conn,
                table_sql=table_sql,
                schema=self.schema,
                table=child,
                enabled=enabled,
                policy=rls_policy_name(tenant, base="tenxgraph_isolation_thread")
                if enabled
                else None,
                expression=expression if enabled else None,
            )
        if enabled:
            await warn_if_rls_bypassed(conn, logger, "PgCheckpointer")

    async def _get_recorded_schema_version(self, conn) -> int:
        """Return the highest schema version recorded in the tracking table.

        Returns 0 when no version has been recorded yet (first run).
        """
        return await recorded_schema_version(conn, self._get_table_name("schema_version"))

    async def _check_and_apply_schema_version(self, conn) -> None:
        """Apply any pending migrations and record the new schema version.

        Runs every migration between the recorded version and the target version
        in ascending order, inside the caller's transaction, then records the
        target version. Idempotent DDL means this is safe on both fresh and
        upgraded databases (see ``_build_migrations``).
        """
        await apply_migrations(
            conn,
            version_table=self._get_table_name("schema_version"),
            target=self._get_current_schema_version(),
            migrations=self._build_migrations(),
            owner="PgCheckpointer",
            logger=logger,
        )

    async def _initialize_schema(self) -> None:
        """
        Initialize database schema if not already done.

        Returns:
            None
        """
        if self._schema_initialized:
            return

        logger.debug(
            "Initializing database schema with types: id_type=%s, user_id_type=%s",
            self.id_type,
            self.user_id_type,
        )

        async with self._schema_init_lock:
            # Re-check under the in-process lock (another coroutine may have run
            # initialization while we waited).
            if self._schema_initialized:
                return

            async def _setup(conn) -> None:
                # Serialize schema creation ACROSS processes (multiple gunicorn workers / k8s
                # replicas starting at once). An in-process flag is not enough: concurrent
                # CREATE TABLE IF NOT EXISTS / CREATE TYPE can still fail with "duplicate key
                # value violates unique constraint pg_type_typname_nsp_index", which the
                # DO-block's duplicate_object guard does not catch. The lock is session-level
                # so it also covers the CREATE INDEX CONCURRENTLY that runs after the
                # transaction; every DDL statement waits at most ddl_lock_timeout for a lock.
                async with ddl_session(
                    conn,
                    lock_key=SCHEMA_INIT_LOCK_ID,
                    timeout=self.ddl_lock_timeout,
                    owner="PgCheckpointer",
                ):
                    async with conn.transaction():
                        # Catalog first: a restart runs no DDL for what is in place.
                        new_tables = await self._create_missing_tables(conn)
                        # Apply pending migrations and record the schema version.
                        await self._check_and_apply_schema_version(conn)
                        # Declared tenant/extra columns and RLS policies (idempotent).
                        await self._ensure_declared_columns(conn)
                        # Tables created just now are invisible to others: plain builds.
                        await self._ensure_indexes(conn, new_tables, concurrently=False)
                    # Indexes missing on tables in use are built without blocking them.
                    await self._ensure_indexes(
                        conn, self._index_tables() - new_tables, concurrently=True
                    )

            async def _operation() -> None:
                pool = await self._get_pg_pool()
                await run_on_connection(pool, _setup, timeout=self.pg_acquire_timeout)

            # Setup is idempotent, so a lost connection is retried like any query.
            try:
                await self._retry_on_connection_error(_operation)
            except Exception as e:
                logger.error("Failed to initialize database schema: %s", describe_error(e))
                raise
            self._schema_initialized = True
            logger.debug("Database schema initialized successfully")

    async def _create_missing_tables(self, conn) -> set[str]:
        """Create the tables (and the role enum) that do not exist yet; return their names.

        Reads the catalog first, so a restart runs no ``CREATE TABLE`` at all.
        """
        created: set[str] = set()
        for table, sql in self._build_table_statements():
            if table is not None:
                exists = await conn.fetchval(
                    "SELECT to_regclass($1) IS NOT NULL", self._get_table_name(table)
                )
                if exists:
                    continue
                created.add(table)
            logger.debug("Executing SQL: %s", sql.strip())
            await conn.execute(sql)
        return created

    @staticmethod
    def _index_tables() -> set[str]:
        """Tables whose indexes ``_ensure_indexes`` manages."""
        return {"threads", "states", "messages", "tool_executions"}

    async def _ensure_indexes(self, conn, tables: set[str], *, concurrently: bool) -> None:
        """Build the missing or invalid base and declared-column indexes of ``tables``.

        Reads the catalog first and skips indexes that are in place: ``CREATE INDEX IF NOT
        EXISTS`` would still wait for a SHARE lock behind every open write. With
        ``concurrently=True`` (tables in use) the builds use ``CREATE INDEX CONCURRENTLY``,
        outside any transaction; an invalid leftover of a failed build is dropped and rebuilt.
        """
        for table in sorted(tables & self._index_tables()):
            table_sql = self._get_table_name(table)
            catalog = await index_catalog(conn, table_sql)
            for name, index_table, definition in self._base_index_specs():
                if index_table != table:
                    continue
                await ensure_index(
                    conn,
                    catalog,
                    name=name,
                    table_sql=table_sql,
                    definition=definition,
                    concurrently=concurrently,
                )
            if table == "threads" and self.columns:
                await ensure_column_indexes(
                    conn,
                    table_sql,
                    self.columns,
                    "idx_threads",
                    concurrently=concurrently,
                    catalog=catalog,
                )

    async def aget_schema_version(self) -> int:
        """The schema version recorded in the database; 0 when the schema is missing.

        One cheap indexed query that never creates anything, for readiness checks. Compare it
        with ``CURRENT_SCHEMA_VERSION`` (also ``self._get_current_schema_version()``).
        """

        async def _read(conn) -> int:
            table = self._get_table_name("schema_version")
            if not await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", table):
                return 0
            return await self._get_recorded_schema_version(conn)

        async def _operation() -> int:
            pool = await self._get_pg_pool()
            return await run_on_connection(pool, _read, timeout=self.pg_acquire_timeout)

        return await self._retry_on_connection_error(_operation)

    ###########################
    #### SETUP METHODS ########
    ###########################

    async def asetup(self) -> Any:
        """
        Asynchronous setup method. Initializes database schema.

        Returns:
            Any: True if setup completed.
        """
        logger.info(
            "Setting up PgCheckpointer (async)",
            extra={
                "id_type": self.id_type,
                "user_id_type": self.user_id_type,
                "schema": self.schema,
            },
        )
        await self._initialize_schema()
        logger.info("PgCheckpointer setup completed")
        return True
