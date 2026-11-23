"""Query running, isolation scoping and thread access checks."""

import functools
import json
import logging
from collections.abc import Sequence
from typing import Any

from tenxgraph.core.exceptions.storage_exceptions import (
    StorageError,
    ThreadNotFoundError,
)
from tenxgraph.storage._pg_common import (
    loads_jsonb,
    retry_on_connection_error,
    run_on_connection,
)
from tenxgraph.storage.columns import (
    build_isolation_where,
    column_value_errors,
    resolve_isolation,
)
from tenxgraph.utils import ThreadInfo
from tenxgraph.utils.log_safety import describe_error

from ._pg_host import PgHostBase
from ._pg_support import (
    _CACHE_VERSION_KEY,
    StateT,
    _Isolation,
    _same_value,
)


logger = logging.getLogger("tenxgraph.checkpointer.pg")


class _PgAccessMixin(PgHostBase):
    """Query running, isolation scoping and thread access checks."""

    async def _retry_on_connection_error(
        self,
        operation,
        *args,
        max_retries=3,
        **kwargs,
    ):
        """Run ``operation(*args, **kwargs)``, retrying on a lost connection.

        Backoff 1s, then 2s (``storage/_pg_common.py``, shared with PgVectorStore). Other
        errors propagate unchanged; the calling method logs them.
        """
        return await retry_on_connection_error(
            functools.partial(operation, *args, **kwargs),
            owner="PgCheckpointer",
            logger=logger,
            max_retries=max_retries,
        )

    async def _run_query(
        self,
        fn,
        *,
        in_transaction: bool = False,
        settings: Sequence[tuple[str, str]] | None = None,
    ):
        """Acquire a pooled connection, run ``fn(conn)``, retrying on conn errors.

        Collapses the ``acquire() as conn`` + ``_retry_on_connection_error``
        boilerplate that every query method previously repeated. When
        ``in_transaction`` is True the callback runs inside ``conn.transaction()``
        so a multi-statement write commits atomically; a connection error retries
        the whole callback on a fresh connection (``max_retries=3``, matching the
        previous per-method wrapping).

        Args:
            fn: Coroutine function taking a single ``conn`` argument.
            in_transaction: Wrap the callback in a database transaction.
            settings: Native RLS values for this call (see ``_isolation``). When given the
                callback always runs in a transaction that first sets them with
                ``set_config(name, value, true)``, so they end with the transaction.

        Returns:
            Whatever ``fn`` returns.
        """

        # First use creates or upgrades the schema (once per process, under an advisory
        # lock), so a fresh database works without a separate setup step.
        await self._initialize_schema()

        async def _operation():
            return await run_on_connection(
                await self._get_pg_pool(),
                fn,
                timeout=self.pg_acquire_timeout,
                settings=settings,
                in_transaction=in_transaction,
            )

        return await self._retry_on_connection_error(_operation, max_retries=3)

    def _run_scoped(
        self,
        fn,
        *,
        settings: Sequence[tuple[str, str]] | None,
        in_transaction: bool = False,
    ):
        """``_run_query`` with RLS settings; without them it is the plain call it always was."""
        if settings:
            return self._run_query(fn, in_transaction=in_transaction, settings=settings)
        if in_transaction:
            return self._run_query(fn, in_transaction=True)
        return self._run_query(fn)

    # JSONB columns may come back decoded or as text/bytes; NULL stays None.
    _loads_jsonb = staticmethod(loads_jsonb)

    def _isolation(self, config: dict[str, Any] | None, *, write: bool = False) -> _Isolation:
        """Resolve the declared columns' values and the tenant isolation for one call.

        The rule is shared with PgVectorStore: ``tenxgraph.storage.columns.resolve_isolation``.
        """
        return resolve_isolation(
            self.columns, config, write=write, row_level_security=self.row_level_security
        )

    def _rls(
        self, config: dict[str, Any] | None, *, write: bool = False
    ) -> list[tuple[str, str]] | None:
        """The RLS settings for a call (``None`` when RLS is off). Writes pass ``write=True``
        so the backstop runs at reach ``own``, like the SQL owner checks."""
        if not self.row_level_security:
            return None
        return self._isolation(config, write=write).settings

    def _thread_filter(
        self,
        user_id: str | int | None,
        config: dict[str, Any] | None,
        params: list[Any],
        isolation: _Isolation | None = None,
        *,
        write: bool = False,
    ) -> str:
        """Conditions on the ``threads`` row for this caller: owner and picked tenant columns.

        Appends the values to ``params`` and returns ``user_id = $n AND "org_id" = $m ...``,
        or an empty string when nothing applies (isolation off, no tenant columns).
        ``write=True`` scopes to the caller's own threads whatever the reach.
        """
        isolation = isolation if isolation is not None else self._isolation(config, write=write)
        parts: list[str] = []
        if self._isolation_active(user_id, config, write=write):
            params.append(user_id)
            parts.append(f"user_id = ${len(params)}")
        tenant_sql = build_isolation_where(isolation.conditions, params)
        if tenant_sql:
            parts.append(tenant_sql)
        return " AND ".join(parts)

    def _thread_row_scope(
        self,
        thread_id: str | int,
        user_id: str | int | None,
        config: dict[str, Any] | None,
    ) -> tuple[str, list[Any]]:
        """Build the WHERE fragment + params for a direct ``threads`` row lookup.

        Unlike :meth:`_thread_scope` (which scopes ``states``/``messages`` through
        a subquery), this keys straight off the ``threads`` table's own ``user_id``
        and tenant columns. Shared by the thread read/delete paths.
        """
        params: list[Any] = [thread_id]
        owner = self._thread_filter(user_id, config, params)
        if owner:
            return f"thread_id = $1 AND {owner}", params
        return "thread_id = $1", params

    def _append_owner_scope(
        self,
        query: str,
        params: list[Any],
        user_id: str | int | None,
        config: dict[str, Any] | None,
        *,
        write: bool = False,
    ) -> str:
        """Append the thread-ownership filter to a ``messages`` query when active.

        Mutates ``params`` in place (appending ``user_id`` and tenant values) and returns
        the query with the extra ``AND thread_id IN (...)`` clause. A no-op when isolation
        is off and no tenant column applies, so a single-tenant caller does not pay for the
        join. ``write=True`` scopes to the caller's own threads whatever the reach.
        """
        owner = self._thread_filter(user_id, config, params, write=write)
        if owner:
            query += (
                f" AND thread_id IN (SELECT thread_id FROM {self._get_table_name('threads')} "  # noqa: S608
                f"WHERE {owner})"
            )
        return query

    def _thread_insert_sql(
        self,
        thread_id: str | int,
        thread_name: Any,
        user_id: str | int,
        meta_json: str,
        isolation: _Isolation,
        tail: str,
    ) -> tuple[str, list[Any]]:
        """``INSERT INTO threads`` with the declared column values, plus ``tail`` (conflict)."""
        columns = "thread_id, thread_name, user_id, meta"
        values = "$1, $2, $3, $4"
        params: list[Any] = [thread_id, thread_name, user_id, meta_json]
        for spec in self.columns.all:
            params.append(isolation.values.get(spec.name))
            columns += f", {spec.quoted}"
            values += f", {spec.placeholder(f'${len(params)}')}"
        sql = (
            f"INSERT INTO {self._get_table_name('threads')} ({columns}) "  # noqa: S608
            f"VALUES ({values}) {tail}"
        )
        return sql, params

    def _insert_value_errors(self):
        """Map a codec error on a declared column of the thread insert to ``ColumnValueError``.

        The declared columns follow the four fixed parameters of :meth:`_thread_insert_sql`.
        """
        return column_value_errors({5 + i: spec for i, spec in enumerate(self.columns.all)})

    async def _check_thread_access(
        self,
        conn,
        thread_id: str | int,
        user_id: str | int,
        config: dict[str, Any],
        isolation: _Isolation,
        *,
        lock: bool,
    ) -> None:
        """Read the thread row back and refuse a writer who does not own it.

        Reading what is actually in the table after the insert-if-absent is race-safe: when a
        concurrent request created the thread, we see the winner's values. Writes always stay
        ``own`` (``isolation`` comes from ``_isolation(config, write=True)``).

        Raises:
            ThreadNotFoundError: The thread belongs to another user or tenant.
        """
        threads = self._get_table_name("threads")
        suffix = " FOR UPDATE" if lock else ""
        row = None
        if self.columns.tenant:
            names = ", ".join(spec.quoted for spec in self.columns.tenant)
            row = await conn.fetchrow(
                f"SELECT user_id, {names} FROM {threads} WHERE thread_id = $1{suffix}",  # noqa: S608
                thread_id,
            )
            owner = row["user_id"] if row is not None else None
        elif lock:
            owner_row = await conn.fetchrow(
                f"SELECT user_id FROM {threads} WHERE thread_id = $1 FOR UPDATE",  # noqa: S608
                thread_id,
            )
            owner = owner_row["user_id"] if owner_row is not None else None
        else:
            owner = await conn.fetchval(
                f"SELECT user_id FROM {threads} WHERE thread_id = $1",  # noqa: S608
                thread_id,
            )
        if owner is None:
            if self.row_level_security:
                # Under RLS a thread of another tenant is invisible rather than mismatched.
                raise ThreadNotFoundError(thread_id)
            raise StorageError(
                message=(
                    "Thread row disappeared during write" if lock else "Thread could not be created"
                ),
                error_code="STORAGE_002",
                context={"thread_id": thread_id},
            )
        if self._isolation_active(user_id, config, write=True) and str(owner) != str(user_id):
            raise ThreadNotFoundError(thread_id)
        for spec, value in isolation.conditions:
            if not _same_value(spec, row[spec.name] if row is not None else None, value):
                raise ThreadNotFoundError(thread_id)
        # The caller passed the check: later cache and stop keys of this call use the thread's
        # own identity (see _key_identity).
        self._remember_identity(config, thread_id, row)

    def _row_to_thread_info(self, row) -> ThreadInfo:
        """Build a :class:`ThreadInfo` from a ``threads`` row."""
        meta_dict = self._loads_jsonb(row["meta"]) if row["meta"] else {}
        return ThreadInfo(
            thread_id=row["thread_id"],
            thread_name=row["thread_name"],
            user_id=row["user_id"],
            metadata=meta_dict,
            run_id=meta_dict.get("run_id"),
            updated_at=row["updated_at"],
        )

    async def _ensure_thread_exists(
        self,
        thread_id: str | int,
        user_id: str | int,
        config: dict[str, Any],
    ) -> None:
        """
        Ensure thread exists in database, create if not.

        Args:
            thread_id (str|int): Thread identifier.
            user_id (str|int): User identifier.
            config (dict): Configuration dictionary.

        Returns:
            None

        Raises:
            Exception: If creation fails.
        """
        try:
            isolation = self._isolation(config, write=True)
            insert_sql, insert_params = self._thread_insert_sql(
                thread_id,
                config.get("thread_name", f"Thread {thread_id}"),
                user_id,
                json.dumps(config.get("thread_meta", {})),
                isolation,
                "ON CONFLICT DO NOTHING",
            )

            async def _check_and_create_thread(conn):
                # Insert-if-absent, then verify ownership from what is actually
                # in the table. Checking existence with (thread_id AND user_id)
                # first was unsafe: for a thread owned by ANOTHER user the
                # lookup missed, the INSERT silently no-opped on conflict, and
                # the caller went on to write into that user's thread.
                # Reading the owner back after the insert is also race-safe:
                # if a concurrent request created the thread, we observe the
                # winner's user_id (and tenant values) rather than assuming we created it.
                with self._insert_value_errors():
                    await conn.execute(insert_sql, *insert_params)
                await self._check_thread_access(
                    conn, thread_id, user_id, config, isolation, lock=False
                )

            await self._run_scoped(_check_and_create_thread, settings=isolation.settings)

        except Exception as e:
            logger.error("Failed to ensure thread exists: %s", describe_error(e))
            raise

    def _serialize_state_payload(self, state: StateT) -> dict[str, Any]:
        """Build the JSON-safe payload persisted for a state.

        Holds the state data plus a small header (see
        :meth:`BaseCheckpointer._encode_state`). No class path is stored: rows are
        rebuilt into the state class bound by ``StateGraph.compile()``.
        """
        return self._encode_state(state)

    def _deserialize_state_payload(self, data: dict[str, Any], thread_id: Any = None) -> Any:
        """Reconstruct a state from a persisted payload using the bound state class.

        The Redis cache copy also carries the checkpoint version, which is not a
        state field and is dropped here.
        """
        data = dict(data)
        data.pop(_CACHE_VERSION_KEY, None)
        return self._decode_state(data, thread_id)

    def _thread_scope(
        self,
        thread_id: str | int,
        user_id: str | int | None,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, list[Any]]:
        """Build the WHERE fragment + params selecting a thread's rows.

        ``states``/``messages`` carry a ``thread_id`` but no ``user_id`` of their
        own; ownership lives on ``threads``. When isolation is active (per the caller's
        reach, or ``enforce_user_isolation`` when there is no identity) we scope through the
        parent table, so a caller can only touch threads they own. Otherwise we key on
        ``thread_id`` alone and skip the join entirely.

        Returns:
            tuple[str, list]: SQL fragment using $1..$n, and the matching params.
        """
        params: list[Any] = [thread_id]
        owner = self._thread_filter(user_id, config, params)
        if owner:
            return (
                f"thread_id = $1 AND thread_id IN "  # noqa: S608
                f"(SELECT thread_id FROM {self._get_table_name('threads')} "
                f"WHERE {owner})",
                params,
            )
        return ("thread_id = $1", params)

    def _isolation_active(
        self,
        user_id: str | int | None,
        config: dict[str, Any] | None = None,
        *,
        write: bool = False,
    ) -> bool:
        """Whether ownership scoping should be applied for this call.

        Driven by the identity in ``config["user"]`` when present (a write is always scoped
        to its own user), falling back to this checkpointer's ``enforce_user_isolation``
        setting otherwise (so ``config`` is optional and a caller that omits it keeps the
        historical behaviour). Scoping still requires an actual ``user_id`` -- scoping on a
        missing one would match nothing.
        """
        return self._isolation_enabled(config, user_id, write=write)

    async def _lock_thread_for_write(
        self,
        conn,
        thread_id: str | int,
        user_id: str | int,
        config: dict[str, Any],
    ) -> int:
        """Ensure the thread exists, take a row lock, and return its current version.

        Creating the thread (if absent) and locking its row serializes concurrent
        writers on the same thread, so their version numbers cannot collide. The
        ownership check enforces per-user isolation on writes.

        Returns:
            int: The highest existing state ``version`` for the thread (0 if none).

        Raises:
            ThreadNotFoundError: If the thread is owned by a different user or tenant.
        """
        isolation = self._isolation(config, write=True)
        insert_sql, insert_params = self._thread_insert_sql(
            thread_id,
            config.get("thread_name", f"Thread {thread_id}"),
            user_id,
            json.dumps(config.get("thread_meta", {})),
            isolation,
            "ON CONFLICT DO NOTHING",
        )
        with self._insert_value_errors():
            await conn.execute(insert_sql, *insert_params)
        await self._check_thread_access(conn, thread_id, user_id, config, isolation, lock=True)

        version_row = await conn.fetchrow(
            f"SELECT COALESCE(MAX(version), 0) AS v "  # noqa: S608
            f"FROM {self._get_table_name('states')} WHERE thread_id = $1",
            thread_id,
        )
        return int(version_row["v"]) if version_row else 0
