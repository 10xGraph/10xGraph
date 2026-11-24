"""Thread rows: create, read, list, clean."""

import json
import logging
from typing import Any

from tenxgraph.core.authz import owner_id
from tenxgraph.core.exceptions.storage_exceptions import (
    ThreadNotFoundError,
)
from tenxgraph.storage.columns import (
    build_isolation_where,
    rls_all_rows_settings,
)
from tenxgraph.storage.columns import (
    rows_affected as _rows_affected,
)
from tenxgraph.storage.sql_filter import compile_filters, validate_filters
from tenxgraph.utils import ThreadInfo
from tenxgraph.utils.log_safety import describe_error

from ._pg_host import PgHostBase
from .base_checkpointer import like_escape, reject_unstorable, validate_page


logger = logging.getLogger("tenxgraph.checkpointer.pg")


class _PgThreadsMixin(PgHostBase):
    """Thread rows: create, read, list, clean."""

    ###########################
    #### THREAD METHODS #######
    ###########################

    async def aput_thread(
        self,
        config: dict[str, Any],
        thread_info: ThreadInfo,
    ) -> bool:
        """
        Create or update thread information.

        Args:
            config (dict): Configuration dictionary.
            thread_info (ThreadInfo): Thread information object.

        The caller becomes the owner of a new thread (``thread_info.user_id`` is ignored).
        An update stays own whatever the caller's reach.

        Returns:
            bool: True if thread was created (inserted).
                  False if thread already existed and was updated.

        Raises:
            ValueError: ``thread_info.thread_id`` differs from ``config["thread_id"]``.
            ThreadNotFoundError: The thread belongs to another user or tenant.
            Exception: If storing fails.
        """
        reject_unstorable(thread_info)
        # Ensure schema is initialized before accessing tables
        thread_id, user_id = self._validate_config(config)
        if str(thread_info.thread_id) != str(config["thread_id"]):
            raise ValueError(
                "ThreadInfo.thread_id must match config['thread_id']: "
                f"{thread_info.thread_id!r} != {str(config['thread_id'])!r}"
            )

        logger.debug("Storing thread info for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            thread_name = thread_info.thread_name
            # A copy: the caller's thread_info.metadata is never changed.
            meta = {**(thread_info.metadata or {}), "run_id": thread_info.run_id}

            isolation = self._isolation(config, write=True)
            insert_sql, insert_params = self._thread_insert_sql(
                thread_id,
                thread_name,
                user_id,
                json.dumps(meta),
                isolation,
                "ON CONFLICT (thread_id) DO NOTHING RETURNING thread_id",
            )

            async def _put_thread(conn):
                # Try to insert; if inserted, RETURNING will return the row. Tenant and extra
                # values are written here only: they never change afterwards.
                with self._insert_value_errors():
                    row = await conn.fetchrow(insert_sql, *insert_params)

                if row:
                    # Insert succeeded
                    return True

                # Insert did nothing (conflict) -> update the existing row.
                # When isolation is on, the UPDATE is scoped by user_id too (whatever the
                # reach: writes stay own): without that, any caller could rename or re-tag
                # another user's thread. RETURNING lets us detect that case.
                # Only set thread_name when a non-None value is provided.
                scoped = self._isolation_active(user_id, config, write=True)

                if thread_name is None:
                    sets, params = "meta = $1, updated_at = NOW()", [json.dumps(meta)]
                else:
                    sets = "thread_name = $1, meta = $2, updated_at = NOW()"
                    params = [thread_name, json.dumps(meta)]

                params.append(thread_id)
                where = f"thread_id = ${len(params)}"
                if scoped:
                    params.append(user_id)
                    where += f" AND user_id = ${len(params)}"
                tenant_sql = build_isolation_where(isolation.conditions, params)
                if tenant_sql:
                    where += f" AND {tenant_sql}"
                    scoped = True

                updated = await conn.fetchrow(
                    f"""
                    UPDATE {self._get_table_name("threads")}
                    SET {sets}
                    WHERE {where}
                    RETURNING thread_id
                    """,  # noqa: S608
                    *params,
                )

                if updated is None and scoped:
                    # The thread exists but is owned by someone else (or was deleted since
                    # the insert): never revealed, the API answers 404.
                    raise ThreadNotFoundError(thread_id)

                return False

            created = await self._run_query(_put_thread, settings=isolation.settings)
            logger.debug("Thread info stored for thread_id=%s (created=%s)", thread_id, created)
            return bool(created)

        except Exception as e:
            logger.error(
                "Failed to store thread info for thread_id=%s: %s", thread_id, describe_error(e)
            )
            raise e

    async def aget_thread(
        self,
        config: dict[str, Any],
    ) -> ThreadInfo | None:
        """
        Get thread information.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ThreadInfo | None: Thread information object or None.

        Raises:
            Exception: If retrieval fails.
        """
        # Ensure schema is initialized before accessing tables
        thread_id, user_id = self._validate_config(config)

        logger.debug("Retrieving thread info for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            # Owner-scoped unless the developer disabled isolation.
            where, params = self._thread_row_scope(thread_id, user_id, config)
            query = f"""
                SELECT thread_id, thread_name, user_id, created_at, updated_at, meta
                FROM {self._get_table_name("threads")}
                WHERE {where}
                """  # noqa: S608

            row = await self._run_query(
                lambda conn: conn.fetchrow(query, *params), settings=self._rls(config)
            )

            if row:
                # The row's owner, not the caller: they differ under reach tenant or all.
                return self._row_to_thread_info(row)

            logger.debug("Thread not found for thread_id=%s, user_id=%s", thread_id, user_id)
            return None

        except Exception as e:
            logger.error(
                "Failed to retrieve thread info for thread_id=%s: %s", thread_id, describe_error(e)
            )
            raise e

    async def aget_thread_owner(self, thread_id: str | int) -> str | int | None:
        """Return the ``user_id`` that owns ``thread_id`` (global, not owner-scoped).

        Resolves ownership across all users so an authorization layer can reject a
        different user acting on the thread. Returns None if no such thread exists.
        """

        try:
            thread_id = self._db_thread_id(thread_id)
        except ValueError:
            return None  # not a valid id for this table, so no such thread exists
        query = f"""
            SELECT user_id
            FROM {self._get_table_name("threads")}
            WHERE thread_id = $1
            """  # noqa: S608
        return await self._run_query(
            lambda conn: conn.fetchval(query, thread_id),
            settings=self._global_settings(),
        )

    def _global_settings(self) -> list[tuple[str, str]] | None:
        """RLS settings for a deliberate cross-tenant lookup (owner/scope resolution)."""
        if not self.row_level_security:
            return None
        return rls_all_rows_settings(self.columns.tenant)

    async def aget_thread_scope(self, thread_id: str | int) -> dict[str, Any] | None:
        """Return ``{"user_id": ..., <tenant column>: ...}`` for a thread, globally.

        Not scoped to any caller (like :meth:`aget_thread_owner`): an authorization layer
        compares these values with the caller's on every column the caller's reach picks.
        Tenant values written before a column existed are ``None``. Returns ``None`` when the
        thread does not exist.
        """
        try:
            thread_id = self._db_thread_id(thread_id)
        except ValueError:
            return None
        names = "".join(f", {spec.quoted}" for spec in self.columns.tenant)
        query = (
            f"SELECT user_id{names} FROM {self._get_table_name('threads')} "  # noqa: S608
            "WHERE thread_id = $1"
        )
        row = await self._run_query(
            lambda conn: conn.fetchrow(query, thread_id),
            settings=self._global_settings(),
        )
        if row is None:
            return None
        scope: dict[str, Any] = {"user_id": row["user_id"]}
        for spec in self.columns.tenant:
            scope[spec.name] = row[spec.name]
        return scope

    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[ThreadInfo]:
        """
        List threads for a user with optional search, filters and pagination.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.
            filters (dict, optional): Conditions on ``thread_name``, ``created_at``,
                ``updated_at`` and the declared extra columns, e.g.
                ``{"plan": "pro", "created_at": {"gte": "2026-01-01T00:00:00+00:00"}}``
                (operators and limits in ``tenxgraph.storage.sql_filter``). ANDed after the
                isolation conditions, so they can only narrow the listing. Tenant columns and
                ``user_id`` are rejected.

        Returns:
            list[ThreadInfo]: List of thread information objects.

        Raises:
            FilterError: ``filters`` is malformed or names a column it may not.
            Exception: If listing fails.
        """
        offset, limit = validate_page(offset, limit)
        conditions = validate_filters(filters, self.filter_schema)
        user_id = owner_id(config)

        # `user_id = user_id or "test-user"` used to sit here, which made the check
        # below dead code and silently listed a fake user's threads whenever the
        # caller forgot user_id. With isolation on that is a confusing empty result;
        # with it off it is a cross-tenant listing. Ask for the id instead.
        if not user_id and self.enforce_user_isolation:
            raise ValueError("user_id must be provided in config to list threads")

        logger.debug("Listing threads for user_id=%s", user_id)

        try:
            # Build query with optional search. With isolation on, a user only
            # ever sees their own threads; with it off (single-tenant) the listing
            # is not restricted by user_id.
            query = f"""
                SELECT thread_id, thread_name, user_id, created_at, updated_at, meta
                FROM {self._get_table_name("threads")}
                WHERE TRUE
            """  # noqa: S608
            params: list[Any] = []

            owner = self._thread_filter(user_id, config, params)
            if owner:
                query += f" AND {owner}"

            if search:
                params.append(f"%{like_escape(search)}%")
                query += f" AND thread_name ILIKE ${len(params)}"

            # Client filters come last and are only ever ANDed, so they cannot widen the
            # isolation above; their $n numbering continues from ``params``.
            filter_sql = compile_filters(conditions, params)
            if filter_sql:
                query += f" AND {filter_sql}"

            query += " ORDER BY updated_at DESC"

            if limit:
                params.append(limit)
                query += f" LIMIT ${len(params)}"

            if offset:
                params.append(offset)
                query += f" OFFSET ${len(params)}"

            rows = await self._run_query(
                lambda conn: conn.fetch(query, *params), settings=self._rls(config)
            )
            if not rows:
                rows = []

            threads = [self._row_to_thread_info(row) for row in rows]
            logger.debug("Found %d threads for user_id=%s", len(threads), user_id)
            return threads

        except Exception as e:
            logger.error("Failed to list threads for user_id=%s: %s", user_id, describe_error(e))
            raise

    async def aclean_thread(self, config: dict[str, Any]) -> int:
        """
        Clean/delete a thread and all associated data (scoped by the caller's reach).

        Args:
            config (dict): Configuration dictionary.

        Returns:
            int: 1 when the thread was deleted, 0 when there is none the caller may see.

        Raises:
            Exception: If cleaning fails.
        """
        # Ensure schema is initialized before accessing tables
        thread_id, user_id = self._validate_config(config)

        logger.debug("Cleaning thread thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            # Owner-scoped unless the developer disabled isolation, so one user
            # cannot delete another's thread by id.
            where, params = self._thread_row_scope(thread_id, user_id, config)

            # Delete thread (cascade will handle messages and states) with retry logic
            query = f"DELETE FROM {self._get_table_name('threads')} WHERE {where}"  # noqa: S608
            if self.columns.tenant:
                # The deleted row's identity names the thread's shared cache entry.
                query += f" RETURNING {self._identity_columns_sql()}"
                row = await self._run_query(
                    lambda conn: conn.fetchrow(query, *params),
                    settings=self._rls(config),
                )
                self._remember_identity(config, thread_id, row)
                deleted = 0 if row is None else 1
            else:
                status = await self._run_query(
                    lambda conn: conn.execute(query, *params),
                    settings=self._rls(config),
                )
                deleted = _rows_affected(status)

            # The stop request and active-run marker go with the thread. Under tenant columns
            # their key was set from the deleted row above, so this is the thread's own key.
            # The cache calls log and swallow Redis errors, so they cannot fail the delete.
            if deleted:
                await self._clear_run_keys(config)

            # Postgres is the source of truth and is already gone. A Redis failure here
            # must not turn a completed delete into an error; the stale cache entry is
            # ignored on read (its version no longer matches Postgres, see
            # aget_state_cache) and expires with its TTL.
            await self._invalidate_state_cache(thread_id, user_id, retries=1, config=config)

            logger.debug("Thread cleaned: thread_id=%s, user_id=%s", thread_id, user_id)
            return deleted

        except Exception as e:
            logger.error("Failed to clean thread thread_id=%s: %s", thread_id, describe_error(e))
            raise
