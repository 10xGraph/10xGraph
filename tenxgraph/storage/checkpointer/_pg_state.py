"""Durable state reads and writes with optimistic versioning."""

import json
import logging
from typing import Any

from tenxgraph.core.exceptions.storage_exceptions import (
    StaleStateError,
    StorageError,
    ThreadNotFoundError,
    TransientStorageError,
)
from tenxgraph.core.state import Message
from tenxgraph.storage.columns import (
    rows_affected as _rows_affected,
)
from tenxgraph.utils import metrics
from tenxgraph.utils.log_safety import describe_error

from ._pg_support import (
    STATE_VERSION_CONFIG_KEY,
    StateT,
    asyncpg,
)


logger = logging.getLogger("tenxgraph.checkpointer.pg")


class _PgStateMixin:
    """Durable state reads and writes with optimistic versioning."""

    async def _state_row_matches(
        self, conn, thread_id: str | int, version: int, state_json: str
    ) -> bool:
        """Whether the stored state row at ``version`` holds the same state as ``state_json``."""
        row = await conn.fetchrow(
            f"SELECT state_data FROM {self._get_table_name('states')} "  # noqa: S608
            f"WHERE thread_id = $1 AND version = $2",
            thread_id,
            version,
        )
        if row is None:
            return False
        return self._loads_jsonb(row["state_data"]) == json.loads(state_json)

    async def _write_state_row(
        self,
        conn,
        thread_id: str | int,
        user_id: str | int,
        state: StateT,
        config: dict[str, Any],
        expected_version: int | None,
        *,
        retried: bool = False,
    ) -> int:
        """Append a new versioned state row inside an existing transaction.

        Performs the optimistic compare-and-swap: if ``expected_version`` is set
        and no longer matches the thread's current version, another execution
        committed in the meantime and this write is rejected as stale rather than
        silently overwriting it. Old rows beyond the history window are pruned.

        ``retried`` is set when this is a repeat of a write whose first attempt died on a
        connection error. The first attempt may have committed before the reply was lost; the
        thread is then exactly one version ahead holding this very state, and the write is
        reported as done instead of failing against its own commit.

        Returns:
            int: The new version assigned to the written row.
        """
        current_version = await self._lock_thread_for_write(conn, thread_id, user_id, config)

        state_json = json.dumps(self._serialize_state_payload(state))
        if (
            retried
            and expected_version is not None
            and current_version == int(expected_version) + 1
            and await self._state_row_matches(conn, thread_id, current_version, state_json)
        ):
            logger.warning(
                "Write for thread_id=%s had already committed before its retry; keeping it",
                thread_id,
            )
            return current_version

        if expected_version is not None and int(expected_version) != current_version:
            raise StaleStateError(
                message=(
                    "State was modified by another execution "
                    f"(expected version {expected_version}, found {current_version})"
                ),
                error_code="STORAGE_CONFLICT_001",
                context={
                    "thread_id": thread_id,
                    "expected_version": expected_version,
                    "current_version": current_version,
                },
            )

        new_version = current_version + 1
        await conn.execute(
            f"""
            INSERT INTO {self._get_table_name("states")}
                (thread_id, version, state_data, meta)
            VALUES ($1, $2, $3, $4)
            """,  # noqa: S608
            thread_id,
            new_version,
            state_json,
            json.dumps(config.get("meta", {})),
        )

        # Prune history beyond the retention window (bounded append-only log).
        if self.state_history_limit and self.state_history_limit > 0:
            await conn.execute(
                f"DELETE FROM {self._get_table_name('states')} "  # noqa: S608
                f"WHERE thread_id = $1 AND version <= $2",
                thread_id,
                new_version - self.state_history_limit,
            )

        return new_version

    ###########################
    #### STATE METHODS ########
    ###########################

    async def aput_state(
        self,
        config: dict[str, Any],
        state: StateT,
    ) -> StateT:
        """
        Store state in PostgreSQL with optimistic concurrency control.

        A new versioned row is appended under a per-thread row lock. When
        ``config['_checkpoint_version']`` is present (set by a prior read within
        the same run), the write performs a compare-and-swap and raises
        :class:`StaleStateError` if another execution advanced the thread first.
        On success the config's version marker is advanced to the new version and
        the Redis cache is refreshed with the stored state at that version, so the
        next run on the thread starts from it.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.

        Returns:
            StateT: The stored state object.

        Raises:
            StorageError: If storing fails.
            StaleStateError: If the optimistic version check fails.
            ThreadNotFoundError: The thread belongs to another user or tenant.
        """
        thread_id, user_id = self._validate_config(config)
        # A missing or malformed tenant value is a config error: raise it as is, before the
        # generic StorageError wrapping below.
        self._isolation(config, write=True)

        logger.debug("Storing state for thread_id=%s, user_id=%s", thread_id, user_id)
        metrics.counter("pg_checkpointer.save_state.attempts").inc()

        expected_version = config.get(STATE_VERSION_CONFIG_KEY)

        with metrics.timer("pg_checkpointer.save_state.duration"):
            try:
                attempts = 0

                async def _store_state(conn):
                    nonlocal attempts
                    attempts += 1
                    return await self._write_state_row(
                        conn,
                        thread_id,
                        user_id,
                        state,
                        config,
                        expected_version,
                        retried=attempts > 1,
                    )

                new_version = await self._run_scoped(
                    _store_state,
                    in_transaction=True,
                    settings=self._rls(config, write=True),
                )
                config[STATE_VERSION_CONFIG_KEY] = new_version
                logger.debug(
                    "State stored for thread_id=%s at version=%s",
                    thread_id,
                    new_version,
                )
                metrics.counter("pg_checkpointer.save_state.success").inc()
                # Write-through. A run loads its state from the cache first, with the
                # durable version embedded for its compare-and-swap. Without this, a
                # state written here (outside a run) left the cache on the previous
                # state AND version: the next run started from stale state, and its
                # first write raised StaleStateError, so that run and this update were
                # both lost. Best-effort and version-guarded, like every cache write.
                await self.aput_state_cache(config, state)
                return state

            except StaleStateError:
                metrics.counter("pg_checkpointer.save_state.conflict").inc()
                # The cache may hold state built on the stale version; drop it so
                # the next read falls back to Postgres instead of re-seeding the
                # same doomed expected version.
                await self._invalidate_state_cache(thread_id, user_id, config=config)
                raise
            except ThreadNotFoundError:
                raise  # another user's thread: nothing was written, the API answers 404
            except Exception as e:
                metrics.counter("pg_checkpointer.save_state.error").inc()
                logger.error(
                    "Failed to store state for thread_id=%s: %s", thread_id, describe_error(e)
                )
                if asyncpg and hasattr(asyncpg, "ConnectionDoesNotExistError"):
                    connection_errors = (
                        asyncpg.ConnectionDoesNotExistError,
                        asyncpg.InterfaceError,
                    )
                    if isinstance(e, connection_errors):
                        raise TransientStorageError(
                            message=f"Connection issue storing state: {describe_error(e)}",
                            error_code="STORAGE_TRANSIENT_001",
                            context={
                                "thread_id": thread_id,
                                "error_type": type(e).__name__,
                            },
                        ) from e
                raise StorageError(
                    message=f"Failed to store state: {describe_error(e)}",
                    error_code="STORAGE_001",
                    context={
                        "thread_id": thread_id,
                        "error_type": type(e).__name__,
                    },
                ) from e

    async def aput_checkpoint(
        self,
        config: dict[str, Any],
        state: StateT,
        messages: list[Message] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StateT:
        """
        Atomically persist a state and its messages in a single transaction.

        This is the durable-checkpoint entry point used by the graph loop. State
        and messages are committed together, so a crash can never leave state
        advanced without the messages that justify it (audit M7). The same
        optimistic version check as :meth:`aput_state` applies.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.
            messages (list[Message], optional): Messages to store atomically.
            metadata (dict, optional): Additional message metadata.

        Returns:
            StateT: The stored state object.

        Raises:
            StorageError: If storing fails.
            StaleStateError: If the optimistic version check fails.
            ThreadNotFoundError: The thread belongs to another user or tenant.
        """
        thread_id, user_id = self._validate_config(config)
        self._isolation(config, write=True)  # config errors raise as is (see aput_state)
        messages = messages or []
        expected_version = config.get(STATE_VERSION_CONFIG_KEY)

        metrics.counter("pg_checkpointer.save_checkpoint.attempts").inc()
        with metrics.timer("pg_checkpointer.save_checkpoint.duration"):
            try:
                attempts = 0

                async def _store_checkpoint(conn):
                    nonlocal attempts
                    attempts += 1
                    version = await self._write_state_row(
                        conn,
                        thread_id,
                        user_id,
                        state,
                        config,
                        expected_version,
                        retried=attempts > 1,
                    )
                    if messages:
                        await self._insert_messages(conn, thread_id, messages, metadata)
                    return version

                new_version = await self._run_scoped(
                    _store_checkpoint,
                    in_transaction=True,
                    settings=self._rls(config, write=True),
                )
                config[STATE_VERSION_CONFIG_KEY] = new_version
                logger.debug(
                    "Checkpoint stored for thread_id=%s at version=%s (%d messages)",
                    thread_id,
                    new_version,
                    len(messages),
                )
                metrics.counter("pg_checkpointer.save_checkpoint.success").inc()
                return state

            except StaleStateError:
                metrics.counter("pg_checkpointer.save_checkpoint.conflict").inc()
                # See aput_state: drop the possibly-poisoned cache so the thread
                # is not wedged into failing every subsequent compare-and-swap.
                await self._invalidate_state_cache(thread_id, user_id, config=config)
                raise
            except ThreadNotFoundError:
                raise  # another user's thread: nothing was written, the API answers 404
            except Exception as e:
                metrics.counter("pg_checkpointer.save_checkpoint.error").inc()
                logger.error(
                    "Failed to store checkpoint for thread_id=%s: %s", thread_id, describe_error(e)
                )
                raise StorageError(
                    message=f"Failed to store checkpoint: {describe_error(e)}",
                    error_code="STORAGE_001",
                    context={"thread_id": thread_id, "error_type": type(e).__name__},
                ) from e

    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve the latest state for a thread from PostgreSQL.

        Reads the highest-version row deterministically (``version DESC,
        state_id DESC``) scoped to the requesting ``user_id`` so a developer who
        enables auth cannot read another user's thread (audit B1/B3). The read
        version is recorded in ``config['_checkpoint_version']`` so a subsequent
        write in the same run can perform its compare-and-swap.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Retrieved state or None.

        Raises:
            Exception: If retrieval fails.
        """
        thread_id, user_id = self._validate_config(config)

        logger.debug("Retrieving state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            scope_sql, scope_params = self._thread_scope(thread_id, user_id, config)
            query = f"""
                SELECT version, state_data FROM {self._get_table_name("states")}
                WHERE {scope_sql}
                ORDER BY version DESC, state_id DESC
                LIMIT 1
                """  # noqa: S608

            row = await self._run_scoped(
                lambda conn: conn.fetchrow(query, *scope_params),
                settings=self._rls(config),
            )

            if row:
                data = self._loads_jsonb(row["state_data"])
                logger.debug(
                    "State found for thread_id=%s at version=%s",
                    thread_id,
                    row["version"],
                )
                config[STATE_VERSION_CONFIG_KEY] = int(row["version"])
                return self._deserialize_state_payload(data, thread_id)

            logger.debug("No state found for thread_id=%s", thread_id)
            return None

        except Exception as e:
            logger.error(
                "Failed to retrieve state for thread_id=%s: %s", thread_id, describe_error(e)
            )
            raise

    async def aclear_state(self, config: dict[str, Any]) -> int:
        """
        Clear state from PostgreSQL and Redis cache (scoped by the caller's reach).

        Args:
            config (dict): Configuration dictionary.

        Returns:
            int: The number of state rows (versions) removed; 0 when the caller's scope
            matched nothing. 0 is not "thread not found": the caller's own thread with no
            state also gives 0.

        Raises:
            Exception: If clearing fails.
        """
        thread_id, user_id = self._validate_config(config)

        logger.debug("Clearing state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            scope_sql, scope_params = self._thread_scope(thread_id, user_id, config)
            # With tenant columns the cache key is the thread's identity: read it (scoped)
            # before the rows go, so the shared entry is the one dropped below.
            await self._resolve_thread_identity(config)

            # Clear from PostgreSQL with retry logic, scoped to the owner.
            query = f"DELETE FROM {self._get_table_name('states')} WHERE {scope_sql}"  # noqa: S608
            status = await self._run_scoped(
                lambda conn: conn.execute(query, *scope_params),
                settings=self._rls(config),
            )

            # Clear from Redis cache (best effort: Postgres is already cleared)
            await self._invalidate_state_cache(thread_id, user_id, retries=1, config=config)
            config.pop(STATE_VERSION_CONFIG_KEY, None)

            logger.debug("State cleared for thread_id=%s", thread_id)
            return _rows_affected(status)

        except Exception as e:
            logger.error("Failed to clear state for thread_id=%s: %s", thread_id, describe_error(e))
            raise
