"""Message reads and writes."""

import json
import logging
from typing import Any

from tenxgraph.core.authz import owner_id
from tenxgraph.core.state import Message
from tenxgraph.storage.columns import (
    rows_affected as _rows_affected,
)
from tenxgraph.utils.log_safety import describe_error

from ._pg_host import PgHostBase
from .base_checkpointer import validate_page


logger = logging.getLogger("tenxgraph.checkpointer.pg")


class _PgMessagesMixin(PgHostBase):
    """Message reads and writes."""

    ###########################
    #### MESSAGE METHODS ######
    ###########################

    async def _insert_messages(
        self,
        conn,
        thread_id: str | int,
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upsert messages on an already-open connection/transaction.

        Shared by :meth:`aput_messages` (standalone) and :meth:`aput_checkpoint`
        (atomic with the state write) so both use identical insert semantics.

        The primary key is ``(thread_id, message_id)``, so ids are per thread and a
        conflict can only be a re-save of a message in this same thread.
        """
        messages_table = self._get_table_name("messages")
        for message in messages:
            await conn.execute(
                f"""
                    INSERT INTO {messages_table} (
                        message_id, thread_id, role, content, tool_calls,
                        tool_call_id, total_tokens, usages, meta
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (thread_id, message_id) DO UPDATE SET
                        role = EXCLUDED.role,
                        content = EXCLUDED.content,
                        tool_calls = EXCLUDED.tool_calls,
                        tool_call_id = EXCLUDED.tool_call_id,
                        total_tokens = EXCLUDED.total_tokens,
                        usages = EXCLUDED.usages,
                        meta = EXCLUDED.meta,
                        updated_at = NOW()
                    """,  # noqa: S608
                message.message_id,
                thread_id,
                message.role,
                json.dumps([block.model_dump(mode="json") for block in message.content]),
                json.dumps(message.tools_calls) if message.tools_calls else None,
                getattr(message, "tool_call_id", None),
                message.usages.total_tokens if message.usages else 0,
                json.dumps(message.usages.model_dump()) if message.usages else None,
                json.dumps({**(metadata or {}), **(message.metadata or {})}),
            )

    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Store messages in PostgreSQL.

        Args:
            config (dict): Configuration dictionary.
            messages (list[Message]): List of messages to store.
            metadata (dict, optional): Additional metadata.

        Returns:
            Any: None

        Raises:
            Exception: If storing fails.
        """
        # Ensure schema is initialized before accessing tables
        thread_id, user_id = self._validate_config(config)

        if not messages:
            logger.debug("No messages to store for thread_id=%s", thread_id)
            return

        logger.debug("Storing %d messages for thread_id=%s", len(messages), thread_id)

        try:
            # Ensure thread exists
            await self._ensure_thread_exists(thread_id, user_id, config)

            # Store messages in batch with retry logic
            await self._run_scoped(
                lambda conn: self._insert_messages(conn, thread_id, messages, metadata),
                in_transaction=True,
                settings=self._rls(config, write=True),
            )
            logger.debug("Stored %d messages for thread_id=%s", len(messages), thread_id)

        except Exception as e:
            logger.error(
                "Failed to store messages for thread_id=%s: %s", thread_id, describe_error(e)
            )
            raise

    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """
        Retrieve a single message by ID.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Message: Retrieved message object.

        Raises:
            Exception: If retrieval fails.
        """
        thread_id = self._message_thread_id(config)
        user_id = owner_id(config)

        # Message ids are unique per thread only, so the thread is part of the key.

        logger.debug("Retrieving message_id=%s for thread_id=%s", message_id, thread_id)

        try:
            query = f"""
                SELECT message_id, thread_id, role, content, tool_calls,
                       tool_call_id, created_at, total_tokens,
                       usages, meta
                FROM {self._get_table_name("messages")}
                WHERE message_id = $1 AND thread_id = $2
            """  # noqa: S608
            params: list[Any] = [message_id, thread_id]
            # Owner scoping: a caller may only read messages on threads they own,
            # even if they know the message/thread id.
            query = self._append_owner_scope(query, params, user_id, config)

            row = await self._run_scoped(
                lambda conn: conn.fetchrow(query, *params), settings=self._rls(config)
            )

            if not row:
                raise ValueError(f"Message not found: {message_id}")

            return self._row_to_message(row)

        except Exception as e:
            logger.error("Failed to retrieve message_id=%s: %s", message_id, describe_error(e))
            raise

    async def alist_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        List messages for a thread with optional search and pagination.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[Message]: List of message objects.

        Raises:
            Exception: If listing fails.
        """
        offset, limit = validate_page(offset, limit)
        thread_id = self._message_thread_id(config)
        user_id = owner_id(config)

        logger.debug("Listing messages for thread_id=%s", thread_id)

        try:
            # Build query with optional search
            query = f"""
                SELECT message_id, thread_id, role, content, tool_calls,
                       tool_call_id, created_at, total_tokens,
                       usages, meta
                FROM {self._get_table_name("messages")}
                WHERE thread_id = $1
            """  # noqa: S608
            params: list[Any] = [thread_id]

            # Owner scoping: listing a thread you do not own returns nothing.
            query = self._append_owner_scope(query, params, user_id, config)

            if search:
                params.append(f"%{search}%")
                query += f" AND content ILIKE ${len(params)}"

            query += " ORDER BY created_at ASC"

            if limit:
                params.append(limit)
                query += f" LIMIT ${len(params)}"

            if offset:
                params.append(offset)
                query += f" OFFSET ${len(params)}"

            rows = await self._run_scoped(
                lambda conn: conn.fetch(query, *params), settings=self._rls(config)
            )
            if not rows:
                rows = []
            messages = [self._row_to_message(row) for row in rows]

            logger.debug("Found %d messages for thread_id=%s", len(messages), thread_id)
            return messages

        except Exception as e:
            logger.error(
                "Failed to list messages for thread_id=%s: %s", thread_id, describe_error(e)
            )
            raise

    async def adelete_message(
        self,
        config: dict[str, Any],
        message_id: str | int,
    ) -> int:
        """
        Delete a message by ID.

        The message is also removed from every retained state version. When that rewrites a
        state, the thread's latest state row moves to a new version, so every cached copy of
        the old state (under any key) fails the version check in ``aget_state_cache`` and is
        dropped, and a run still holding the old state gets ``StaleStateError`` on its next
        save instead of writing the deleted message back.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            int: The number of messages removed; 0 when the message does not exist or the
            caller's scope does not reach its thread.

        Raises:
            Exception: If deletion fails.
        """
        thread_id = self._message_thread_id(config)
        user_id = owner_id(config)

        # Message ids are unique per thread only, so the thread is part of the key.

        logger.debug("Deleting message_id=%s for thread_id=%s", message_id, thread_id)

        try:
            query = (
                f"DELETE FROM {self._get_table_name('messages')} "  # noqa: S608
                f"WHERE message_id = $1 AND thread_id = $2"
            )
            params: list[Any] = [message_id, thread_id]
            # Owner scoping: never let a caller delete a message on a thread they
            # do not own, even with a known message_id.
            query = self._append_owner_scope(query, params, user_id, config)
            scope_sql, scope_params = self._thread_scope(thread_id, user_id, config)
            states = self._get_table_name("states")
            threads = self._get_table_name("threads")

            async def _delete(conn):
                # Same row lock a state write takes, so a writer cannot re-save the state
                # between our read and our rewrite (or take the version we move to below).
                if self.columns.tenant:
                    # Read the thread's identity with the lock, under the caller's scope, so
                    # the cache entry dropped below is the thread's shared one.
                    row_sql, row_params = self._thread_row_scope(thread_id, user_id, config)
                    owner_row = await conn.fetchrow(
                        f"SELECT {self._identity_columns_sql()} FROM {threads} "  # noqa: S608
                        f"WHERE {row_sql} FOR UPDATE",
                        *row_params,
                    )
                    self._remember_identity(config, thread_id, owner_row)
                else:
                    await conn.execute(
                        f"SELECT 1 FROM {threads} WHERE thread_id = $1 FOR UPDATE",  # noqa: S608
                        thread_id,
                    )
                deleted = _rows_affected(await conn.execute(query, *params))
                # The message also lives inside every retained state version; leaving it
                # there brings it back on the next run and in GET state.
                rows = await conn.fetch(
                    f"SELECT state_id, version, state_data FROM {states} "  # noqa: S608
                    f"WHERE {scope_sql}",
                    *scope_params,
                )
                rewritten = False
                for row in rows:
                    payload = self._loads_jsonb(row["state_data"])
                    if self._strip_message_from_payload(payload, message_id):
                        rewritten = True
                        await conn.execute(
                            f"UPDATE {states} SET state_data = $1 WHERE state_id = $2",  # noqa: S608
                            json.dumps(payload),
                            row["state_id"],
                        )
                if rewritten:
                    # An in-place rewrite keeps MAX(version), so a cached copy of the old
                    # state (under any caller's key) would still pass the version check.
                    # Moving the latest row to a new version makes every such copy stale.
                    latest = max(rows, key=lambda r: (int(r["version"]), int(r["state_id"])))
                    await conn.execute(
                        f"UPDATE {states} SET version = $1 WHERE state_id = $2",  # noqa: S608
                        int(latest["version"]) + 1,
                        latest["state_id"],
                    )
                return deleted

            deleted = await self._run_scoped(
                _delete, in_transaction=True, settings=self._rls(config)
            )
            # The cached state carries the message too. Dropping it makes the next read
            # rebuild from Postgres.
            if user_id:
                await self._invalidate_state_cache(thread_id, user_id, retries=1, config=config)
            logger.debug("Deleted message_id=%s (%d row)", message_id, deleted)
            return deleted

        except Exception as e:
            logger.error("Failed to delete message_id=%s: %s", message_id, describe_error(e))
            raise

    async def aupdate_message_metadata(
        self,
        config: dict[str, Any],
        message_id: str | int,
        metadata: dict[str, Any],
    ) -> Message | None:
        """
        Shallow-merge ``metadata`` into a stored message's metadata.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.
            metadata (dict): Keys to set; a key replaces the same stored key.

        Returns:
            Message | None: The updated message, or None if it does not exist in this
            thread (or the caller does not own the thread: a write stays own whatever the
            reach).
        """
        thread_id = self._message_thread_id(config)
        user_id = owner_id(config)

        logger.debug("Updating metadata of message_id=%s for thread_id=%s", message_id, thread_id)

        try:
            query = f"""
                UPDATE {self._get_table_name("messages")}
                SET meta = COALESCE(meta, '{{}}'::jsonb) || $3::jsonb, updated_at = NOW()
                WHERE message_id = $1 AND thread_id = $2
            """  # noqa: S608
            params: list[Any] = [message_id, thread_id, json.dumps(metadata)]
            # Owner scoping: a thread you do not own is reported as "not found". A write, so
            # it stays own whatever the reach.
            query = self._append_owner_scope(query, params, user_id, config, write=True)
            query += """
                RETURNING message_id, thread_id, role, content, tool_calls,
                          tool_call_id, created_at, total_tokens,
                          usages, meta
            """

            row = await self._run_scoped(
                lambda conn: conn.fetchrow(query, *params),
                settings=self._rls(config, write=True),
            )
            return self._row_to_message(row) if row else None

        except Exception as e:
            logger.error(
                "Failed to update metadata of message_id=%s: %s", message_id, describe_error(e)
            )
            raise

    def _row_to_message(self, row) -> Message:  # noqa: PLR0912, PLR0915
        """
        Convert database row to Message object with robust JSON handling.

        Args:
            row: Database row.

        Returns:
            Message: Message object.
        """
        from tenxgraph.core.state.message import TokenUsages

        # Handle usages JSONB
        usages = None
        if row["usages"]:
            try:
                usages = TokenUsages(**self._loads_jsonb(row["usages"]))
            except Exception:
                usages = None

        # Handle tool_calls JSONB
        tool_calls = None
        if row["tool_calls"]:
            try:
                tool_calls = self._loads_jsonb(row["tool_calls"])
            except Exception:
                tool_calls = None

        # Handle meta JSONB
        metadata = {}
        if row["meta"]:
            try:
                metadata = self._loads_jsonb(row["meta"])
            except Exception:
                metadata = {}

        # Handle content TEXT/JSONB -> list of blocks
        content_raw = row["content"]
        content_value: list[Any] = []
        if content_raw is None:
            content_value = []
        elif isinstance(content_raw, bytes | bytearray):
            try:
                parsed = json.loads(content_raw.decode())
                if isinstance(parsed, list):
                    content_value = parsed
                elif isinstance(parsed, dict):
                    content_value = [parsed]
                else:
                    content_value = [{"type": "text", "text": str(parsed), "annotations": []}]
            except Exception:
                content_value = [
                    {
                        "type": "text",
                        "text": content_raw.decode(errors="ignore"),
                        "annotations": [],
                    }
                ]
        elif isinstance(content_raw, str):
            # Try JSON parse first
            try:
                parsed = json.loads(content_raw)
                if isinstance(parsed, list):
                    content_value = parsed
                elif isinstance(parsed, dict):
                    content_value = [parsed]
                else:
                    content_value = [{"type": "text", "text": content_raw, "annotations": []}]
            except Exception:
                content_value = [{"type": "text", "text": content_raw, "annotations": []}]
        elif isinstance(content_raw, list):
            content_value = content_raw
        elif isinstance(content_raw, dict):
            content_value = [content_raw]
        else:
            content_value = [{"type": "text", "text": str(content_raw), "annotations": []}]

        return Message(
            message_id=row["message_id"],
            role=row["role"],
            content=content_value,
            tools_calls=tool_calls,
            timestamp=row["created_at"].timestamp() if row["created_at"] else 0,
            metadata=metadata,
            usages=usages,
        )
