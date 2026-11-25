"""Tool execution ledger."""

import json
import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from tenxgraph.storage._pg_common import loads_jsonb
from tenxgraph.storage.columns import (
    role_bypasses_rls,
)
from tenxgraph.storage.columns import (
    rows_affected as _rows_affected,
)

from ._pg_host import PgHostBase
from .base_checkpointer import reject_unstorable


logger = logging.getLogger("tenxgraph.checkpointer.pg")


class _PgLedgerMixin(PgHostBase):
    """Tool execution ledger."""

    ##################################
    #### TOOL EXECUTION LEDGER #######
    ##################################

    async def aget_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any] | None:
        """Return the recorded result of a completed tool call, if there is one.

        Consulted before a tool runs. A hit means this exact tool call already
        completed in an earlier attempt at this node (the process died before the
        node finished, and the node is now being replayed), so the tool must NOT
        be called again -- we return what it returned last time.

        Scoped to the thread's owner when isolation is on (identity reach
        ``own``, else ``enforce_user_isolation``), like the message reads, so a caller
        naming another user's thread gets no record. With isolation off it skips the join.
        """
        thread_id, user_id = self._validate_config(config)

        # Outside the try: a scope this caller cannot satisfy raises instead of reading as
        # "no record".
        params: list[Any] = [thread_id, str(tool_call_id)]
        query = self._append_owner_scope(
            f"SELECT result FROM {self._get_table_name('tool_executions')} "  # noqa: S608
            f"WHERE thread_id = $1 AND tool_call_id = $2",
            params,
            user_id,
            config,
        )
        settings = self._rls(config)
        try:
            row = await self._run_query(
                lambda conn: conn.fetchrow(query, *params), settings=settings
            )
            if not row:
                return None

            return loads_jsonb(row["result"])

        except Exception as e:
            # A ledger read failure must not take the run down. Falling back to
            # "no record" means the tool runs again -- at-least-once, which is the
            # old behaviour, rather than a hard failure.
            logger.error(
                "Failed to read tool ledger for thread_id=%s tool_call_id=%s: %s",
                thread_id,
                tool_call_id,
                e,
            )
            return None

    async def aput_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
        result: dict[str, Any],
    ) -> Any | None:
        """Durably record that a tool call completed.

        Written as soon as the tool returns, so a crash later in the same node
        cannot cause it to be re-fired on replay. ON CONFLICT DO NOTHING keeps the
        first recorded result authoritative if this ever races with itself.

        Unlike the read side, a write failure here is raised: silently failing to
        record a completed side effect is exactly what leads to a double charge on
        the next replay, so the caller must know.
        """
        reject_unstorable(result)
        thread_id, user_id = self._validate_config(config)

        isolation = self._isolation(config, write=True)
        query = f"""
            INSERT INTO {self._get_table_name("tool_executions")}
                (thread_id, tool_call_id, result)
            VALUES ($1, $2, $3)
            ON CONFLICT (thread_id, tool_call_id) DO NOTHING
            """  # noqa: S608

        async def _record(conn) -> None:
            # One connection for the thread check and the insert. Both statements are
            # idempotent, so a retry after a lost connection redoes them safely.
            await self._ensure_thread_row(conn, thread_id, user_id, config, isolation, lock=False)
            await conn.execute(query, thread_id, str(tool_call_id), json.dumps(result))

        await self._run_query(_record, settings=isolation.settings)
        logger.debug(
            "Recorded tool execution thread_id=%s tool_call_id=%s",
            thread_id,
            tool_call_id,
        )
        return True

    async def aclear_tool_results(
        self,
        config: dict[str, Any],
        origin_message_ids: Sequence[str] | None = None,
    ) -> int:
        """Delete the thread's ledger rows (only those of ``origin_message_ids`` when given).

        A row belongs to an origin id when its key starts with ``"<id>:"``, compared with
        ``left()`` on bound values (no LIKE, so ``%`` and ``_`` in ids stay literal). Scoped
        like the ledger write: the caller's own thread (``user_id`` and every tenant column
        when isolation is on), with the RLS values set when RLS is on.
        """
        if origin_message_ids is not None and not origin_message_ids:
            return 0
        thread_id, user_id = self._validate_config(config)
        params: list[Any] = [thread_id]
        where = "thread_id = $1"
        if origin_message_ids is not None:
            params.append([str(i) for i in origin_message_ids])
            where += (
                " AND EXISTS (SELECT 1 FROM unnest($2::text[]) AS o(id) "
                "WHERE left(tool_call_id, length(o.id) + 1) = o.id || ':')"
            )
        query = self._append_owner_scope(
            f"DELETE FROM {self._get_table_name('tool_executions')} "  # noqa: S608
            f"WHERE {where}",
            params,
            user_id,
            config,
            write=True,
        )
        status = await self._run_query(
            lambda conn: conn.execute(query, *params),
            settings=self._rls(config, write=True),
        )
        return _rows_affected(status)

    async def aprune_tool_results(self, older_than: timedelta) -> int:
        """Delete ledger rows of every thread recorded more than ``older_than`` ago.

        Not scoped to a user or tenant. With ``row_level_security=True`` the rows are only
        visible to a role that bypasses RLS (superuser or ``BYPASSRLS``): any other role
        deletes nothing, and a warning says so.
        """
        query = (
            f"DELETE FROM {self._get_table_name('tool_executions')} "  # noqa: S608
            f"WHERE created_at < now() - $1::interval"
        )

        async def _prune(conn) -> Any:
            if self.row_level_security and not await role_bypasses_rls(conn):
                logger.warning(
                    "PgCheckpointer: row_level_security=True and this database role does not "
                    "bypass RLS, so aprune_tool_results sees no rows. Run it with a "
                    "maintenance role that is a superuser or has BYPASSRLS."
                )
            return await conn.execute(query, older_than)

        deleted = _rows_affected(await self._run_query(_prune))
        logger.info("Pruned %d tool ledger rows older than %s", deleted, older_than)
        return deleted
