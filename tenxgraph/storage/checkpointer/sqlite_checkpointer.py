"""SQLite-backed checkpointer for client-side and single-user agent workloads.

This module provides :class:`SqliteCheckpointer`, a :class:`BaseCheckpointer`
implementation that keeps *everything* — durable state, the hot "realtime" state
cache, a generic TTL cache, messages, and threads — in a single local SQLite
database file. No external services (Postgres, Redis) are required.

It is the right choice for:

- Client-side / embedded agents: a desktop app (Tauri, Electron, PyInstaller)
  that ships a Python sidecar, or a local CLI agent.
- Dedicated-room deployments where each user has their own process and their own
  database file.

It is **not** the right choice for multi-user servers where many users share one
backend. SQLite serializes writers and does not scale horizontally; use
:class:`~tenxgraph.storage.checkpointer.pg_checkpointer.PgCheckpointer` there.

Install with::

    pip install 10xgraph[sqlite_checkpoint]
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from tenxgraph.core.exceptions import StorageError
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils.callable_utils import run_coroutine
from tenxgraph.utils.private_fs import mkdir_private, touch_private
from tenxgraph.utils.thread_info import ThreadInfo

from .base_checkpointer import BaseCheckpointer, validate_page


try:
    import aiosqlite

    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False
    aiosqlite = None  # type: ignore[assignment]

logger = logging.getLogger("tenxgraph.checkpointer.sqlite")

StateT = TypeVar("StateT", bound="AgentState")


# Default database location for desktop / single-user agents.
def _default_home_dir() -> Path:
    """Return ``~/.10xgraph``, falling back to a pre-existing legacy ``~/.agentflow``."""
    home = Path.home()
    new_dir = home / ".10xgraph"
    legacy_dir = home / ".agentflow"
    if not new_dir.exists() and legacy_dir.exists():
        return legacy_dir
    return new_dir


# Left unset: the home directory is looked up when a checkpointer is built, not at import, so
# ``import tenxgraph`` works where there is no HOME and no passwd entry. Set it to override.
DEFAULT_DB_PATH: str | None = None


def _default_db_path() -> str:
    return str(_default_home_dir() / "checkpointer.db")


DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS af_states (
        thread_id TEXT NOT NULL PRIMARY KEY,
        state_data TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS af_state_cache (
        thread_id TEXT NOT NULL PRIMARY KEY,
        state_data TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS af_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT NOT NULL,
        message_id TEXT NOT NULL,
        message_data TEXT NOT NULL,
        created_at REAL NOT NULL,
        UNIQUE (thread_id, message_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_af_messages_thread ON af_messages (thread_id)",
    """
    CREATE TABLE IF NOT EXISTS af_threads (
        thread_id TEXT NOT NULL PRIMARY KEY,
        thread_data TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS af_cache (
        namespace TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        expires_at REAL,
        PRIMARY KEY (namespace, key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_af_cache_namespace ON af_cache (namespace)",
    # Thread ownership: first writer of a thread owns it. Used to enforce owner-only
    # access when the request policy (config["authz"]) asks for it.
    """
    CREATE TABLE IF NOT EXISTS af_thread_owner (
        thread_id TEXT NOT NULL PRIMARY KEY,
        owner_id  TEXT NOT NULL
    )
    """,
    # Tool execution ledger: a completed tool call is recorded against its id so a node
    # replayed after a crash does not fire it again.
    """
    CREATE TABLE IF NOT EXISTS af_tool_executions (
        thread_id TEXT NOT NULL,
        tool_call_id TEXT NOT NULL,
        result TEXT NOT NULL,
        PRIMARY KEY (thread_id, tool_call_id)
    )
    """,
)


def _enum_handler(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _dumps(data: Any) -> str:
    return json.dumps(data, default=_enum_handler)


class SqliteCheckpointer(BaseCheckpointer[StateT]):
    """SQLite-backed checkpointer that stores all data in one local ``.db`` file.

    Everything a graph needs to persist — durable state, the realtime state
    cache, generic TTL cache, messages, and threads — lives in a single SQLite
    file accessed through ``aiosqlite`` (fully async). A single persistent
    connection is opened lazily; WAL journal mode is enabled for better
    read concurrency and writes are serialized behind an ``asyncio.Lock``.

    Rows store state data only. On read they are rebuilt into the state class
    bound by ``StateGraph.compile()`` (see :meth:`BaseCheckpointer.bind_state_type`),
    matching :class:`~tenxgraph.storage.checkpointer.pg_checkpointer.PgCheckpointer`.

    Args:
        db_path: Path to the SQLite database file. Defaults to
            ``~/.10xgraph/checkpointer.db`` (or ``~/.agentflow/checkpointer.db``
            when only the legacy directory exists). Parent directories are created on
            setup. Use ``":memory:"`` for an ephemeral in-process database
            (useful for tests).

    Raises:
        ImportError: If ``aiosqlite`` is not installed.

    Example::

        checkpointer = SqliteCheckpointer("agent.db")
        graph = builder.compile(checkpointer=checkpointer)
        await graph.ainvoke({"messages": [...]}, config={"thread_id": "t1"})
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if not HAS_AIOSQLITE:
            raise ImportError(
                "SqliteCheckpointer requires 'aiosqlite'. "
                "Install with: pip install 10xgraph[sqlite_checkpoint]"
            )

        if db_path == ":memory:":
            self.db_path: str | Path = ":memory:"
        else:
            self.db_path = Path(db_path) if db_path else Path(DEFAULT_DB_PATH or _default_db_path())

        self._conn: aiosqlite.Connection | None = None
        self._setup_done = False
        self._setup_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_config_key(self, config: dict[str, Any]) -> str:
        return str(config.get("thread_id", ""))

    async def _record_owner(self, conn: "aiosqlite.Connection", config: dict[str, Any]) -> bool:
        """Record the caller as the thread's owner on first write (needs a user_id).

        Runs inside the write transaction and reads the owner back, so two first writers that
        both passed the unlocked pre-check cannot both think they own the thread. Returns
        False when isolation is active and another user owns it.
        """
        user_id = config.get("user_id") if isinstance(config, dict) else None
        if not user_id:
            return True
        key = self._get_config_key(config)
        await conn.execute(
            "INSERT OR IGNORE INTO af_thread_owner (thread_id, owner_id) VALUES (?, ?)",
            (key, str(user_id)),
        )
        if not self._isolation_enabled(config, user_id):
            return True
        async with conn.execute(
            "SELECT owner_id FROM af_thread_owner WHERE thread_id = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
        return row is None or str(row["owner_id"]) == str(user_id)

    async def _owner_denied(self, config: dict[str, Any]) -> bool:
        """True when isolation is active and the caller does not own this thread."""
        user_id = config.get("user_id") if isinstance(config, dict) else None
        if not self._isolation_enabled(config, user_id):
            return False
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT owner_id FROM af_thread_owner WHERE thread_id = ?",
            (self._get_config_key(config),),
        ) as cur:
            row = await cur.fetchone()
        return row is not None and str(row["owner_id"]) != str(user_id)

    def _serialize_state(self, state: StateT) -> str:
        return _dumps(self._encode_state(state))

    def _deserialize_state(self, raw: str, thread_id: Any = None) -> StateT:
        return self._decode_state(json.loads(raw), thread_id)

    async def _ensure_setup(self) -> None:
        if self._setup_done:
            return
        async with self._setup_lock:
            if not self._setup_done:
                await self.asetup()

    async def _get_conn(self) -> "aiosqlite.Connection":
        """Return the shared connection, opening and configuring it if needed."""
        if self._conn is None:
            self._prepare_db_file()
            conn = await aiosqlite.connect(self.db_path)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def _prepare_db_file(self) -> None:
        """Create the db directory (0o700) and an empty db file (0o600) before SQLite does.

        SQLite gives the ``-wal`` and ``-shm`` files the mode of the database file, so they
        are owner-only too. An existing directory or file keeps its mode.
        """
        if isinstance(self.db_path, Path):
            mkdir_private(self.db_path.parent)
            touch_private(self.db_path)

    @asynccontextmanager
    async def _write(self) -> AsyncIterator["aiosqlite.Connection"]:
        """Serialize a write transaction and commit on success."""
        conn = await self._get_conn()
        async with self._write_lock:
            try:
                yield conn
            except BaseException:
                # Without this the half-done statements stay pending and the next
                # successful write commits them.
                await conn.rollback()
                raise
            await conn.commit()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        return run_coroutine(self.asetup())

    async def asetup(self) -> None:
        """Create the database directory (if any) and all required tables."""
        conn = await self._get_conn()
        async with self._write_lock:
            for stmt in DDL_STATEMENTS:
                await conn.execute(stmt)
            await conn.commit()
        self._setup_done = True
        logger.info("SqliteCheckpointer setup complete: %s", self.db_path)

    # ------------------------------------------------------------------
    # State — async
    # ------------------------------------------------------------------

    async def aput_state(self, config: dict[str, Any], state: StateT) -> StateT:
        await self._ensure_setup()
        if await self._owner_denied(config):
            raise StorageError("Thread is owned by a different user")
        key = self._get_config_key(config)
        data = self._serialize_state(state)
        async with self._write() as conn:
            if not await self._record_owner(conn, config):
                raise StorageError("Thread is owned by a different user")
            await conn.execute(
                """
                INSERT INTO af_states (thread_id, state_data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    state_data = excluded.state_data,
                    updated_at = excluded.updated_at
                """,
                (key, data, time.time()),
            )
            # Write-through, in the same transaction: a run reads the cache first, so
            # leaving it behind would start the next run from the replaced state.
            await self._upsert_state_cache(conn, key, data)
        logger.debug("Stored state for thread: %s", key)
        return state

    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        await self._ensure_setup()
        if await self._owner_denied(config):
            return None
        key = self._get_config_key(config)
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT state_data FROM af_states WHERE thread_id = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._deserialize_state(row["state_data"], config.get("thread_id"))

    async def aclear_state(self, config: dict[str, Any]) -> bool:
        await self._ensure_setup()
        if await self._owner_denied(config):
            return True  # no-op, do not reveal existence
        key = self._get_config_key(config)
        async with self._write() as conn:
            await conn.execute("DELETE FROM af_states WHERE thread_id = ?", (key,))
            # A run reads the cache first, so a cached copy would bring the state back.
            await conn.execute("DELETE FROM af_state_cache WHERE thread_id = ?", (key,))
        return True

    # ------------------------------------------------------------------
    # State cache — async (hot layer, same file, no Redis)
    # ------------------------------------------------------------------

    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> StateT:
        await self._ensure_setup()
        # Same rule as aput_state: a non-owner's write is dropped, and the first writer of a
        # thread becomes its owner, so a cached state can only be read back by its writer.
        if await self._owner_denied(config):
            return state
        key = self._get_config_key(config)
        data = self._serialize_state(state)
        async with self._write() as conn:
            if not await self._record_owner(conn, config):
                return state
            await self._upsert_state_cache(conn, key, data)
        return state

    @staticmethod
    async def _upsert_state_cache(conn: Any, key: str, data: str) -> None:
        """Write one thread's cached state on an open connection."""
        await conn.execute(
            """
            INSERT INTO af_state_cache (thread_id, state_data, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                state_data = excluded.state_data,
                updated_at = excluded.updated_at
            """,
            (key, data, time.time()),
        )

    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        await self._ensure_setup()
        if await self._owner_denied(config):
            return None
        key = self._get_config_key(config)
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT state_data FROM af_state_cache WHERE thread_id = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._deserialize_state(row["state_data"], config.get("thread_id"))

    async def aclear_state_cache(self, config: dict[str, Any]) -> bool:
        await self._ensure_setup()
        if await self._owner_denied(config):
            return True  # no-op, do not reveal existence
        key = self._get_config_key(config)
        async with self._write() as conn:
            await conn.execute("DELETE FROM af_state_cache WHERE thread_id = ?", (key,))
        return True

    # ------------------------------------------------------------------
    # Generic cache — async (TTL, lazy prune on read)
    # ------------------------------------------------------------------

    async def aput_cache_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> Any | None:
        await self._ensure_setup()
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        async with self._write() as conn:
            await conn.execute(
                """
                INSERT INTO af_cache (namespace, key, value, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at
                """,
                (namespace, key, _dumps(value), expires_at),
            )
        return value

    async def aget_cache_value(self, namespace: str, key: str) -> Any | None:
        await self._ensure_setup()
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT value, expires_at FROM af_cache WHERE namespace = ? AND key = ?",
            (namespace, key),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        expires_at = row["expires_at"]
        if expires_at is not None and expires_at <= time.time():
            async with self._write() as conn:
                await conn.execute(
                    "DELETE FROM af_cache WHERE namespace = ? AND key = ?",
                    (namespace, key),
                )
            return None
        return json.loads(row["value"])

    async def aclear_cache_value(self, namespace: str, key: str) -> Any | None:
        await self._ensure_setup()
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT value FROM af_cache WHERE namespace = ? AND key = ?",
            (namespace, key),
        ) as cursor:
            row = await cursor.fetchone()
        async with self._write() as conn:
            await conn.execute(
                "DELETE FROM af_cache WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
        return json.loads(row["value"]) if row else None

    async def aclear_cache_value_if(self, namespace: str, key: str, expected: Any) -> bool:
        """Compare and delete in one statement."""
        await self._ensure_setup()
        async with self._write() as conn:
            cursor = await conn.execute(
                "DELETE FROM af_cache WHERE namespace = ? AND key = ? AND value = ?",
                (namespace, key, _dumps(expected)),
            )
            return cursor.rowcount > 0

    async def aput_cache_value_if_absent(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Insert unless a live value exists, in one statement; an expired row is replaced."""
        await self._ensure_setup()
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds else None
        async with self._write() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO af_cache (namespace, key, value, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at
                WHERE af_cache.expires_at IS NOT NULL AND af_cache.expires_at <= ?
                """,
                (namespace, key, _dumps(value), expires_at, now),
            )
            return cursor.rowcount > 0

    async def alist_cache_keys(
        self,
        namespace: str,
        prefix: str | None = None,
    ) -> list[str]:
        await self._ensure_setup()
        async with self._write() as conn:
            await conn.execute(
                "DELETE FROM af_cache WHERE namespace = ? AND expires_at IS NOT NULL "
                "AND expires_at <= ?",
                (namespace, time.time()),
            )
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT key FROM af_cache WHERE namespace = ?", (namespace,)
        ) as cursor:
            rows = await cursor.fetchall()
        keys = [row["key"] for row in rows]
        if prefix is not None:
            keys = [k for k in keys if k.startswith(prefix)]
        return keys

    # ------------------------------------------------------------------
    # Messages — async
    # ------------------------------------------------------------------

    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        await self._ensure_setup()
        if await self._owner_denied(config):
            raise StorageError("Thread is owned by a different user")
        key = self._get_config_key(config)
        now = time.time()
        async with self._write() as conn:
            if not await self._record_owner(conn, config):
                raise StorageError("Thread is owned by a different user")
            for msg in messages:
                await conn.execute(
                    """
                    INSERT INTO af_messages (thread_id, message_id, message_data, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(thread_id, message_id) DO UPDATE SET
                        message_data = excluded.message_data
                    """,
                    (key, str(msg.message_id), _dumps(msg.model_dump(mode="json")), now),
                )
        logger.debug("Stored %d messages for thread: %s", len(messages), key)
        return True

    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        await self._ensure_setup()
        key = self._get_config_key(config)
        if await self._owner_denied(config):
            raise IndexError(f"Message with ID {message_id} not found for thread: {key}")
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT message_data FROM af_messages WHERE thread_id = ? AND message_id = ?",
            (key, str(message_id)),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise IndexError(f"Message with ID {message_id} not found for thread: {key}")
        return Message.model_validate(json.loads(row["message_data"]))

    async def aupdate_message_metadata(
        self,
        config: dict[str, Any],
        message_id: str | int,
        metadata: dict[str, Any],
    ) -> Message | None:
        await self._ensure_setup()
        if await self._owner_denied(config):
            return None
        key = self._get_config_key(config)
        # Read and write under the write lock so two updates cannot lose each other's keys.
        async with self._write() as conn:
            async with conn.execute(
                "SELECT message_data FROM af_messages WHERE thread_id = ? AND message_id = ?",
                (key, str(message_id)),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None
            message = Message.model_validate(json.loads(row["message_data"]))
            message.metadata = {**(message.metadata or {}), **metadata}
            await conn.execute(
                "UPDATE af_messages SET message_data = ? WHERE thread_id = ? AND message_id = ?",
                (_dumps(message.model_dump(mode="json")), key, str(message_id)),
            )
        return message

    async def alist_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        start, limit = validate_page(offset, limit)
        await self._ensure_setup()
        if await self._owner_denied(config):
            return []
        key = self._get_config_key(config)
        conn = await self._get_conn()
        sql = "SELECT message_data FROM af_messages WHERE thread_id = ? ORDER BY id"
        params: list[Any] = [key]
        if not search:
            # No text filter: the window is applied by SQLite, so only it is read.
            sql += " LIMIT ? OFFSET ?"
            params += [limit if limit else -1, start]
        needle = search.lower() if search else None
        messages: list[Message] = []
        skipped = 0
        async with conn.execute(sql, params) as cursor:
            async for row in cursor:
                msg = Message.model_validate(json.loads(row["message_data"]))
                if needle is not None:
                    if not (hasattr(msg, "content") and needle in str(msg.content).lower()):
                        continue
                    # The match depends on the decoded message, so skip and stop here.
                    if skipped < start:
                        skipped += 1
                        continue
                messages.append(msg)
                if needle is not None and limit and len(messages) >= limit:
                    break
        return messages

    async def adelete_message(self, config: dict[str, Any], message_id: str | int) -> bool:
        await self._ensure_setup()
        key = self._get_config_key(config)
        if await self._owner_denied(config):
            raise IndexError(f"Message with ID {message_id} not found for thread: {key}")
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT id FROM af_messages WHERE thread_id = ? AND message_id = ?",
            (key, str(message_id)),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise IndexError(f"Message with ID {message_id} not found for thread: {key}")
        async with self._write() as conn:
            await conn.execute(
                "DELETE FROM af_messages WHERE thread_id = ? AND message_id = ?",
                (key, str(message_id)),
            )
            # The state and its cache copy carry the message too; remove it there in the
            # same transaction so a deleted message cannot come back on the next run.
            for table in ("af_states", "af_state_cache"):
                async with conn.execute(
                    f"SELECT state_data FROM {table} WHERE thread_id = ?",  # noqa: S608
                    (key,),
                ) as cursor:
                    state_row = await cursor.fetchone()
                if state_row is None:
                    continue
                payload = json.loads(state_row["state_data"])
                if self._strip_message_from_payload(payload, message_id):
                    await conn.execute(
                        f"UPDATE {table} SET state_data = ? WHERE thread_id = ?",  # noqa: S608
                        (_dumps(payload), key),
                    )
        return True

    # ------------------------------------------------------------------
    # Threads — async
    # ------------------------------------------------------------------

    async def aput_thread(self, config: dict[str, Any], thread_info: ThreadInfo) -> bool:
        await self._ensure_setup()
        key = self._get_config_key(config) or str(thread_info.thread_id)
        if str(thread_info.thread_id) != key:
            raise ValueError(
                "ThreadInfo.thread_id must match config['thread_id']: "
                f"{thread_info.thread_id!r} != {key!r}"
            )
        if await self._owner_denied(config):
            raise StorageError("Thread is owned by a different user")
        data = _dumps(thread_info.model_dump(mode="json"))
        async with self._write() as conn:
            if not await self._record_owner(conn, config):
                raise StorageError("Thread is owned by a different user")
            await conn.execute(
                """
                INSERT INTO af_threads (thread_id, thread_data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    thread_data = excluded.thread_data,
                    updated_at = excluded.updated_at
                """,
                (key, data, time.time()),
            )
        return True

    async def aget_thread(self, config: dict[str, Any]) -> ThreadInfo | None:
        await self._ensure_setup()
        if await self._owner_denied(config):
            return None
        key = self._get_config_key(config)
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT thread_data FROM af_threads WHERE thread_id = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return ThreadInfo.model_validate(json.loads(row["thread_data"]))

    async def aget_thread_owner(self, thread_id: str | int) -> str | int | None:
        """Return the owning ``user_id`` for ``thread_id`` (global, not owner-scoped)."""
        await self._ensure_setup()
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT thread_data FROM af_threads WHERE thread_id = ?", (str(thread_id),)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row["thread_data"]).get("user_id")

    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        start, limit = validate_page(offset, limit)
        await self._ensure_setup()
        user_id = config.get("user_id") if isinstance(config, dict) else None
        isolate = self._isolation_enabled(config, user_id)
        conn = await self._get_conn()
        sql = "SELECT thread_data FROM af_threads"
        params: list[Any] = []
        # Owner-scope the listing when the policy asks for it.
        if isolate:
            sql += " WHERE CAST(json_extract(thread_data, '$.user_id') AS TEXT) = ?"
            params.append(str(user_id))
        sql += " ORDER BY updated_at DESC"
        if not search:
            # No text filter: the window is applied by SQLite, so only it is read.
            sql += " LIMIT ? OFFSET ?"
            params += [limit if limit else -1, start]
        needle = search.lower() if search else None
        threads: list[ThreadInfo] = []
        skipped = 0
        async with conn.execute(sql, params) as cursor:
            async for row in cursor:
                thread = ThreadInfo.model_validate(json.loads(row["thread_data"]))
                if needle is not None:
                    if not any(needle in str(v).lower() for v in thread.model_dump().values()):
                        continue
                    # The match depends on the decoded thread, so skip and stop here.
                    if skipped < start:
                        skipped += 1
                        continue
                threads.append(thread)
                if needle is not None and limit and len(threads) >= limit:
                    break
        return threads

    async def aclean_thread(self, config: dict[str, Any]) -> bool:
        await self._ensure_setup()
        if await self._owner_denied(config):
            return False  # not the owner -> no-op
        key = self._get_config_key(config)
        async with self._write() as conn:
            await conn.execute("DELETE FROM af_threads WHERE thread_id = ?", (key,))
            await conn.execute("DELETE FROM af_messages WHERE thread_id = ?", (key,))
            await conn.execute("DELETE FROM af_states WHERE thread_id = ?", (key,))
            await conn.execute("DELETE FROM af_state_cache WHERE thread_id = ?", (key,))
            await conn.execute("DELETE FROM af_tool_executions WHERE thread_id = ?", (key,))
            await conn.execute("DELETE FROM af_thread_owner WHERE thread_id = ?", (key,))
        return True

    # ------------------------------------------------------------------
    # Tool execution ledger
    # ------------------------------------------------------------------

    async def aget_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any] | None:
        """Return the recorded result of a completed tool call, if there is one."""
        await self._ensure_setup()
        if await self._owner_denied(config):
            return None  # not the owner -> no record, do not reveal existence
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT result FROM af_tool_executions WHERE thread_id = ? AND tool_call_id = ?",
            (self._get_config_key(config), str(tool_call_id)),
        ) as cur:
            row = await cur.fetchone()
        return json.loads(row["result"]) if row else None

    async def aput_tool_result(
        self,
        config: dict[str, Any],
        tool_call_id: str,
        result: dict[str, Any],
    ) -> Any | None:
        """Durably record that a tool call completed; the first recorded result wins."""
        await self._ensure_setup()
        async with self._write() as conn:
            if not await self._record_owner(conn, config):
                raise StorageError("Thread is owned by a different user")
            await conn.execute(
                "INSERT OR IGNORE INTO af_tool_executions (thread_id, tool_call_id, result) "
                "VALUES (?, ?, ?)",
                (self._get_config_key(config), str(tool_call_id), _dumps(result)),
            )
        return True

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    async def arelease(self) -> bool:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._setup_done = False
        logger.info("SqliteCheckpointer released: %s", self.db_path)
        return True

    def release(self) -> bool:
        return run_coroutine(self.arelease())
