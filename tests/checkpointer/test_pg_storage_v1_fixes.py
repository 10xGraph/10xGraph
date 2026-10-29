"""PgCheckpointer fixes for 1.0 (mock based; real-database variants are in tests/integration).

G-42 lazy schema, G-11 delete message, G-170 stale cache, G-36 upsert, G-40 keys, G-43 timeouts.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tenxgraph.storage.checkpointer.pg_checkpointer import (
    CURRENT_SCHEMA_VERSION,
    PgCheckpointer,
)


class _Txn:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *a):
        return False


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return False


def _make(monkeypatch, **kwargs):
    monkeypatch.setattr("tenxgraph.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("tenxgraph.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)
    cp = PgCheckpointer(postgres_dsn="postgres://x", redis=AsyncMock(), **kwargs)
    conn = AsyncMock()
    conn.transaction = MagicMock(return_value=_Txn())
    conn.fetchrow = AsyncMock(return_value=None)
    cp._pg_pool = MagicMock()
    cp._pg_pool.acquire = MagicMock(return_value=_Acquire(conn))
    cp._conn = conn
    return cp, conn


@pytest.mark.real_schema_init
@pytest.mark.asyncio
async def test_first_query_creates_schema_once(monkeypatch):
    cp, conn = _make(monkeypatch)
    conn.fetchval = AsyncMock(return_value=None)
    await cp.aget_thread_owner("t1")
    await cp.aget_thread_owner("t1")
    sql = [c.args[0] for c in conn.execute.call_args_list]
    assert any("pg_advisory_xact_lock" in q for q in sql)
    assert sum(q.strip().startswith("CREATE TABLE IF NOT EXISTS \"public\".\"threads\"") for q in sql) == 1
    assert cp._schema_initialized


@pytest.mark.asyncio
async def test_schema_version_is_zero_when_table_missing(monkeypatch):
    cp, conn = _make(monkeypatch)
    conn.fetchval = AsyncMock(return_value=False)
    assert await cp.aget_schema_version() == 0
    conn.fetchval = AsyncMock(return_value=True)
    conn.fetchrow = AsyncMock(return_value={"version": CURRENT_SCHEMA_VERSION})
    assert await cp.aget_schema_version() == CURRENT_SCHEMA_VERSION


@pytest.mark.asyncio
async def test_delete_message_rewrites_every_state_version(monkeypatch):
    cp, conn = _make(monkeypatch)
    keep = {"message_id": "keep"}
    gone = {"message_id": "gone"}
    conn.fetch = AsyncMock(
        return_value=[
            {"state_id": 1, "state_data": json.dumps({"context": [keep, gone]})},
            {"state_id": 2, "state_data": json.dumps({"context": [keep]})},
        ]
    )
    await cp.adelete_message({"thread_id": "t1", "user_id": "u1"}, "gone")
    updates = [c.args for c in conn.execute.call_args_list if c.args[0].startswith("UPDATE")]
    assert len(updates) == 1 and updates[0][2] == 1
    assert json.loads(updates[0][1])["context"] == [keep]
    cp.redis.delete.assert_awaited()


@pytest.mark.asyncio
async def test_stale_cache_is_dropped_and_postgres_state_served(monkeypatch):
    cp, conn = _make(monkeypatch)
    cached = {"context": [], "__checkpoint_version__": 9}
    cp.redis.get = AsyncMock(return_value=json.dumps(cached))
    conn.fetchval = AsyncMock(return_value=12)
    conn.fetchrow = AsyncMock(return_value={"version": 12, "state_data": json.dumps({})})
    cfg = {"thread_id": "t1", "user_id": "u1"}
    await cp.aget_state_cache(cfg)
    assert cfg["_checkpoint_version"] == 12
    cp.redis.delete.assert_awaited()


@pytest.mark.asyncio
async def test_matching_cache_is_served(monkeypatch):
    cp, conn = _make(monkeypatch)
    cp.redis.get = AsyncMock(return_value=json.dumps({"context": [], "__checkpoint_version__": 4}))
    conn.fetchval = AsyncMock(return_value=4)
    cfg = {"thread_id": "t1", "user_id": "u1"}
    assert await cp.aget_state_cache(cfg) is not None
    assert cfg["_checkpoint_version"] == 4
    cp.redis.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_message_upsert_updates_tool_calls_and_meta(monkeypatch):
    from tenxgraph.core.state import Message

    cp, conn = _make(monkeypatch)
    await cp._insert_messages(conn, "t1", [Message.text_message("x", message_id="m")])
    sql = conn.execute.call_args.args[0]
    for col in ("tool_calls = EXCLUDED", "role = EXCLUDED", "meta = EXCLUDED"):
        assert col in sql


def test_redis_keys_are_prefixed_and_unambiguous(monkeypatch):
    cp, _ = _make(monkeypatch, redis_key_prefix="prod:")
    a = cp._get_thread_key("a:b", "c")
    b = cp._get_thread_key("a", "b:c")
    assert a != b and a.startswith("prod:state_cache:")


@pytest.mark.asyncio
async def test_clean_thread_survives_redis_failure(monkeypatch):
    cp, conn = _make(monkeypatch)
    cp.redis.delete = AsyncMock(side_effect=ConnectionError("down"))
    await cp.aclean_thread({"thread_id": "t1", "user_id": "u1"})


def test_timeouts_default_and_override(monkeypatch):
    cp, _ = _make(monkeypatch)
    assert cp.pg_acquire_timeout == 30.0
    from unittest.mock import patch

    with patch("asyncpg.create_pool") as create:
        cp._create_pg_pool(None, "postgres://u", {"command_timeout": 5})
    assert create.call_args.kwargs["command_timeout"] == 5
    assert create.call_args.kwargs["timeout"] == 30.0
