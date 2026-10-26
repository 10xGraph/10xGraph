"""Message ids are unique per thread, so one thread cannot claim another thread's id.

The messages table is keyed ``(thread_id, message_id)``. Before schema v4 the key was
``message_id`` alone, so a user could pre-claim an id another thread later needed and make
that thread's checkpoint fail.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tenxgraph.core.state import Message
from tenxgraph.storage.checkpointer.pg_checkpointer import CURRENT_SCHEMA_VERSION, PgCheckpointer


@pytest.fixture
def checkpointer():
    with patch("asyncpg.create_pool"):
        return PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
            user_id_type="string",
            id_type="string",
        )


@pytest.mark.asyncio
async def test_upsert_conflicts_on_thread_and_message_id(checkpointer):
    conn = AsyncMock()
    await checkpointer._insert_messages(
        conn, "own-thread", [Message.text_message("hi", message_id="m1")]
    )
    sql = conn.execute.call_args.args[0]
    assert "ON CONFLICT (thread_id, message_id) DO UPDATE" in sql
    assert "ON CONFLICT (message_id)" not in sql


@pytest.mark.asyncio
async def test_rewriting_a_message_in_the_same_thread_still_works(checkpointer):
    conn = AsyncMock()
    messages = [Message.text_message("a", message_id="m1"), Message.text_message("b")]
    await checkpointer._insert_messages(conn, "t1", messages)
    assert conn.execute.await_count == 2


@pytest.mark.parametrize("id_type", ["string", "int", "bigint"])
def test_new_schema_keys_messages_per_thread(id_type):
    with patch("asyncpg.create_pool"):
        cp = PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
            id_type=id_type,
        )
    messages_sql = next(
        s for s in cp._build_create_tables_sql() if "TABLE IF NOT EXISTS" in s and '"messages"' in s
    )
    assert "PRIMARY KEY (thread_id, message_id)" in messages_sql
    assert "message_id SERIAL PRIMARY KEY" not in messages_sql
    assert "message_id" in messages_sql and "PRIMARY KEY" not in messages_sql.split("thread_id")[0]


def test_migration_v4_swaps_the_message_key(checkpointer):
    assert CURRENT_SCHEMA_VERSION == 4
    migrations = checkpointer._build_migrations()
    assert set(migrations) == {2, 3, 4}
    (statement,) = migrations[4]
    assert "DROP CONSTRAINT" in statement
    assert "ADD PRIMARY KEY (thread_id, message_id)" in statement
    # Guarded, so it does nothing on a schema that already has the two-column key.
    assert "array_length(conkey, 1) = 1" in statement
    assert '"messages"' in statement


@pytest.mark.asyncio
async def test_get_and_delete_message_filter_by_thread(checkpointer):
    cfg = {"thread_id": "t1", "user_id": "u1"}
    captured: list = []

    async def run(fn):
        conn = AsyncMock()
        conn.fetchrow.return_value = None
        await fn(conn)
        captured.append(conn)

    checkpointer._run_query = run
    await checkpointer.adelete_message(cfg, "m1")
    sql, *params = captured[0].execute.call_args.args
    assert "message_id = $1 AND thread_id = $2" in sql
    assert params[:2] == ["m1", "t1"]

    with pytest.raises(ValueError, match="not found"):
        await checkpointer.aget_message(cfg, "m1")
    sql, *params = captured[1].fetchrow.call_args.args
    assert "message_id = $1 AND thread_id = $2" in sql
    assert params[:2] == ["m1", "t1"]

    with pytest.raises(ValueError):
        await checkpointer.adelete_message({"user_id": "u1"}, "m1")
    with pytest.raises(ValueError):
        await checkpointer.aget_message({"user_id": "u1"}, "m1")


@pytest.mark.asyncio
async def test_upgrade_from_v3_runs_the_message_key_migration(checkpointer):
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"version": 3})
    await checkpointer._check_and_apply_schema_version(conn)
    executed = [call.args[0] for call in conn.execute.await_args_list]
    assert any("ADD PRIMARY KEY (thread_id, message_id)" in sql for sql in executed)
    assert not any("uq_states_thread_version" in sql for sql in executed)
