"""A message upsert must never modify a message that lives in another thread.

``message_id`` is the table's primary key, so ``ON CONFLICT (message_id)`` also fires for a
message in a different thread. Without a thread guard, writing a message into your own thread
with another thread's ``message_id`` rewrote that other thread's message.
"""

from unittest.mock import AsyncMock, patch

import pytest

from agentflow.core.exceptions import StorageError
from agentflow.core.state import Message
from agentflow.storage.checkpointer.pg_checkpointer import PgCheckpointer


@pytest.fixture
def checkpointer():
    with patch("asyncpg.create_pool"):
        return PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
            user_id_type="string",
            id_type="string",
        )


def _conn(status: str) -> AsyncMock:
    conn = AsyncMock()
    conn.execute.return_value = status
    return conn


@pytest.mark.asyncio
async def test_upsert_only_updates_rows_in_the_same_thread(checkpointer):
    conn = _conn("INSERT 0 1")
    await checkpointer._insert_messages(
        conn, "own-thread", [Message.text_message("hi", message_id="m1")]
    )
    sql = conn.execute.call_args.args[0]
    assert "ON CONFLICT (message_id) DO UPDATE" in sql
    assert ".thread_id = EXCLUDED.thread_id" in sql


@pytest.mark.asyncio
async def test_message_id_from_another_thread_is_rejected(checkpointer):
    # The guarded upsert affects no row when the id already belongs to another thread.
    conn = _conn("INSERT 0 0")
    with pytest.raises(StorageError) as exc:
        await checkpointer._insert_messages(
            conn, "own-thread", [Message.text_message("forged", message_id="victim-msg")]
        )
    assert exc.value.error_code == "STORAGE_FORBIDDEN_002"


@pytest.mark.asyncio
async def test_rewriting_a_message_in_the_same_thread_still_works(checkpointer):
    # Re-saving a message (same id, same thread) updates it: 1 row affected.
    conn = _conn("INSERT 0 1")
    messages = [Message.text_message("a", message_id="m1"), Message.text_message("b")]
    await checkpointer._insert_messages(conn, "t1", messages)
    assert conn.execute.await_count == 2
