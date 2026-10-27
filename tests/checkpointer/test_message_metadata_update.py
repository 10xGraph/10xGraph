"""``aupdate_message_metadata`` merges keys into a stored message's metadata."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.state import Message
from tenxgraph.storage.checkpointer import BaseCheckpointer, InMemoryCheckpointer
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer


OWNER = {"thread_id": "t1", "user_id": "alice", "authz": {"scope": "owner"}}
OTHER = {"thread_id": "t1", "user_id": "bob", "authz": {"scope": "owner"}}


def _msg(mid: str, **metadata) -> Message:
    message = Message.text_message("hello", message_id=mid)
    message.metadata = metadata
    return message


@pytest.fixture(params=["memory", "sqlite"])
def cp(request):
    if request.param == "memory":
        return InMemoryCheckpointer()
    return SqliteCheckpointer(":memory:")


@pytest.mark.asyncio
async def test_merge_replaces_same_key_and_keeps_others(cp):
    await cp.aput_messages(OWNER, [_msg("m1", keep=1, feedback={"rating": "up"})])
    updated = await cp.aupdate_message_metadata(
        OWNER, "m1", {"feedback": {"rating": "down"}, "extra": True}
    )
    assert updated is not None
    assert updated.metadata == {"keep": 1, "feedback": {"rating": "down"}, "extra": True}
    stored = await cp.aget_message(OWNER, "m1")
    assert stored.metadata == updated.metadata


@pytest.mark.asyncio
async def test_missing_message_returns_none(cp):
    await cp.aput_messages(OWNER, [_msg("m1")])
    assert await cp.aupdate_message_metadata(OWNER, "nope", {"a": 1}) is None
    # Same id in a different thread is a different message.
    assert await cp.aupdate_message_metadata({"thread_id": "t2"}, "m1", {"a": 1}) is None


@pytest.mark.asyncio
async def test_non_owner_gets_none_and_nothing_changes(cp):
    await cp.aput_messages(OWNER, [_msg("m1")])
    assert await cp.aupdate_message_metadata(OTHER, "m1", {"feedback": "x"}) is None
    assert (await cp.aget_message(OWNER, "m1")).metadata == {}


@pytest.mark.asyncio
async def test_updates_only_the_message_in_its_own_thread(cp):
    await cp.aput_messages({"thread_id": "a"}, [_msg("same")])
    await cp.aput_messages({"thread_id": "b"}, [_msg("same")])
    await cp.aupdate_message_metadata({"thread_id": "a"}, "same", {"k": 1})
    assert (await cp.aget_message({"thread_id": "a"}, "same")).metadata == {"k": 1}
    assert (await cp.aget_message({"thread_id": "b"}, "same")).metadata == {}


def test_sync_wrapper():
    cp = InMemoryCheckpointer()
    cp.put_messages(OWNER, [_msg("m1")])
    assert cp.update_message_metadata(OWNER, "m1", {"a": 1}).metadata == {"a": 1}


@pytest.mark.asyncio
async def test_base_class_default_raises():
    class Minimal(InMemoryCheckpointer):
        aupdate_message_metadata = BaseCheckpointer.aupdate_message_metadata

    with pytest.raises(NotImplementedError, match="Minimal"):
        await Minimal().aupdate_message_metadata(OWNER, "m1", {})


@pytest.fixture
def pg():
    with patch("asyncpg.create_pool"):
        return PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
            user_id_type="string",
            id_type="string",
        )


def _run_with(row):
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=row)

    async def run(fn):
        return await fn(conn)

    return run, conn


@pytest.mark.asyncio
async def test_pg_issues_a_single_scoped_update(pg):
    pg._row_to_message = lambda row: "converted"
    pg._run_query, conn = _run_with({"any": "row"})
    result = await pg.aupdate_message_metadata(OWNER, "m1", {"feedback": {"rating": "up"}})
    assert result == "converted"
    sql, *params = conn.fetchrow.call_args.args
    assert "COALESCE(meta, '{}'::jsonb) || $3::jsonb" in sql
    assert "WHERE message_id = $1 AND thread_id = $2" in sql
    assert "RETURNING" in sql and sql.index("RETURNING") > sql.index("WHERE")
    assert params[:3] == ["m1", "t1", '{"feedback": {"rating": "up"}}']
    # Owner isolation adds the ownership clause before RETURNING.
    assert params[3] == "alice"
    assert sql.index("threads") < sql.index("RETURNING")


@pytest.mark.asyncio
async def test_pg_returns_none_when_no_row(pg):
    pg._run_query, _ = _run_with(None)
    assert await pg.aupdate_message_metadata({"thread_id": "t1"}, "m1", {"a": 1}) is None


@pytest.mark.asyncio
async def test_pg_requires_thread_id(pg):
    with pytest.raises(ValueError):
        await pg.aupdate_message_metadata({}, "m1", {"a": 1})
