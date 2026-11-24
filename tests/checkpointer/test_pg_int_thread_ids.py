"""The id type follows the setup: ``id_type`` text takes string ids, int and bigint take integers.

A mismatch raises ``ValueError`` before any SQL (no raw asyncpg error).
"""

from unittest.mock import AsyncMock, patch

import pytest

from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer


def _checkpointer(id_type: str) -> PgCheckpointer:
    with patch("asyncpg.create_pool"):
        return PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
            user_id_type="string",
            id_type=id_type,
        )


@pytest.mark.parametrize("id_type", ["int", "bigint"])
def test_integer_tables_take_int_ids_and_refuse_strings(id_type):
    cp = _checkpointer(id_type)
    assert cp._validate_config({"thread_id": 42, "user_id": "u"}) == (42, "u")
    with pytest.raises(ValueError, match="must be an integer"):
        cp._validate_config({"thread_id": "42", "user_id": "u"})


def test_string_tables_take_string_ids_and_refuse_ints():
    cp = _checkpointer("string")
    assert cp._validate_config({"thread_id": "42", "user_id": "u"}) == ("42", "u")
    with pytest.raises(ValueError, match="must be a string"):
        cp._validate_config({"thread_id": 42, "user_id": "u"})
    with pytest.raises(ValueError, match="message_id must be a string"):
        cp._db_message_id(5)


def test_non_integer_id_is_a_clear_error():
    with pytest.raises(ValueError, match="must be an integer"):
        _checkpointer("bigint")._validate_config({"thread_id": "abc", "user_id": "u"})
    with pytest.raises(ValueError, match="must be an integer"):
        _checkpointer("bigint")._validate_config({"thread_id": True, "user_id": "u"})


@pytest.mark.asyncio
async def test_owner_lookup_passes_an_int():
    cp = _checkpointer("bigint")
    conn = AsyncMock()
    conn.fetchval.return_value = "alice"

    async def run(fn, **_kwargs):
        return await fn(conn)

    cp._run_query = run

    assert await cp.aget_thread_owner(42) == "alice"
    assert conn.fetchval.call_args.args[1] == 42


@pytest.mark.asyncio
async def test_owner_lookup_of_a_non_integer_id_finds_nothing():
    cp = _checkpointer("bigint")
    cp._run_query = AsyncMock()
    assert await cp.aget_thread_owner("abc") is None
    cp._run_query.assert_not_awaited()


def _capture(cp: PgCheckpointer) -> list:
    """Route ``_run_query`` to a mock connection and return the connections used."""
    conns: list = []

    async def run(fn, **_kwargs):
        conn = AsyncMock()
        conn.fetchrow.return_value = None
        conn.fetch.return_value = []
        conn.execute.return_value = "DELETE 0"
        conns.append(conn)
        return await fn(conn)

    cp._run_query = run
    cp._run_query = run
    return conns


@pytest.mark.asyncio
@pytest.mark.parametrize("id_type", ["int", "bigint"])
async def test_message_methods_pass_an_int_thread_id(id_type):
    cp = _checkpointer(id_type)
    cfg = {"thread_id": 42, "user_id": "u"}
    conns = _capture(cp)

    with pytest.raises(ValueError, match="not found"):
        await cp.aget_message(cfg, 7)
    assert conns[-1].fetchrow.call_args.args[1:3] == (7, 42)

    await cp.alist_messages(cfg)
    assert conns[-1].fetch.call_args.args[1] == 42

    await cp.aupdate_message_metadata(cfg, 7, {"k": "v"})
    assert conns[-1].fetchrow.call_args.args[1:3] == (7, 42)

    await cp.adelete_message(cfg, 7)
    delete_args = next(
        c.args for c in conns[-1].execute.call_args_list if "DELETE FROM" in c.args[0]
    )
    assert delete_args[1:3] == (7, 42)


@pytest.mark.asyncio
async def test_message_methods_reject_a_non_integer_thread_id():
    cp = _checkpointer("int")
    _capture(cp)
    cfg = {"thread_id": "abc", "user_id": "u"}
    for call in (
        cp.aget_message(cfg, 1),
        cp.alist_messages(cfg),
        cp.adelete_message(cfg, 1),
        cp.aupdate_message_metadata(cfg, 1, {}),
    ):
        with pytest.raises(ValueError, match="must be an integer"):
            await call
    good = {"thread_id": 42, "user_id": "u"}
    for call in (
        cp.aget_message(good, "m1"),
        cp.adelete_message(good, "m1"),
        cp.aupdate_message_metadata(good, "m1", {}),
    ):
        with pytest.raises(ValueError, match="message_id must be an integer"):
            await call
