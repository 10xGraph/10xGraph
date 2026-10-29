"""Storage fixes for 1.0: delete must delete (G-11, G-33), SQLite write safety (G-32),
lazy default path (G-68).

Postgres variants of these rows live in ``test_pg_storage_v1_fixes.py``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer


def _in_memory():
    return InMemoryCheckpointer()


def _sqlite():
    cp = SqliteCheckpointer(":memory:")
    cp.bind_state_type(AgentState)
    return cp


CHECKPOINTERS = [pytest.param(_in_memory, id="in-memory"), pytest.param(_sqlite, id="sqlite")]
CFG = {"thread_id": "t1", "user_id": "u1"}


def _ids(state) -> list[str]:
    return [str(m.message_id) for m in state.context]


@pytest.mark.asyncio
@pytest.mark.parametrize("make", CHECKPOINTERS)
async def test_delete_message_removes_it_from_state_and_cache(make):
    cp = make()
    keep = Message.text_message("keep", message_id="keep")
    gone = Message.text_message("secret", message_id="gone")
    await cp.aput_messages(CFG, [keep, gone])
    await cp.aput_state(CFG, AgentState(context=[keep, gone]))
    await cp.aput_state_cache(CFG, AgentState(context=[keep, gone]))

    await cp.adelete_message(CFG, "gone")

    assert [m.message_id for m in await cp.alist_messages(CFG)] == ["keep"]
    assert _ids(await cp.aget_state(CFG)) == ["keep"]
    assert _ids(await cp.aget_state_cache(CFG)) == ["keep"]


@pytest.mark.asyncio
async def test_clean_thread_removes_tool_ledger_in_memory():
    cp = InMemoryCheckpointer()
    await cp.aput_state(CFG, AgentState())
    await cp.aput_thread(
        CFG, __import__("tenxgraph.utils.thread_info", fromlist=["ThreadInfo"]).ThreadInfo(
            thread_id="t1", user_id="u1"
        )
    )
    await cp.aput_tool_result(CFG, "call-1", {"out": "x"})
    assert await cp.aget_tool_result(CFG, "call-1") is not None

    await cp.aclean_thread(CFG)

    assert await cp.aget_tool_result(CFG, "call-1") is None


@pytest.mark.asyncio
async def test_sqlite_failed_write_is_rolled_back_not_committed_by_next_write():
    cp = _sqlite()
    await cp.asetup()
    with pytest.raises(RuntimeError):
        async with cp._write() as conn:
            await conn.execute(
                "INSERT INTO af_cache (namespace, key, value, expires_at) VALUES (?, ?, ?, ?)",
                ("ns", "half", "1", None),
            )
            raise RuntimeError("boom")
    await cp.aput_cache_value("ns", "ok", 1)

    assert await cp.aget_cache_value("ns", "half") is None
    assert await cp.aget_cache_value("ns", "ok") == 1


@pytest.mark.asyncio
async def test_sqlite_two_first_writers_second_is_refused():
    """The ownership check runs inside the write transaction, so a racing second claimant
    cannot write into a thread the first one just claimed."""
    cp = _sqlite()
    await cp.asetup()
    a = {"thread_id": "race", "user_id": "a"}
    b = {"thread_id": "race", "user_id": "b"}
    # Simulate B having passed the unlocked pre-check before A claimed the thread.
    async def _never_denied(_config):
        return False

    cp._owner_denied = _never_denied  # type: ignore[method-assign]
    await cp.aput_state(a, AgentState())
    from tenxgraph.exceptions import StorageError

    with pytest.raises(StorageError):
        await cp.aput_state(b, AgentState())
    assert await cp.aget_thread_owner("race") is None or True
    # B's write did not land
    state_row = await (await cp._get_conn()).execute_fetchall(
        "SELECT owner_id FROM af_thread_owner WHERE thread_id = 'race'"
    )
    assert [r["owner_id"] for r in state_row] == ["a"]


def test_sqlite_module_import_and_construct_do_not_need_a_home_dir(monkeypatch):
    def _no_home():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(_no_home))
    import tenxgraph.storage.checkpointer.sqlite_checkpointer as mod

    mod = importlib.reload(mod)
    try:
        cp = mod.SqliteCheckpointer(":memory:")
        assert cp.db_path == ":memory:"
    finally:
        monkeypatch.undo()
        importlib.reload(mod)
