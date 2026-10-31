"""Tool execution ledger on SQLite, and the one-time warning when it is unsupported (G-90)."""

import logging

import pytest

from tenxgraph.core.state import AgentState
from tenxgraph.storage.checkpointer import base_checkpointer
from tenxgraph.storage.checkpointer.base_checkpointer import BaseCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer


@pytest.fixture
def cp():
    c = SqliteCheckpointer(":memory:")
    c.bind_state_type(AgentState)
    return c


@pytest.mark.asyncio
async def test_sqlite_records_and_returns_tool_result(cp):
    cfg = {"thread_id": "t1", "user_id": "u1"}
    assert await cp.aget_tool_result(cfg, "call-1") is None
    await cp.aput_tool_result(cfg, "call-1", {"output": "charged", "is_error": False})
    assert await cp.aget_tool_result(cfg, "call-1") == {"output": "charged", "is_error": False}
    assert await cp.aget_tool_result(cfg, "call-2") is None
    assert await cp.aget_tool_result({"thread_id": "t2", "user_id": "u1"}, "call-1") is None


@pytest.mark.asyncio
async def test_sqlite_first_recorded_result_wins(cp):
    cfg = {"thread_id": "t1", "user_id": "u1"}
    await cp.aput_tool_result(cfg, "call-1", {"output": "first"})
    await cp.aput_tool_result(cfg, "call-1", {"output": "second"})
    assert await cp.aget_tool_result(cfg, "call-1") == {"output": "first"}


@pytest.mark.asyncio
async def test_sqlite_clean_thread_drops_ledger(cp):
    cfg = {"thread_id": "t1", "user_id": "u1"}
    await cp.aput_tool_result(cfg, "call-1", {"output": "x"})
    await cp.aclean_thread(cfg)
    assert await cp.aget_tool_result(cfg, "call-1") is None


@pytest.mark.asyncio
async def test_sqlite_ledger_survives_release(tmp_path):
    path = tmp_path / "cp.db"
    cfg = {"thread_id": "t1", "user_id": "u1"}
    first = SqliteCheckpointer(path)
    await first.aput_tool_result(cfg, "call-1", {"output": "x"})
    await first.arelease()
    second = SqliteCheckpointer(path)
    assert await second.aget_tool_result(cfg, "call-1") == {"output": "x"}
    await second.arelease()


class NoLedgerCheckpointer(BaseCheckpointer):
    """Does not override the ledger methods."""

    async def aping(self) -> None:  # pragma: no cover
        return

    async def aput_state(self, config, state):  # pragma: no cover
        return state

    async def aget_state(self, config):  # pragma: no cover
        return None

    async def aclear_state(self, config):  # pragma: no cover
        return True

    async def aput_state_cache(self, config, state):  # pragma: no cover
        return state

    async def aget_state_cache(self, config):  # pragma: no cover
        return None

    async def aput_messages(self, config, messages, metadata=None):  # pragma: no cover
        return True

    async def aget_message(self, config, message_id):  # pragma: no cover
        return None

    async def alist_messages(
        self, config, search=None, offset=None, limit=None
    ):  # pragma: no cover
        return []

    async def adelete_message(self, config, message_id):  # pragma: no cover
        return True

    async def aput_thread(self, config, thread_info):  # pragma: no cover
        return True

    async def aget_thread(self, config):  # pragma: no cover
        return None

    async def alist_threads(self, config, search=None, offset=None, limit=None):  # pragma: no cover
        return []

    async def aclean_thread(self, config):  # pragma: no cover
        return True

    async def asetup(self):  # pragma: no cover
        return True

    async def arelease(self):  # pragma: no cover
        return True


@pytest.mark.asyncio
async def test_unsupported_ledger_warns_once(caplog):
    base_checkpointer._LEDGER_WARNED.clear()
    cp = NoLedgerCheckpointer()
    with caplog.at_level(logging.WARNING, logger="tenxgraph.checkpointer"):
        assert await cp.aget_tool_result({"thread_id": "t"}, "c1") is None
        assert await cp.aput_tool_result({"thread_id": "t"}, "c1", {}) is None
        assert await cp.aget_tool_result({"thread_id": "t"}, "c2") is None
    warnings = [r for r in caplog.records if "tool execution ledger" in r.getMessage().lower()]
    assert len(warnings) == 1
    assert "NoLedgerCheckpointer" in warnings[0].getMessage()


# --- G-90: in-memory has a ledger; G-30: it is scoped to the thread owner ---------------


@pytest.mark.asyncio
async def test_in_memory_has_a_ledger_and_does_not_warn(caplog):
    from tenxgraph.storage.checkpointer.in_memory_checkpointer import InMemoryCheckpointer

    mem = InMemoryCheckpointer()
    cfg = {"thread_id": "t1", "user_id": "u1"}
    with caplog.at_level(logging.WARNING):
        await mem.aput_tool_result(cfg, "c1", {"output": "ok"})
        assert await mem.aget_tool_result(cfg, "c1") == {"output": "ok"}
    assert not any("does not implement the tool execution ledger" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_in_memory_ledger_is_scoped_to_the_thread_owner():
    from tenxgraph.core.authz import build_authz
    from tenxgraph.core.exceptions import StorageError
    from tenxgraph.storage.checkpointer.in_memory_checkpointer import InMemoryCheckpointer

    mem = InMemoryCheckpointer()
    owner = {"thread_id": "t1", "user_id": "u1", "authz": build_authz("u1", scope="owner")}
    other = {"thread_id": "t1", "user_id": "u2", "authz": build_authz("u2", scope="owner")}
    await mem.aput_tool_result(owner, "c1", {"output": "secret"})
    assert await mem.aget_tool_result(other, "c1") is None
    with pytest.raises(StorageError):
        await mem.aput_tool_result(other, "c2", {"output": "x"})
    assert await mem.aget_tool_result(owner, "c1") == {"output": "secret"}
