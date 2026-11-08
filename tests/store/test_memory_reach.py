"""Agent and user memory ignore the caller's reach (AUTH-S0-03)."""

from unittest.mock import AsyncMock

import pytest

from tenxgraph.core.authz import Identity, Reach, owner_filter
from tenxgraph.core.graph.agent_internal.memory import AgentMemoryMixin
from tenxgraph.core.state import AgentState, Message
from tenxgraph.prebuilt.tools.memory import (
    _memory_scope_config,
    make_agent_memory_tool,
    make_user_memory_tool,
)
from tenxgraph.storage.store import AgentMemoryConfig, MemoryConfig, UserMemoryConfig
from tenxgraph.storage.store.base_store import BaseStore, system_user_id


AGENT_OWNER = system_user_id("a1")


class _Recorder(BaseStore):
    def __init__(self):
        self.asearch_mock = AsyncMock(return_value=[])

    async def asearch(self, *a, **k):
        return await self.asearch_mock(*a, **k)

    async def aping(self):
        return None

    async def astore(self, *a, **k):
        return "id"

    async def aget(self, *a, **k):
        return None

    async def aget_all(self, *a, **k):
        return []

    async def aupdate(self, *a, **k):
        return None

    async def adelete(self, *a, **k):
        return None

    async def aforget_memory(self, *a, **k):
        return None

    async def arelease(self):
        return None


def _config(reach):
    identity = Identity(user_id="alice", tenant={"tenant_id": "acme"}, reach=reach)
    return {"user": identity, "thread_id": "t1"}


def _memory(store):
    return MemoryConfig(
        retrieval_mode="preload",
        user_memory=UserMemoryConfig(store=store),
        agent_memory=AgentMemoryConfig(enabled=True, agent_id="a1", store=store),
    )


def _filters(store):
    return [owner_filter(c.args[0], ("tenant_id",)) for c in store.asearch_mock.call_args_list]


@pytest.mark.parametrize("reach", [Reach.OWN, Reach.TENANT, Reach.ALL])
@pytest.mark.asyncio
async def test_tools_pin_owner(reach):
    store = _Recorder()
    mc = _memory(store)
    await make_user_memory_tool(mc)(text="x", config=_config(reach), store=store, task_manager=None)
    await make_agent_memory_tool(mc)(
        query="x", config=_config(reach), store=store, task_manager=None
    )
    user_f, agent_f = _filters(store)
    assert user_f == {"user_id": "alice", "tenant_id": "acme"}
    assert agent_f == {"user_id": AGENT_OWNER, "tenant_id": "acme"}


@pytest.mark.parametrize("reach", [Reach.OWN, Reach.TENANT, Reach.ALL])
@pytest.mark.asyncio
async def test_preload_pins_owner(reach):
    store = _Recorder()
    holder = AgentMemoryMixin.__new__(AgentMemoryMixin)
    holder._memory_config = _memory(store)
    state = AgentState(context=[Message.text_message("hi", role="user")])
    await holder._search_memory_block(state, _config(reach))
    user_f, agent_f = _filters(store)
    assert user_f == {"user_id": "alice", "tenant_id": "acme"}
    assert agent_f == {"user_id": AGENT_OWNER, "tenant_id": "acme"}


def test_no_identity_unchanged():
    mc = _memory(None)
    cfg = _memory_scope_config({"user_id": "u1"}, mc, mc.agent_memory, scope="agent")
    assert cfg["user_id"] == AGENT_OWNER
    assert "user" not in cfg
    cfg = _memory_scope_config({"user_id": "u1"}, mc, mc.user_memory, scope="user")
    assert cfg["user_id"] == "u1"
    assert "user" not in cfg


def test_fixed_user_id_applies_only_without_identity():
    # AR-CORE-08: the identity wins over UserMemoryConfig.user_id (never widens to a shared id).
    mc = MemoryConfig(user_memory=UserMemoryConfig(store=None, user_id="fixed"))
    cfg = _memory_scope_config(_config(Reach.OWN), mc, mc.user_memory, scope="user")
    assert owner_filter(cfg, ("tenant_id",)) == {"user_id": "alice", "tenant_id": "acme"}
    cfg = _memory_scope_config({"user_id": "u1"}, mc, mc.user_memory, scope="user")
    assert cfg["user_id"] == "fixed"
