"""Store operations need a concrete user_id; system_user_id(name) owns shared data (PLAN Q5)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.authz import SCOPE_NONE, build_authz
from tenxgraph.prebuilt.tools.memory import _memory_scope_config
from tenxgraph.storage.store import (
    SYSTEM_USER_PREFIX,
    AgentMemoryConfig,
    MemoryConfig,
    UserMemoryConfig,
    is_system_user_id,
    system_user_id,
)
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.storage.store.mem0_store import Mem0Store
from tenxgraph.storage.store.qdrant_store import QdrantStore


OWNER = "__system__:kb"


class _Embedding:
    dimension = 4

    async def aembed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def client():
    mock = AsyncMock()
    mock.get_collections.return_value = MagicMock(collections=[])
    mock.retrieve.return_value = []
    mock.query_points.return_value = MagicMock(points=[])
    mock.scroll.return_value = ([], None)
    return mock


@pytest.fixture
def qdrant(client):
    with patch("qdrant_client.AsyncQdrantClient", return_value=client):
        return QdrantStore(embedding=_Embedding(), path="./test_data")


BAD_CONFIGS = [{}, {"user_id": None}, {"user_id": ""}, {"user_id": "   "}]


def _calls(store):
    return {
        "astore": lambda c: store.astore(c, "x"),
        "asearch": lambda c: store.asearch(c, "q"),
        "aget": lambda c: store.aget(c, "m1"),
        "aget_all": lambda c: store.aget_all(c),
        "aupdate": lambda c: store.aupdate(c, "m1", "y"),
        "adelete": lambda c: store.adelete(c, "m1"),
        "aforget_memory": lambda c: store.aforget_memory(c),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("config", BAD_CONFIGS)
@pytest.mark.parametrize(
    "op", ["astore", "asearch", "aget", "aget_all", "aupdate", "adelete", "aforget_memory"]
)
async def test_qdrant_operation_raises_without_user_id(qdrant, client, op, config):
    with pytest.raises(ValueError, match="user_id is required"):
        await _calls(qdrant)[op](config)
    client.upsert.assert_not_called()
    client.delete.assert_not_called()
    client.query_points.assert_not_called()
    client.scroll.assert_not_called()


@pytest.mark.asyncio
async def test_qdrant_read_filters_by_user(qdrant, client):
    await qdrant.asearch({"user_id": "alice"}, "q")
    flt = client.query_points.call_args.kwargs["query_filter"]
    assert [c.key for c in flt.must] == ["user_id"]
    assert flt.must[0].match.value == "alice"


@pytest.mark.asyncio
async def test_qdrant_scope_none_reads_all_users(qdrant, client):
    config = {"user_id": "", "user": {"authz": build_authz("admin", scope=SCOPE_NONE)}}
    await qdrant.asearch(config, "q")
    assert client.query_points.call_args.kwargs["query_filter"] is None
    await qdrant.aget_all(config)
    assert client.scroll.call_args.kwargs["scroll_filter"] is None


@pytest.mark.asyncio
async def test_qdrant_scope_none_write_still_needs_owner(qdrant, client):
    config = {"user": {"authz": build_authz("admin", scope=SCOPE_NONE)}}
    with pytest.raises(ValueError, match="user_id is required"):
        await qdrant.astore(config, "x")
    with pytest.raises(ValueError, match="user_id is required"):
        await qdrant.aforget_memory(config)
    client.upsert.assert_not_called()
    client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_qdrant_system_user_can_write_and_forget(qdrant, client):
    await qdrant.astore({"user_id": OWNER}, "shared fact")
    assert client.upsert.call_args.kwargs["points"][0].payload["user_id"] == OWNER
    await qdrant.aforget_memory({"user_id": "alice", "thread_id": "t1"})
    flt = client.delete.call_args.kwargs["points_selector"].filter
    assert {c.key for c in flt.must} == {"user_id", "thread_id"}


@pytest.mark.asyncio
@pytest.mark.parametrize("config", BAD_CONFIGS)
async def test_mem0_raises_without_user_id(config):
    store = Mem0Store(config={}, app_id="app")
    for op in ("astore", "asearch", "aget", "aget_all", "adelete", "aforget_memory"):
        with pytest.raises(ValueError, match="user_id is required"):
            await _calls(store)[op](config)


def test_system_user_id_exports():
    import tenxgraph.storage.store as store_pkg
    from tenxgraph import storage

    assert SYSTEM_USER_PREFIX == "__system__"
    assert system_user_id("bot") == "__system__:bot"
    assert system_user_id("  bot ") == "__system__:bot"
    for mod in (storage, store_pkg):
        # other tests reload the store package, so compare behaviour, not identity
        assert mod.system_user_id("bot") == "__system__:bot"
        assert mod.is_system_user_id("__system__:bot")
        assert mod.SYSTEM_USER_PREFIX == SYSTEM_USER_PREFIX
        for name in ("SYSTEM_USER_PREFIX", "system_user_id", "is_system_user_id"):
            assert name in mod.__all__
    assert not hasattr(storage, "SYSTEM_USER_ID")


@pytest.mark.parametrize("name", ["", "   ", None, 3])
def test_system_user_id_rejects_empty(name):
    with pytest.raises(ValueError):
        system_user_id(name)


@pytest.mark.parametrize("value", ["__system__", "__system__:x", "__system__x"])
def test_is_system_user_id_true(value):
    assert is_system_user_id(value)


@pytest.mark.parametrize("value", ["system", "x__system__", "alice", "", 5, None])
def test_is_system_user_id_false(value):
    assert not is_system_user_id(value)


class _FakeStore(BaseStore):
    """Tiny in-memory store that scopes reads by user like the real stores."""

    def __init__(self):
        self.rows: list[dict] = []

    async def asetup(self):
        return None

    async def astore(self, config, content, **kwargs):
        self.rows.append(
            {
                "user_id": self._require_owner_id(config),
                "thread_id": config.get("thread_id"),
                "content": content,
            }
        )
        return str(len(self.rows))

    async def asearch(self, config, query, **kwargs):
        user_id = self._require_user_id(config)
        return [
            r
            for r in self.rows
            if (user_id is None or r["user_id"] == user_id)  # thread_id ignored, like Qdrant
        ]

    async def aget(self, *a, **k): ...
    async def aget_all(self, *a, **k): ...
    async def aupdate(self, *a, **k): ...
    async def adelete(self, *a, **k): ...
    async def aforget_memory(self, *a, **k): ...


def _agent_cfg(run_user: str) -> dict:
    memory = MemoryConfig(
        agent_memory=AgentMemoryConfig(enabled=True, agent_id="bot", app_id="app")
    )
    return _memory_scope_config(
        {"user_id": run_user, "thread_id": "run-thread"},
        memory,
        memory.agent_memory,
        scope="agent",
    )


def test_agent_scope_config_always_uses_system_owner():
    for run_user in ("alice", "bob", ""):
        cfg = _agent_cfg(run_user)
        assert cfg["user_id"] == "__system__:bot"
        assert cfg["agent_id"] == "bot" and cfg["thread_id"] == "bot" and cfg["app_id"] == "app"


@pytest.mark.asyncio
async def test_agent_memory_shared_across_users_user_memory_isolated():
    store = _FakeStore()
    await store.astore({"user_id": "__system__:bot", "thread_id": "bot"}, "shared agent fact")
    await store.astore({"user_id": "alice"}, "alice secret")

    for run_user in ("alice", "bob"):
        found = await store.asearch(_agent_cfg(run_user), "q")
        assert [r["content"] for r in found] == ["shared agent fact"]

    user_memory = MemoryConfig(user_memory=UserMemoryConfig())
    for run_user, expected in (("alice", ["alice secret"]), ("bob", [])):
        cfg = _memory_scope_config(
            {"user_id": run_user}, user_memory, user_memory.user_memory, scope="user"
        )
        assert [r["content"] for r in await store.asearch(cfg, "q")] == expected


@pytest.mark.asyncio
async def test_agents_with_different_ids_do_not_share_memories():
    store = _FakeStore()
    await store.astore({"user_id": "__system__:a", "thread_id": "a"}, "fact of a")
    await store.astore({"user_id": "__system__:b", "thread_id": "b"}, "fact of b")

    def cfg_for(agent_id):
        memory = MemoryConfig(agent_memory=AgentMemoryConfig(enabled=True, agent_id=agent_id))
        return _memory_scope_config(
            {"user_id": "alice"}, memory, memory.agent_memory, scope="agent"
        )

    assert cfg_for("a")["user_id"] == "__system__:a"
    assert [r["content"] for r in await store.asearch(cfg_for("a"), "q")] == ["fact of a"]
    assert [r["content"] for r in await store.asearch(cfg_for("b"), "q")] == ["fact of b"]


def test_agent_owner_falls_back_to_app_id():
    memory = MemoryConfig(agent_memory=AgentMemoryConfig(enabled=True, app_id="app"))
    cfg = _memory_scope_config({}, memory, memory.agent_memory, scope="agent")
    assert cfg["user_id"] == "__system__:app"


def test_agent_memory_enabled_without_ids_raises_at_setup():
    from tenxgraph.core.graph.agent_internal.memory import AgentMemoryMixin

    class _A(AgentMemoryMixin):
        system_prompt: list = []
        _tool_node = None
        tool_node_name = None

    bad = MemoryConfig(agent_memory=AgentMemoryConfig(enabled=True))
    with pytest.raises(ValueError, match="neither agent_id nor app_id"):
        _A()._setup_memory(bad)
    # defaults (agent memory disabled) still set up fine
    _A()._setup_memory(MemoryConfig(inject_system_prompt=False, user_memory=None))
