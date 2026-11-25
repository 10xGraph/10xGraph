"""Memory provenance (origin) and expiry (expires_at, ttl): schema, InMemoryStore, memory layer."""

import json
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.prebuilt.tools.memory import make_agent_memory_tool, make_user_memory_tool
from tenxgraph.qa.testing import InMemoryStore
from tenxgraph.storage.store import TRUSTED_ORIGINS, MemoryOrigin, MemoryRecord
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.storage.store.long_term_memory import _do_write
from tenxgraph.storage.store.memory_config import (
    AgentMemoryConfig,
    MemoryConfig,
    UserMemoryConfig,
)
from tenxgraph.storage.store.store_schema import MemorySearchResult, MemoryType
from tenxgraph.utils.background_task_manager import BackgroundTaskManager


CFG = {"user_id": "u1"}
PAST = datetime.now(UTC) - timedelta(hours=1)
FUTURE = datetime.now(UTC) + timedelta(days=1)


def _agent(memory: MemoryConfig) -> Agent:
    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        return Agent(
            model="gpt-4o",
            provider="openai",
            reasoning_config=None,
            tool_node=ToolNode([]),
            memory=memory,
        )


class _PlainStore(InMemoryStore):
    """A custom store that does not declare provenance support."""

    supports_provenance = False


class TestSchema:
    def test_exports_and_trusted_origins(self):
        assert TRUSTED_ORIGINS == (MemoryOrigin.USER, MemoryOrigin.SYSTEM)
        assert {o.value for o in MemoryOrigin} == {"user", "agent", "tool", "system"}

    def test_naive_expires_at_refused_on_models(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            MemoryRecord(content="x", expires_at=datetime(2030, 1, 1))
        with pytest.raises(ValueError, match="timezone-aware"):
            MemorySearchResult(expires_at=datetime(2030, 1, 1))

    def test_aware_expires_at_kept_in_utc(self):
        tz = timezone(timedelta(hours=2))
        record = MemoryRecord(content="x", expires_at=datetime(2030, 1, 1, 2, tzinfo=tz))
        assert record.expires_at == datetime(2030, 1, 1, tzinfo=UTC)

    def test_base_store_defaults(self):
        assert BaseStore.supports_provenance is False

    @pytest.mark.asyncio
    async def test_base_adelete_expired_not_implemented(self):
        with pytest.raises(NotImplementedError):
            await BaseStore.adelete_expired(_PlainStore())


@pytest.mark.asyncio
class TestInMemoryStore:
    async def test_origin_and_expiry_stored_not_in_metadata(self):
        store = InMemoryStore()
        mid = await store.astore(
            CFG,
            "fact",
            metadata={"origin": "system", "expires_at": "x", "k": 1},
            origin=MemoryOrigin.USER,
            expires_at=FUTURE,
        )
        got = await store.aget(CFG, mid)
        assert got.origin == MemoryOrigin.USER
        assert got.expires_at == FUTURE
        assert "origin" not in got.metadata and "expires_at" not in got.metadata
        assert got.metadata["k"] == 1

    async def test_naive_and_bad_origin_refused(self):
        store = InMemoryStore()
        with pytest.raises(ValueError, match="timezone-aware"):
            await store.astore(CFG, "x", expires_at=datetime(2030, 1, 1))
        with pytest.raises(ValueError, match="unknown origin"):
            await store.astore(CFG, "x", origin="web")

    async def test_expired_hidden_everywhere_and_top_k_not_shrunk(self):
        store = InMemoryStore()
        live = [await store.astore(CFG, f"note live {i}") for i in range(2)]
        expired = [await store.astore(CFG, f"note old {i}", expires_at=PAST) for i in range(3)]
        found = await store.asearch(CFG, "note", limit=2)
        assert sorted(r.id for r in found) == sorted(live)
        assert {r.id for r in await store.aget_all(CFG)} == set(live)
        assert await store.aget(CFG, expired[0]) is None

    async def test_expired_can_be_updated_and_deleted(self):
        store = InMemoryStore()
        mid = await store.astore(CFG, "old", expires_at=PAST)
        await store.aupdate(CFG, mid, "revived", expires_at=FUTURE)
        assert (await store.aget(CFG, mid)).content == "revived"
        other = await store.astore(CFG, "gone", expires_at=PAST)
        await store.adelete(CFG, other)
        assert other not in store.memories

    async def test_update_keeps_or_replaces_provenance(self):
        store = InMemoryStore()
        mid = await store.astore(CFG, "a", origin=MemoryOrigin.TOOL, expires_at=FUTURE)
        await store.aupdate(CFG, mid, "b")
        kept = await store.aget(CFG, mid)
        assert (kept.origin, kept.expires_at) == (MemoryOrigin.TOOL, FUTURE)
        await store.aupdate(CFG, mid, "c", origin=MemoryOrigin.SYSTEM)
        assert (await store.aget(CFG, mid)).origin == MemoryOrigin.SYSTEM

    async def test_origins_filter(self):
        store = InMemoryStore()
        ids = {
            o: await store.astore(CFG, f"note {o}", origin=o)
            for o in (MemoryOrigin.USER, MemoryOrigin.AGENT, MemoryOrigin.SYSTEM)
        }
        unknown = await store.astore(CFG, "note unknown")
        # No filter (the default): every origin, including agent and unknown.
        assert len(await store.asearch(CFG, "note")) == 4
        trusted = await store.asearch(CFG, "note", origins=list(TRUSTED_ORIGINS))
        assert {r.id for r in trusted} == {ids[MemoryOrigin.USER], ids[MemoryOrigin.SYSTEM]}
        with_unknown = await store.aget_all(CFG, origins=[MemoryOrigin.USER, None])
        assert {r.id for r in with_unknown} == {ids[MemoryOrigin.USER], unknown}
        with pytest.raises(ValueError, match="must not be empty"):
            await store.asearch(CFG, "note", origins=[])

    async def test_adelete_expired(self):
        store = InMemoryStore()
        keep = await store.astore(CFG, "keep")
        for i in range(5):
            await store.astore(CFG, f"old {i}", expires_at=PAST)
        await store.astore({**CFG, "collection": "other"}, "old other", expires_at=PAST)
        assert await store.adelete_expired(batch_size=2, collection="memories") == 5
        assert await store.adelete_expired() == 1
        assert list(store.memories) == [keep]
        with pytest.raises(ValueError, match="batch_size"):
            await store.adelete_expired(batch_size=0)


@pytest.mark.asyncio
class TestMemoryLayer:
    async def _remember(self, memory: MemoryConfig, store: BaseStore, **kwargs):
        manager = BackgroundTaskManager()
        result = await make_user_memory_tool(memory)(
            action="remember", store=store, task_manager=manager, config=CFG, **kwargs
        )
        await manager.wait_for_all()
        return json.loads(result)

    async def test_remember_writes_agent_origin_and_no_expiry_by_default(self):
        store = InMemoryStore()
        await self._remember(MemoryConfig(), store, text="likes tea")
        (mem,) = store.memories.values()
        assert mem.origin == MemoryOrigin.AGENT
        assert mem.expires_at is None

    async def test_ttl_sets_aware_expires_at_and_scope_wins(self):
        store = InMemoryStore()
        memory = MemoryConfig(
            ttl=timedelta(days=30), user_memory=UserMemoryConfig(ttl=timedelta(days=1))
        )
        before = datetime.now(UTC)
        await self._remember(memory, store, text="likes tea")
        (mem,) = store.memories.values()
        assert mem.expires_at.tzinfo is not None
        assert before + timedelta(days=1) <= mem.expires_at <= datetime.now(UTC) + timedelta(days=1)

    async def test_direct_astore_defaults_to_unknown(self):
        store = InMemoryStore()
        mid = await store.astore(CFG, "seeded")
        assert (await store.aget(CFG, mid)).origin is None

    async def test_origins_reach_tool_search(self):
        store = InMemoryStore()
        await store.astore(CFG, "tea user", origin=MemoryOrigin.USER)
        await store.astore(CFG, "tea agent", origin=MemoryOrigin.AGENT)
        default = make_user_memory_tool(MemoryConfig())
        trusted = make_user_memory_tool(MemoryConfig(origins=list(TRUSTED_ORIGINS)))
        args = {"action": "search", "text": "tea", "store": store, "config": CFG}
        manager = BackgroundTaskManager()
        assert len(json.loads(await default(task_manager=manager, **args))) == 2
        found = json.loads(await trusted(task_manager=manager, **args))
        assert [r["content"] for r in found] == ["tea user"]

    async def test_origins_reach_agent_tool_search(self):
        store = InMemoryStore()
        owner_cfg = {"user_id": "__system__:bot"}
        await store.astore(owner_cfg, "rule system", origin=MemoryOrigin.SYSTEM)
        await store.astore(owner_cfg, "rule tool", origin=MemoryOrigin.TOOL)
        memory = MemoryConfig(
            agent_memory=AgentMemoryConfig(enabled=True, agent_id="bot", origins=["system"])
        )
        result = await make_agent_memory_tool(memory)(
            query="rule", store=store, task_manager=BackgroundTaskManager(), config=CFG
        )
        assert [r["content"] for r in json.loads(result)] == ["rule system"]

    async def test_origins_reach_preload(self):
        store = InMemoryStore()
        await store.astore(CFG, "Editor vim user", origin=MemoryOrigin.USER)
        await store.astore(CFG, "Editor vim agent", origin=MemoryOrigin.AGENT)
        memory = MemoryConfig(retrieval_mode="preload", store=store, origins=list(TRUSTED_ORIGINS))
        block = await _agent(memory)._build_memory_block(
            AgentState(context=[Message.text_message("vim", role="user")]), CFG
        )
        assert "Editor vim user" in block and "Editor vim agent" not in block

    async def test_dedup_searches_ignore_origins(self):
        store = MagicMock(spec=InMemoryStore)
        store.supports_memory_key_upsert = False
        store.asearch = AsyncMock(return_value=[])
        store.astore = AsyncMock(return_value="m1")
        store.generate_framework_id = AsyncMock(return_value="m1")
        await _do_write(
            store,
            CFG,
            "store",
            "x",
            "",
            MemoryType.EPISODIC,
            "general",
            {"memory_key": "k"},
            "merge",
            origin=MemoryOrigin.AGENT,
        )
        # Key and similarity dedup both search every origin.
        assert store.asearch.await_count == 2
        assert all("origins" not in c.kwargs for c in store.asearch.await_args_list)
        assert store.astore.await_args.kwargs["origin"] == MemoryOrigin.AGENT

    async def test_key_update_passes_provenance(self):
        store = InMemoryStore()
        # InMemoryStore matches the key query as a substring of the content.
        await self._remember(MemoryConfig(), store, text="uses vim", memory_key="vim")
        mid = next(iter(store.memories))
        store.memories[mid] = store.memories[mid].model_copy(update={"origin": None})
        await self._remember(MemoryConfig(), store, text="still vim", memory_key="vim")
        assert len(store.memories) == 1
        assert store.memories[mid].origin == MemoryOrigin.AGENT


def test_sync_delete_expired():
    store = InMemoryStore()
    store.memories.clear()
    assert store.delete_expired() == 0


class TestCapabilityRefusal:
    def test_agent_refuses_origins_or_ttl_on_store_without_support(self):
        for memory in (
            MemoryConfig(store=_PlainStore(), origins=list(TRUSTED_ORIGINS)),
            MemoryConfig(store=_PlainStore(), ttl=timedelta(days=1)),
            MemoryConfig(user_memory=UserMemoryConfig(store=_PlainStore(), ttl=timedelta(1))),
        ):
            with pytest.raises(ValueError, match="supports_provenance"):
                _agent(memory)

    def test_agent_accepts_plain_store_without_origins_or_ttl(self):
        assert _agent(MemoryConfig(store=_PlainStore()))._memory_config is not None

    @pytest.mark.asyncio
    async def test_injected_store_refused_at_call_time(self):
        result = await make_user_memory_tool(MemoryConfig(ttl=timedelta(days=1)))(
            action="remember",
            text="x",
            store=_PlainStore(),
            task_manager=BackgroundTaskManager(),
            config=CFG,
        )
        assert "supports_provenance" in json.loads(result)["error"]

    def test_config_validation(self):
        with pytest.raises(ValueError):
            MemoryConfig(origins=[])
        with pytest.raises(ValueError):
            MemoryConfig(origins=["web"])
        with pytest.raises(ValueError):
            UserMemoryConfig(ttl=timedelta(seconds=-1))
