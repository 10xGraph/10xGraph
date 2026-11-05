"""Search settings in three levels: store default, MemoryConfig / scope, method call."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from tenxgraph.core.state import AgentState, Message
from tenxgraph.prebuilt.tools.memory import (
    _memory_scope_retrieval_strategy,
    _search_scope_memory,
    make_agent_memory_tool,
    make_user_memory_tool,
)
from tenxgraph.qa.testing.in_memory_store import InMemoryStore
from tenxgraph.storage.store import AgentMemoryConfig, MemoryConfig, UserMemoryConfig
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.storage.store.qdrant_store import QdrantStore
from tenxgraph.storage.store.store_schema import (
    DistanceMetric,
    MemorySearchResult,
    RetrievalStrategy,
)


CFG = {"user_id": "u1", "thread_id": "t1"}


class _Embedding:
    dimension = 4

    async def aembed(self, text):
        return [0.1, 0.2, 0.3, 0.4]

    async def aembed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]


def _qdrant(**kw):
    client = AsyncMock()
    client.get_collections.return_value = MagicMock(collections=[])
    client.query_points.return_value = MagicMock(points=[])
    with patch("qdrant_client.AsyncQdrantClient", return_value=client):
        store = QdrantStore(embedding=_Embedding(), path="./test_data", **kw)
    return store, client


class _Recorder(BaseStore):
    def __init__(self):
        self.asearch_mock = AsyncMock(return_value=[])

    async def aping(self):
        return None

    async def astore(self, *a, **k):
        return "id"

    async def asearch(self, *a, **k):
        return await self.asearch_mock(*a, **k)

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


class TestEnum:
    def test_text_member(self):
        assert RetrievalStrategy.TEXT == "text"
        assert RetrievalStrategy("text") is RetrievalStrategy.TEXT


class TestQdrantLevels:
    @pytest.mark.asyncio
    async def test_none_uses_store_defaults(self):
        store, client = _qdrant()
        await store.asearch(CFG, "q", retrieval_strategy=None, distance_metric=None)
        client.query_points.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_matching_call_values_accepted(self):
        store, _ = _qdrant(distance_metric=DistanceMetric.DOT_PRODUCT)
        await store.asearch(
            CFG,
            "q",
            retrieval_strategy=RetrievalStrategy.SIMILARITY,
            distance_metric=DistanceMetric.DOT_PRODUCT,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy", [RetrievalStrategy.TEXT, RetrievalStrategy.HYBRID])
    async def test_unsupported_call_strategy_raises(self, strategy):
        store, client = _qdrant()
        with pytest.raises(ValueError, match="not supported by QdrantStore"):
            await store.asearch(CFG, "q", retrieval_strategy=strategy)
        client.query_points.assert_not_called()

    def test_unsupported_store_default_raises(self):
        with pytest.raises(ValueError, match="not supported by QdrantStore"):
            _qdrant(retrieval_strategy=RetrievalStrategy.HYBRID)

    @pytest.mark.asyncio
    async def test_mismatched_call_metric_raises(self):
        store, client = _qdrant()  # cosine
        with pytest.raises(ValueError, match="fixed at collection creation"):
            await store.asearch(CFG, "q", distance_metric=DistanceMetric.EUCLIDEAN)
        client.query_points.assert_not_called()


class TestConfigValidation:
    def test_defaults_none(self):
        mc = MemoryConfig()
        assert mc.retrieval_strategy is None
        assert mc.user_memory.retrieval_strategy is None
        assert mc.agent_memory.retrieval_strategy is None

    def test_str_coerced_to_enum(self):
        mc = MemoryConfig(
            retrieval_strategy="text",
            user_memory=UserMemoryConfig(retrieval_strategy="hybrid"),
            agent_memory=AgentMemoryConfig(retrieval_strategy=RetrievalStrategy.SIMILARITY),
        )
        assert mc.retrieval_strategy is RetrievalStrategy.TEXT
        assert mc.user_memory.retrieval_strategy is RetrievalStrategy.HYBRID
        assert mc.agent_memory.retrieval_strategy is RetrievalStrategy.SIMILARITY

    def test_invalid_rejected(self):
        with pytest.raises(ValidationError):
            MemoryConfig(retrieval_strategy="nope")
        with pytest.raises(ValidationError):
            UserMemoryConfig(retrieval_strategy="nope")

    def test_scope_wins_over_memory_config(self):
        mc = MemoryConfig(
            retrieval_strategy="text",
            user_memory=UserMemoryConfig(retrieval_strategy="hybrid"),
            agent_memory=AgentMemoryConfig(enabled=True),
        )
        assert _memory_scope_retrieval_strategy(mc, mc.user_memory) is RetrievalStrategy.HYBRID
        assert _memory_scope_retrieval_strategy(mc, mc.agent_memory) is RetrievalStrategy.TEXT

    def test_none_when_unset(self):
        mc = MemoryConfig()
        assert _memory_scope_retrieval_strategy(mc, mc.user_memory) is None


class TestCallSites:
    @pytest.mark.asyncio
    async def test_search_scope_memory_passes_strategy(self):
        store = _Recorder()
        await _search_scope_memory(
            store=store,
            config=CFG,
            query="q",
            memory_type="episodic",
            category="general",
            limit=3,
            score_threshold=None,
            task_manager=None,
            retrieval_strategy=RetrievalStrategy.TEXT,
        )
        assert store.asearch_mock.call_args.kwargs["retrieval_strategy"] is RetrievalStrategy.TEXT

    @pytest.mark.asyncio
    async def test_search_scope_memory_default_none(self):
        store = _Recorder()
        await _search_scope_memory(
            store=store,
            config=CFG,
            query="q",
            memory_type="episodic",
            category="general",
            limit=3,
            score_threshold=None,
            task_manager=None,
        )
        assert store.asearch_mock.call_args.kwargs["retrieval_strategy"] is None

    @pytest.mark.asyncio
    async def test_user_tool_uses_scope_then_memory_config(self):
        store = _Recorder()
        mc = MemoryConfig(
            store=store,
            retrieval_strategy="text",
            user_memory=UserMemoryConfig(user_id="u1"),
        )
        tool_fn = make_user_memory_tool(mc)
        await tool_fn(action="search", text="hi", config=dict(CFG), store=store, task_manager=None)
        assert store.asearch_mock.call_args.kwargs["retrieval_strategy"] is RetrievalStrategy.TEXT

        store2 = _Recorder()
        mc2 = MemoryConfig(
            store=store2,
            retrieval_strategy="text",
            user_memory=UserMemoryConfig(user_id="u1", retrieval_strategy="hybrid"),
        )
        await make_user_memory_tool(mc2)(
            action="search", text="hi", config=dict(CFG), store=store2, task_manager=None
        )
        assert (
            store2.asearch_mock.call_args.kwargs["retrieval_strategy"]
            is RetrievalStrategy.HYBRID
        )

    @pytest.mark.asyncio
    async def test_agent_tool_passes_strategy(self):
        store = _Recorder()
        mc = MemoryConfig(
            store=store,
            retrieval_strategy="text",
            agent_memory=AgentMemoryConfig(enabled=True, agent_id="a1"),
        )
        await make_agent_memory_tool(mc)(
            query="hi", config=dict(CFG), store=store, task_manager=None
        )
        assert store.asearch_mock.call_args.kwargs["retrieval_strategy"] is RetrievalStrategy.TEXT

    @pytest.mark.asyncio
    async def test_preload_passes_strategy(self):
        from tenxgraph.core.graph.agent_internal.memory import AgentMemoryMixin

        store = _Recorder()
        store.asearch_mock.return_value = [
            MemorySearchResult(id="m", content="x", score=0.9)
        ]
        mc = MemoryConfig(
            retrieval_mode="preload",
            retrieval_strategy="text",
            user_memory=UserMemoryConfig(store=store, user_id="u1"),
            agent_memory=AgentMemoryConfig(
                enabled=True, agent_id="a1", store=store, retrieval_strategy="hybrid"
            ),
        )
        holder = AgentMemoryMixin.__new__(AgentMemoryMixin)
        holder._memory_config = mc
        state = AgentState(context=[Message.text_message("hi", role="user")])
        await holder._search_memory_block(state, dict(CFG))
        strategies = [c.kwargs["retrieval_strategy"] for c in store.asearch_mock.call_args_list]
        assert strategies == [RetrievalStrategy.TEXT, RetrievalStrategy.HYBRID]


class TestInMemoryStore:
    @pytest.mark.asyncio
    async def test_accepts_none(self):
        store = InMemoryStore()
        await store.asearch(CFG, "q", retrieval_strategy=None, distance_metric=None)
        await store.asearch(CFG, "q", retrieval_strategy=RetrievalStrategy.TEXT)
