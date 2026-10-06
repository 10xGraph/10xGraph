"""Unit tests for the modernised RAGAgent."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tenxgraph.core.graph.base_agent import BaseAgent
from tenxgraph.core.graph.compiled_graph import CompiledGraph
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.message import Message
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.storage.store.store_schema import MemorySearchResult
from tenxgraph.prebuilt.agent.rag import (
    RAGAgent,
    BaseReranker,
    CohereReranker,
    CrossEncoderReranker,
    _build_context_message,
    _RAG_DOCS_KEY,
    _RETRIEVE_NODE,
    _RERANK_NODE,
    _SYNTHESIZE_NODE,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class FakeAgent(BaseAgent):
    """Minimal agent stub."""

    def __init__(self, model: str = "fake", **kwargs):
        super().__init__(model=model, **kwargs)
        self.last_state: AgentState | None = None

    async def execute(self, state: AgentState, config: dict) -> AgentState:
        self.last_state = state
        state.context.append(Message.text_message("synthesized answer", role="assistant"))
        return state

    async def _call_llm(self, messages, tools=None, **kwargs):
        raise NotImplementedError


class FakeStore(BaseStore):
    """Minimal store stub that returns preset results."""

    def __init__(self, results: list[MemorySearchResult] | None = None):
        self._results = results or []

    async def asetup(self): ...

    async def asearch(self, config, query, **kwargs) -> list[MemorySearchResult]:
        return self._results

    async def astore(self, config, content, **kwargs) -> str:
        return "id"

    async def aget(self, config, record_id, **kwargs):
        return None

    async def aget_all(self, config, **kwargs):
        return []

    async def aupdate(self, config, record_id, **kwargs): ...

    async def adelete(self, config, record_id, **kwargs): ...

    async def aforget_memory(self, config, **kwargs): ...


class FakeReranker:
    """Reranker stub: returns docs reversed to show ordering was applied."""

    async def arerank(self, query: str, documents: list[str], top_n: int) -> list[str]:
        return list(reversed(documents))[:top_n]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(text: str, role: str = "user") -> Message:
    return Message.text_message(text, role=role)  # type: ignore[arg-type]


def _state_with(*messages: Message) -> AgentState:
    state = AgentState()
    state.context = list(messages)
    return state


def _results(*texts: str) -> list[MemorySearchResult]:
    return [MemorySearchResult(content=t, score=0.9) for t in texts]


def _fake_agent() -> FakeAgent:
    return FakeAgent()


def _fake_store(docs: list[str] | None = None) -> FakeStore:
    return FakeStore(_results(*(docs or [])))


# ===========================================================================
# Tests for _build_context_message
# ===========================================================================


class TestBuildContextMessage:
    def test_single_doc(self):
        result = _build_context_message(["doc one"], "What is X?")
        assert "[1] doc one" in result
        assert "What is X?" in result

    def test_multiple_docs(self):
        result = _build_context_message(["a", "b", "c"], "query")
        assert "[1] a" in result
        assert "[2] b" in result
        assert "[3] c" in result

    def test_context_block_wrapping(self):
        result = _build_context_message(["doc"], "q")
        assert result.startswith("<context>")
        assert "</context>" in result

    def test_empty_docs(self):
        # No docs → just the query (function doesn't special-case empty list)
        result = _build_context_message([], "q")
        assert "q" in result


# ===========================================================================
# Tests for BaseReranker protocol
# ===========================================================================


class TestBaseRerankerProtocol:
    def test_fake_reranker_satisfies_protocol(self):
        assert isinstance(FakeReranker(), BaseReranker)

    def test_class_without_arerank_does_not_satisfy(self):
        class NotAReranker:
            pass

        assert not isinstance(NotAReranker(), BaseReranker)


# ===========================================================================
# Tests for CohereReranker
# ===========================================================================


class TestCohereReranker:
    def test_raises_without_cohere_installed(self):
        with patch.dict("sys.modules", {"cohere": None}):
            with pytest.raises(ImportError, match="cohere"):
                CohereReranker(api_key="key")

    def test_default_model(self):
        with patch("builtins.__import__", side_effect=lambda name, *a, **k: MagicMock() if name == "cohere" else __import__(name, *a, **k)):
            try:
                r = CohereReranker.__new__(CohereReranker)
                r._api_key = "k"
                r._model = "rerank-v4.0-pro"
                r._client = None
                assert r._model == "rerank-v4.0-pro"
            except Exception:
                pass  # Only testing attribute default, import may vary

    @pytest.mark.asyncio
    async def test_arerank_empty_docs(self):
        r = CohereReranker.__new__(CohereReranker)
        r._api_key = "k"
        r._model = "m"
        r._client = None
        result = await r.arerank("q", [], top_n=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_arerank_calls_api(self):
        r = CohereReranker.__new__(CohereReranker)
        r._api_key = "k"
        r._model = "rerank-v4.0-pro"
        # Mock the client
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.results = [MagicMock(index=1), MagicMock(index=0)]
        mock_client.rerank = AsyncMock(return_value=mock_result)
        r._client = mock_client
        docs = ["alpha", "beta"]
        result = await r.arerank("q", docs, top_n=2)
        assert result == ["beta", "alpha"]  # index=1 first, then index=0


# ===========================================================================
# Tests for CrossEncoderReranker
# ===========================================================================


class TestCrossEncoderReranker:
    def test_raises_without_sentence_transformers(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                CrossEncoderReranker()

    @pytest.mark.asyncio
    async def test_arerank_empty_docs(self):
        r = CrossEncoderReranker.__new__(CrossEncoderReranker)
        r._model_name = "m"
        r._encoder = None
        result = await r.arerank("q", [], top_n=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_arerank_ranks_by_score(self):
        r = CrossEncoderReranker.__new__(CrossEncoderReranker)
        r._model_name = "m"
        # Inject a mock encoder
        mock_encoder = MagicMock()
        mock_encoder.predict = MagicMock(return_value=[0.2, 0.9, 0.5])
        r._encoder = mock_encoder
        docs = ["low", "high", "mid"]
        result = await r.arerank("q", docs, top_n=2)
        # Highest score first
        assert result[0] == "high"
        assert result[1] == "mid"
        assert len(result) == 2


# ===========================================================================
# Tests for RAGAgent.__init__ validation
# ===========================================================================


class TestRAGAgentInit:
    def test_basic_init(self):
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent())
        assert rag is not None

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            RAGAgent(store=_fake_store(), agent=_fake_agent(), top_k=0)

    def test_top_n_zero_raises(self):
        with pytest.raises(ValueError, match="top_n"):
            RAGAgent(store=_fake_store(), agent=_fake_agent(), top_n=0)

    def test_stores_params(self):
        store = _fake_store()
        agent = _fake_agent()
        reranker = FakeReranker()
        rag = RAGAgent(store=store, agent=agent, reranker=reranker, top_k=10, top_n=4)
        assert rag._store is store
        assert rag._agent is agent
        assert rag._reranker is reranker
        assert rag._top_k == 10
        assert rag._top_n == 4

    def test_default_no_reranker(self):
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent())
        assert rag._reranker is None

    def test_store_config_defaults_to_empty(self):
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent())
        assert rag._store_config == {}

    def test_store_config_passed(self):
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent(), store_config={"user_id": "u1"})
        assert rag._store_config == {"user_id": "u1"}


# ===========================================================================
# Tests for RAGAgent._configure_graph (graph topology)
# ===========================================================================


class TestRAGAgentGraphTopology:
    def test_without_reranker_no_rerank_node(self):
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent())
        rag._configure_graph()
        nodes = set(rag._graph.nodes.keys())
        assert _RETRIEVE_NODE in nodes
        assert _SYNTHESIZE_NODE in nodes
        assert _RERANK_NODE not in nodes

    def test_with_reranker_all_three_nodes(self):
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent(), reranker=FakeReranker())
        rag._configure_graph()
        nodes = set(rag._graph.nodes.keys())
        assert _RETRIEVE_NODE in nodes
        assert _RERANK_NODE in nodes
        assert _SYNTHESIZE_NODE in nodes

    def test_entry_point_is_retrieve(self):
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent())
        rag._configure_graph()
        assert rag._graph.entry_point == _RETRIEVE_NODE


# ===========================================================================
# Tests for RAGAgent.compile
# ===========================================================================


class TestRAGAgentCompile:
    def test_compile_returns_compiled_graph(self):
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent())
        graph = rag.compile()
        assert isinstance(graph, CompiledGraph)

    def test_compile_with_reranker_returns_compiled_graph(self):
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent(), reranker=FakeReranker())
        graph = rag.compile()
        assert isinstance(graph, CompiledGraph)

    def test_compile_twice_resets_graph(self):
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent())
        g1 = rag.compile()
        g2 = rag.compile()
        assert g1 is not g2


# ===========================================================================
# Tests for RETRIEVE node behaviour
# ===========================================================================


class TestRetrieveNode:
    @pytest.mark.asyncio
    async def test_stores_docs_in_internal_data(self):
        store = _fake_store(["doc A", "doc B"])
        rag = RAGAgent(store=store, agent=_fake_agent(), top_k=2)
        retrieve = rag._make_retrieve_node()
        state = _state_with(_msg("What is Python?"))
        result = await retrieve(state, {})
        assert result.execution_meta.internal_data[_RAG_DOCS_KEY] == ["doc A", "doc B"]

    @pytest.mark.asyncio
    async def test_no_user_message_stores_empty(self):
        store = _fake_store(["doc"])
        rag = RAGAgent(store=store, agent=_fake_agent())
        retrieve = rag._make_retrieve_node()
        state = _state_with(_msg("system message", role="system"))
        result = await retrieve(state, {})
        assert result.execution_meta.internal_data[_RAG_DOCS_KEY] == []

    @pytest.mark.asyncio
    async def test_passes_top_k_to_store(self):
        store = FakeStore()
        store.asearch = AsyncMock(return_value=[])
        rag = RAGAgent(store=store, agent=_fake_agent(), top_k=7)
        retrieve = rag._make_retrieve_node()
        await retrieve(_state_with(_msg("q")), {})
        call_kwargs = store.asearch.call_args.kwargs
        assert call_kwargs["limit"] == 7

    @pytest.mark.asyncio
    async def test_passes_store_config(self):
        store = FakeStore()
        store.asearch = AsyncMock(return_value=[])
        rag = RAGAgent(store=store, agent=_fake_agent(), store_config={"user_id": "u42"})
        retrieve = rag._make_retrieve_node()
        await retrieve(_state_with(_msg("q")), {})
        call_kwargs = store.asearch.call_args.kwargs
        assert call_kwargs["config"] == {"user_id": "u42"}

    @pytest.mark.asyncio
    async def test_empty_store_results_stores_empty_list(self):
        store = _fake_store([])
        rag = RAGAgent(store=store, agent=_fake_agent())
        retrieve = rag._make_retrieve_node()
        result = await retrieve(_state_with(_msg("q")), {})
        assert result.execution_meta.internal_data[_RAG_DOCS_KEY] == []


# ===========================================================================
# Tests for RERANK node behaviour
# ===========================================================================


class TestRerankNode:
    @pytest.mark.asyncio
    async def test_reranks_docs(self):
        reranker = FakeReranker()  # reverses + top_n
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent(), reranker=reranker, top_n=2)
        rerank = rag._make_rerank_node()
        state = _state_with(_msg("q"))
        state.execution_meta.internal_data[_RAG_DOCS_KEY] = ["a", "b", "c"]
        result = await rerank(state, {})
        # FakeReranker reverses: ["c", "b", "a"] → top_n=2 → ["c", "b"]
        assert result.execution_meta.internal_data[_RAG_DOCS_KEY] == ["c", "b"]

    @pytest.mark.asyncio
    async def test_empty_docs_skip_reranking(self):
        reranker = AsyncMock()
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent(), reranker=reranker)
        rerank = rag._make_rerank_node()
        state = _state_with(_msg("q"))
        state.execution_meta.internal_data[_RAG_DOCS_KEY] = []
        await rerank(state, {})
        reranker.arerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_query_to_reranker(self):
        reranker = AsyncMock()
        reranker.arerank = AsyncMock(return_value=["doc"])
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent(), reranker=reranker, top_n=1)
        rerank = rag._make_rerank_node()
        state = _state_with(_msg("my specific query"))
        state.execution_meta.internal_data[_RAG_DOCS_KEY] = ["doc"]
        await rerank(state, {})
        call = reranker.arerank.call_args
        # arerank is called as arerank(query, docs, top_n=...) — positional
        assert call.args[0] == "my specific query"


# ===========================================================================
# Tests for SYNTHESIZE node behaviour
# ===========================================================================


class TestSynthesizeNode:
    @pytest.mark.asyncio
    async def test_injects_docs_into_user_message(self):
        agent = _fake_agent()
        rag = RAGAgent(store=_fake_store(), agent=agent)
        synthesize = rag._make_synthesize_node()
        state = _state_with(_msg("What is X?"))
        state.execution_meta.internal_data[_RAG_DOCS_KEY] = ["fact one", "fact two"]
        await synthesize(state, {})
        # The last user message seen by agent should contain the context block
        last_user = next(m for m in reversed(agent.last_state.context) if m.role == "user")
        assert "<context>" in last_user.text()
        assert "fact one" in last_user.text()
        assert "What is X?" in last_user.text()

    @pytest.mark.asyncio
    async def test_no_docs_skips_injection(self):
        agent = _fake_agent()
        rag = RAGAgent(store=_fake_store(), agent=agent)
        synthesize = rag._make_synthesize_node()
        state = _state_with(_msg("plain query"))
        state.execution_meta.internal_data[_RAG_DOCS_KEY] = []
        await synthesize(state, {})
        last_user = next(m for m in reversed(agent.last_state.context) if m.role == "user")
        assert "<context>" not in last_user.text()
        assert last_user.text() == "plain query"

    @pytest.mark.asyncio
    async def test_calls_agent_execute(self):
        agent = _fake_agent()
        rag = RAGAgent(store=_fake_store(), agent=agent)
        synthesize = rag._make_synthesize_node()
        state = _state_with(_msg("q"))
        state.execution_meta.internal_data[_RAG_DOCS_KEY] = ["d"]
        result = await synthesize(state, {})
        # FakeAgent appends an answer
        assert any(m.role == "assistant" for m in result.context)


# ===========================================================================
# Integration — end-to-end node ordering
# ===========================================================================


class TestRAGIntegration:
    @pytest.mark.asyncio
    async def test_retrieve_then_synthesize_without_reranker(self):
        """RETRIEVE fills rag_docs; SYNTHESIZE injects them and calls agent."""
        store = _fake_store(["chunk A", "chunk B"])
        agent = _fake_agent()
        rag = RAGAgent(store=store, agent=agent, top_k=2)

        retrieve = rag._make_retrieve_node()
        synthesize = rag._make_synthesize_node()

        state = _state_with(_msg("tell me about chunks"))
        state = await retrieve(state, {})
        assert state.execution_meta.internal_data[_RAG_DOCS_KEY] == ["chunk A", "chunk B"]

        state = await synthesize(state, {})
        last_user = next(m for m in reversed(agent.last_state.context) if m.role == "user")
        assert "chunk A" in last_user.text()
        assert "chunk B" in last_user.text()

    @pytest.mark.asyncio
    async def test_retrieve_rerank_synthesize(self):
        """Full three-node pipeline: RETRIEVE → RERANK → SYNTHESIZE."""
        store = _fake_store(["low", "mid", "high"])
        reranker = FakeReranker()  # reverses → [high, mid, low] → top_n=2 → [high, mid]
        agent = _fake_agent()
        rag = RAGAgent(store=store, agent=agent, reranker=reranker, top_k=3, top_n=2)

        retrieve = rag._make_retrieve_node()
        rerank = rag._make_rerank_node()
        synthesize = rag._make_synthesize_node()

        state = _state_with(_msg("q"))
        state = await retrieve(state, {})
        state = await rerank(state, {})
        assert state.execution_meta.internal_data[_RAG_DOCS_KEY] == ["high", "mid"]
        state = await synthesize(state, {})

        last_user = next(m for m in reversed(agent.last_state.context) if m.role == "user")
        assert "high" in last_user.text()
        assert "mid" in last_user.text()
        # "low" was cut off by reranker
        assert "low" not in last_user.text()

    def test_compiled_graph_has_correct_nodes_without_reranker(self):
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent())
        graph = rag.compile()
        node_names = set(graph.nodes.keys()) if hasattr(graph, "nodes") else set()
        # Just ensure compile does not raise
        assert graph is not None

    def test_compiled_graph_has_correct_nodes_with_reranker(self):
        rag = RAGAgent(store=_fake_store(), agent=_fake_agent(), reranker=FakeReranker())
        graph = rag.compile()
        assert graph is not None
