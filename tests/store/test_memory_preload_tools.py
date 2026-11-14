"""Preload write tool (gap 3) and the unsupported retrieval strategy warning (gap 5)."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.prebuilt.tools.memory import make_user_memory_tool
from tenxgraph.qa.testing import InMemoryStore
from tenxgraph.storage.store import base_store
from tenxgraph.storage.store.base_store import warn_unsupported_strategy
from tenxgraph.storage.store.long_term_memory import get_agent_memory_system_prompt
from tenxgraph.storage.store.memory_config import (
    AgentMemoryConfig,
    MemoryConfig,
    UserMemoryConfig,
)
from tenxgraph.storage.store.pgvector_store import PgVectorStore
from tenxgraph.storage.store.qdrant_store import QdrantStore
from tenxgraph.storage.store.store_schema import MemoryType, RetrievalStrategy
from tenxgraph.utils.background_task_manager import BackgroundTaskManager


def _agent(memory: MemoryConfig, tool_node: ToolNode | None) -> Agent:
    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        return Agent(
            model="gpt-4o",
            provider="openai",
            reasoning_config=None,
            tool_node=tool_node,
            memory=memory,
        )


def _schema(tool_fn) -> dict:
    return ToolNode([tool_fn]).all_tools_sync()[0]["function"]


def _pg_store() -> PgVectorStore:
    return PgVectorStore(None, pool=MagicMock())


@pytest.fixture(autouse=True)
def _reset_warned():
    base_store._WARNED_STRATEGIES.clear()
    yield
    base_store._WARNED_STRATEGIES.clear()


def _strategy_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.getMessage().startswith("MemoryConfig")
    ]


class TestPreloadTools:
    def test_preload_exposes_only_user_tool_with_remember_and_forget(self):
        memory = MemoryConfig(
            retrieval_mode="preload",
            user_memory=UserMemoryConfig(),
            agent_memory=AgentMemoryConfig(enabled=True, agent_id="bot"),
        )

        tools = memory.model_facing_tools()

        assert [t.__name__ for t in tools] == ["user_memory_tool"]
        schema = _schema(tools[0])
        assert schema["parameters"]["properties"]["action"]["enum"] == ["remember", "forget"]
        assert schema["parameters"]["properties"]["action"]["default"] == "remember"
        assert "search" not in schema["description"].lower()

    def test_postload_tool_keeps_all_actions(self):
        schema = _schema(make_user_memory_tool(MemoryConfig()))
        assert schema["parameters"]["properties"]["action"]["enum"] == [
            "search",
            "remember",
            "forget",
        ]

    def test_no_retrieval_exposes_no_tools(self):
        memory = MemoryConfig(retrieval_mode="no_retrieval", user_memory=UserMemoryConfig())
        assert memory.model_facing_tools() == []

    @pytest.mark.parametrize("actions", [(), ("search", "delete")])
    def test_invalid_actions_are_refused(self, actions):
        with pytest.raises(ValueError, match="actions must be"):
            make_user_memory_tool(MemoryConfig(), actions)

    @pytest.mark.asyncio
    async def test_disallowed_action_returns_error(self):
        store = InMemoryStore()
        tool_fn = make_user_memory_tool(MemoryConfig(), ("remember", "forget"))

        result = json.loads(
            await tool_fn(
                action="search",
                text="editor",
                store=store,
                task_manager=BackgroundTaskManager(),
                config={"user_id": "u1"},
            )
        )

        assert result == {"error": "action 'search' is not available; use one of: remember, forget"}

    def test_preload_prompt_has_remember_and_forget_lines(self):
        prompt = get_agent_memory_system_prompt(
            MemoryConfig(retrieval_mode="preload", user_memory=UserMemoryConfig())
        )

        assert "<user_memories_...>" in prompt
        assert "action='remember'" in prompt
        assert "action='forget' and its memory_key" in prompt
        assert "memory_key" in prompt
        assert "action='search'" not in prompt

    def test_preload_prompt_without_user_memory_has_no_tool_lines(self):
        prompt = get_agent_memory_system_prompt(
            MemoryConfig(retrieval_mode="preload", user_memory=None)
        )
        assert "user_memory_tool" not in prompt

    def test_preload_with_user_memory_needs_a_tool_node(self):
        memory = MemoryConfig(retrieval_mode="preload", user_memory=UserMemoryConfig())
        with pytest.raises(RuntimeError, match=r"retrieval_mode 'preload'.*user_memory_tool"):
            _agent(memory, tool_node=None)

    def test_preload_without_user_memory_needs_no_tool_node(self):
        memory = MemoryConfig(retrieval_mode="preload", user_memory=None)
        assert _agent(memory, tool_node=None)._tool_node is None

    @pytest.mark.asyncio
    async def test_preload_finds_semantic_memory_with_default_config(self):
        store = InMemoryStore()
        await store.astore(
            {"user_id": "u1"}, "Editor: vim", memory_type=MemoryType.SEMANTIC, category="editor"
        )
        memory = MemoryConfig(retrieval_mode="preload", store=store)
        agent = _agent(memory, tool_node=ToolNode([]))

        block = await agent._build_memory_block(
            AgentState(context=[Message.text_message("vim", role="user")]),
            {"user_id": "u1"},
        )

        assert block is not None
        assert "Editor: vim" in block


class TestUnsupportedStrategyWarning:
    def test_helper_warns_once_per_store_class_and_strategy(self, caplog):
        store = _pg_store()
        with caplog.at_level(logging.WARNING, logger="tenxgraph.store"):
            warn_unsupported_strategy(store, "temporal")
            warn_unsupported_strategy(_pg_store(), RetrievalStrategy.TEMPORAL)
            warn_unsupported_strategy(store, "graph_traversal")

        messages = _strategy_warnings(caplog)
        tail = (
            "is not supported by PgVectorStore (supported: similarity, hybrid, text); every "
            "memory search will fail. Use created_at filters for time ranges."
        )
        assert messages == [
            f"MemoryConfig retrieval_strategy 'temporal' {tail}",
            f"MemoryConfig retrieval_strategy 'graph_traversal' {tail}",
        ]

    def test_no_warning_for_supported_none_or_undeclared(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tenxgraph.store"):
            warn_unsupported_strategy(_pg_store(), "hybrid")
            warn_unsupported_strategy(_pg_store(), None)
            warn_unsupported_strategy(InMemoryStore(), "temporal")

        assert _strategy_warnings(caplog) == []

    def test_qdrant_declares_similarity_only(self):
        assert QdrantStore.supported_retrieval_strategies == frozenset(
            {RetrievalStrategy.SIMILARITY}
        )

    def test_agent_setup_warns_for_scope_and_top_level_strategy(self, caplog):
        memory = MemoryConfig(
            store=_pg_store(),
            retrieval_strategy="temporal",
            user_memory=UserMemoryConfig(),
            agent_memory=AgentMemoryConfig(
                enabled=True, agent_id="bot", retrieval_strategy="graph_traversal"
            ),
        )
        with caplog.at_level(logging.WARNING, logger="tenxgraph.store"):
            _agent(memory, tool_node=ToolNode([]))

        messages = _strategy_warnings(caplog)
        assert len(messages) == 2
        assert "'temporal'" in messages[0]
        assert "'graph_traversal'" in messages[1]

    def test_agent_setup_no_warning_for_supported_strategy(self, caplog):
        memory = MemoryConfig(store=_pg_store(), retrieval_strategy="text")
        with caplog.at_level(logging.WARNING, logger="tenxgraph.store"):
            _agent(memory, tool_node=ToolNode([]))

        assert _strategy_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_di_store_warns_once_on_tool_calls(self, caplog):
        tool_fn = make_user_memory_tool(MemoryConfig(retrieval_strategy="temporal"))
        with caplog.at_level(logging.WARNING, logger="tenxgraph.store"):
            for _ in range(2):
                result = json.loads(
                    await tool_fn(
                        action="search",
                        text="editor",
                        store=_pg_store(),
                        task_manager=BackgroundTaskManager(),
                        config={"user_id": "u1"},
                    )
                )
                assert "error" in result

        assert len(_strategy_warnings(caplog)) == 1

    @pytest.mark.asyncio
    async def test_preload_search_warns(self, caplog):
        memory = MemoryConfig(
            retrieval_mode="preload", store=_pg_store(), retrieval_strategy="temporal"
        )
        agent = _agent(memory, tool_node=ToolNode([]))
        base_store._WARNED_STRATEGIES.clear()
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger="tenxgraph.store"):
            block = await agent._build_memory_block(
                AgentState(context=[Message.text_message("hi", role="user")]),
                {"user_id": "u1"},
            )

        assert block is None
        assert len(_strategy_warnings(caplog)) == 1
