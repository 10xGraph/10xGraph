"""Tests for tenxgraph.store.long_term_memory module."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.store.long_term_memory import (
    DEFAULT_READ_MODE,
    _IDENTICAL_SCORE_THRESHOLD,
    MemoryIntegration,
    ReadMode,
    _do_write,
    _find_duplicate_by_key,
    _find_duplicate_by_similarity,
    _format_search_results,
    _strip_thread_id,
    _validate_memory_type,
    create_memory_preload_node,
    get_memory_system_prompt,
)
from tenxgraph.prebuilt.tools.memory import (
    make_agent_memory_tool,
    make_user_memory_tool,
    memory_tool,
)
from tenxgraph.storage.store.memory_config import (
    AgentMemoryConfig,
    MemoryConfig,
    UserMemoryConfig,
)
from tenxgraph.storage.store.store_schema import MemorySearchResult, MemoryType
from tenxgraph.core.graph.tool_node import ToolNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_store():
    store = AsyncMock()
    store.astore = AsyncMock(return_value="mem-001")
    store.asearch = AsyncMock(return_value=[])
    store.aget = AsyncMock(return_value=None)
    store.aupdate = AsyncMock()
    store.adelete = AsyncMock()
    return store


@pytest.fixture()
def mock_task_manager():
    mgr = MagicMock()
    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = False
    task.add_done_callback = MagicMock()

    def _create_task(coro, *args, **kwargs):
        if hasattr(coro, "close"):
            coro.close()
        return task

    mgr.create_task = MagicMock(side_effect=_create_task)
    mgr.get_task_count = MagicMock(return_value=0)
    mgr.wait_for_all = AsyncMock()
    return mgr


@pytest.fixture()
def sample_search_results():
    return [
        MemorySearchResult(
            id="r1",
            content="User prefers dark mode",
            score=0.92,
            memory_type=MemoryType.SEMANTIC,
            metadata={"source": "chat"},
        ),
        MemorySearchResult(
            id="r2",
            content="User works with Python 3.12",
            score=0.85,
            memory_type=MemoryType.EPISODIC,
            metadata={},
        ),
    ]


@pytest.fixture()
def sample_state():
    return AgentState(
        context=[Message.text_message("What is my preferred editor?", role="user")]
    )


@pytest.fixture()
def config():
    return {"user_id": "u1", "thread_id": "t1"}


# ---------------------------------------------------------------------------
# ReadMode & defaults
# ---------------------------------------------------------------------------


class TestReadModeDefaults:
    def test_default_read_mode_is_no_retrieval(self):
        assert DEFAULT_READ_MODE == ReadMode.NO_RETRIEVAL

    def test_read_mode_values(self):
        assert ReadMode.NO_RETRIEVAL.value == "no_retrieval"
        assert ReadMode.PRELOAD.value == "preload"
        assert ReadMode.POSTLOAD.value == "postload"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestValidateMemoryType:
    def test_valid_type(self):
        assert _validate_memory_type("semantic") == MemoryType.SEMANTIC

    def test_invalid_type_returns_episodic(self):
        assert _validate_memory_type("invalid") == MemoryType.EPISODIC

    def test_all_valid_types(self):
        for mt in MemoryType:
            assert _validate_memory_type(mt.value) == mt


class TestFormatSearchResults:
    def test_empty(self):
        assert _format_search_results([]) == []

    def test_formats_correctly(self, sample_search_results):
        formatted = _format_search_results(sample_search_results)
        assert len(formatted) == 2
        assert formatted[0]["id"] == "r1"
        assert formatted[0]["content"] == "User prefers dark mode"
        assert formatted[0]["score"] == 0.92
        assert formatted[0]["memory_type"] == "semantic"
        assert formatted[1]["id"] == "r2"


# ---------------------------------------------------------------------------
# _do_write
# ---------------------------------------------------------------------------


class TestDoWrite:
    @pytest.mark.asyncio
    async def test_store(self, mock_store, config):
        # asearch returns empty → no duplicate → proceeds with astore
        mock_store.asearch.return_value = []
        result = await _do_write(
            mock_store, config, "store", "hello", "", MemoryType.EPISODIC, "general", None, "merge"
        )
        assert result == {"status": "stored", "memory_id": "mem-001"}
        mock_store.astore.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_replace(self, mock_store, config):
        result = await _do_write(
            mock_store, config, "update", "new text", "m1", MemoryType.EPISODIC, "general", {"x": 1}, "replace"
        )
        assert result == {"status": "updated", "memory_id": "m1"}
        mock_store.aupdate.assert_awaited_once_with({"user_id": "u1"}, "m1", "new text", metadata={"x": 1}, replace_metadata=True)

    @pytest.mark.asyncio
    async def test_update_merge(self, mock_store, config):
        mock_store.aget.return_value = MemorySearchResult(
            id="m1", content="old", metadata={"a": 1}
        )
        result = await _do_write(
            mock_store, config, "update", "new", "m1", MemoryType.EPISODIC, "general", {"b": 2}, "merge"
        )
        assert result["status"] == "updated"
        mock_store.aupdate.assert_awaited_once_with(
            {"user_id": "u1"}, "m1", "new", metadata={"a": 1, "b": 2}
        )

    @pytest.mark.asyncio
    async def test_update_merge_no_existing(self, mock_store, config):
        mock_store.aget.return_value = None
        result = await _do_write(
            mock_store, config, "update", "new", "m1", MemoryType.EPISODIC, "general", {"b": 2}, "merge"
        )
        assert result["status"] == "updated"
        mock_store.aupdate.assert_awaited_once_with({"user_id": "u1"}, "m1", "new", metadata={"b": 2})

    @pytest.mark.asyncio
    async def test_delete(self, mock_store, config):
        result = await _do_write(
            mock_store, config, "delete", "", "m1", MemoryType.EPISODIC, "general", None, "merge"
        )
        assert result == {"status": "deleted", "memory_id": "m1"}
        mock_store.adelete.assert_awaited_once_with({"user_id": "u1"}, "m1")


# ---------------------------------------------------------------------------
# memory_tool
# ---------------------------------------------------------------------------


class TestMemoryToolSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_store, mock_task_manager, sample_search_results, config):
        mock_store.asearch.return_value = sample_search_results
        result = json.loads(
            await memory_tool(
                action="search",
                query="preferences",
                store=mock_store,
                task_manager=mock_task_manager,
                config=config,
            )
        )
        assert len(result) == 2
        assert result[0]["id"] == "r1"
        mock_store.asearch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_empty_query_error(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="search", query="", store=mock_store,
                task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result
        assert "query" in result["error"]

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, mock_store, mock_task_manager, config):
        mock_store.asearch.return_value = []
        await memory_tool(
            action="search", query="test", score_threshold=0.5,
            store=mock_store, task_manager=mock_task_manager, config=config,
        )
        call_kwargs = mock_store.asearch.call_args
        assert call_kwargs.kwargs.get("score_threshold") == 0.5


class TestMemoryToolStore:
    @pytest.mark.asyncio
    async def test_store_success(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="store", content="user likes Python",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert result["status"] == "scheduled"
        mock_task_manager.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_empty_content_error(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="store", content="",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result


class TestMemoryToolUpdate:
    @pytest.mark.asyncio
    async def test_update_merge(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="update", memory_id="m1", content="new",
                metadata={"b": 2}, write_mode="merge",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert result["status"] == "scheduled"
        mock_task_manager.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_replace(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="update", memory_id="m1", content="new",
                metadata={"b": 2}, write_mode="replace",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert result["status"] == "scheduled"
        mock_task_manager.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_no_memory_id_error(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="update", memory_id="", content="new",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_update_no_content_error(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="update", memory_id="m1", content="",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result


class TestMemoryToolDelete:
    @pytest.mark.asyncio
    async def test_delete_success(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="delete", memory_id="m1",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert result["status"] == "scheduled"
        mock_task_manager.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_no_memory_id_error(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="delete", memory_id="",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result


class TestMemoryToolNoStore:
    @pytest.mark.asyncio
    async def test_no_store_returns_error(self, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="search", query="test",
                store=None, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result
        assert "no memory store" in result["error"]


class TestMemoryToolWriteScheduling:
    @pytest.mark.asyncio
    async def test_store_schedules_task(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="store", content="data",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert result["status"] == "scheduled"
        mock_task_manager.create_task.assert_called_once()
        mock_store.astore.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_schedules_task(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="delete", memory_id="m1",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert result["status"] == "scheduled"
        mock_task_manager.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_not_scheduled(self, mock_store, mock_task_manager, config):
        """Search is always synchronous — not scheduled as background task."""
        mock_store.asearch.return_value = []
        result = json.loads(
            await memory_tool(
                action="search", query="test",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert isinstance(result, list)
        mock_task_manager.create_task.assert_not_called()


class TestMemoryToolExceptionHandling:
    @pytest.mark.asyncio
    async def test_store_scheduling_exception_returns_error(self, mock_store, mock_task_manager, config):
        mock_task_manager.create_task.side_effect = RuntimeError("task manager failed")
        result = json.loads(
            await memory_tool(
                action="store", content="data",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result
        assert "task manager failed" in result["error"]

    @pytest.mark.asyncio
    async def test_search_exception_returns_error(self, mock_store, mock_task_manager, config):
        mock_store.asearch.side_effect = RuntimeError("timeout")
        result = json.loads(
            await memory_tool(
                action="search", query="test",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# Agent-level memory tools
# ---------------------------------------------------------------------------


class TestAgentLevelMemoryTools:
    @pytest.mark.asyncio
    async def test_user_memory_tool_search_uses_cross_thread_config(
        self, mock_store, mock_task_manager, sample_search_results, config
    ):
        memory = MemoryConfig(user_memory=UserMemoryConfig(user_id="u1"))
        tool_fn = make_user_memory_tool(memory)
        mock_store.asearch.return_value = sample_search_results

        result = json.loads(
            await tool_fn(
                action="search",
                text="preferences",
                store=mock_store,
                task_manager=mock_task_manager,
                config=config,
            )
        )

        assert len(result) == 2
        call_args = mock_store.asearch.call_args
        assert call_args.args[0] == {"user_id": "u1"}

    @pytest.mark.asyncio
    async def test_user_memory_tool_remember_schedules_write(
        self, mock_store, mock_task_manager, config
    ):
        memory = MemoryConfig(user_memory=UserMemoryConfig(user_id="u1"))
        tool_fn = make_user_memory_tool(memory)

        result = json.loads(
            await tool_fn(
                action="remember",
                text="User prefers short answers",
                store=mock_store,
                task_manager=mock_task_manager,
                config=config,
            )
        )

        assert result == {"status": "scheduled", "action": "remember"}
        mock_task_manager.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_memory_tool_search_is_read_only(
        self, mock_store, mock_task_manager, sample_search_results, config
    ):
        memory = MemoryConfig(
            agent_memory=AgentMemoryConfig(enabled=True, agent_id="agent-1")
        )
        tool_fn = make_agent_memory_tool(memory)
        mock_store.asearch.return_value = sample_search_results

        result = json.loads(
            await tool_fn(
                query="release policy",
                store=mock_store,
                task_manager=mock_task_manager,
                config=config,
            )
        )

        assert len(result) == 2
        call_args = mock_store.asearch.call_args
        assert call_args.args[0]["thread_id"] == "agent-1"
        mock_task_manager.create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_agent_tool_schemas_hide_admin_fields(self):
        memory = MemoryConfig(
            user_memory=UserMemoryConfig(),
            agent_memory=AgentMemoryConfig(enabled=True),
        )
        tools = ToolNode(memory.model_facing_tools())
        schemas = await tools.all_tools()

        schema_by_name = {s["function"]["name"]: s for s in schemas}
        user_props = schema_by_name["user_memory_tool"]["function"]["parameters"]["properties"]
        agent_props = schema_by_name["agent_memory_tool"]["function"]["parameters"]["properties"]

        assert set(user_props) == {"action", "text", "memory_type", "category", "limit"}
        assert set(agent_props) == {"query", "memory_type", "category", "limit"}
        assert "memory_id" not in user_props
        assert "memory_key" not in user_props
        assert "action" not in agent_props

    def test_preload_does_not_expose_model_facing_tools(self):
        memory = MemoryConfig(
            retrieval_mode="preload",
            user_memory=UserMemoryConfig(),
            agent_memory=AgentMemoryConfig(enabled=True),
        )

        assert memory.model_facing_tools() == []


# ---------------------------------------------------------------------------
# _strip_thread_id
# ---------------------------------------------------------------------------


class TestStripThreadId:
    def test_removes_thread_id(self):
        cfg = {"user_id": "u1", "thread_id": "t1", "memory_type": "episodic"}
        result = _strip_thread_id(cfg)
        assert "thread_id" not in result
        assert result["user_id"] == "u1"
        assert result["memory_type"] == "episodic"

    def test_original_dict_not_mutated(self):
        cfg = {"user_id": "u1", "thread_id": "t1"}
        _strip_thread_id(cfg)
        assert "thread_id" in cfg  # original unchanged

    def test_no_thread_id_returns_copy(self):
        cfg = {"user_id": "u1"}
        result = _strip_thread_id(cfg)
        assert result == {"user_id": "u1"}
        assert result is not cfg  # always a copy

    def test_empty_dict(self):
        assert _strip_thread_id({}) == {}


# ---------------------------------------------------------------------------
# create_memory_preload_node
# ---------------------------------------------------------------------------


class TestPreloadNode:
    @pytest.mark.asyncio
    async def test_basic_preload(self, mock_store, sample_search_results, config):
        mock_store.asearch.return_value = sample_search_results

        node_fn = create_memory_preload_node(mock_store, limit=5)
        state = AgentState(
            context=[Message.text_message("Tell me about my setup", role="user")]
        )
        result = await node_fn(state, config)

        assert len(result) == 1
        assert result[0].role == "system"
        text = result[0].text()
        assert "User prefers dark mode" in text
        assert "Long-term Memory Context" in text
        mock_store.asearch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_preload_no_results(self, mock_store, config):
        mock_store.asearch.return_value = []
        node_fn = create_memory_preload_node(mock_store)
        state = AgentState(
            context=[Message.text_message("hello", role="user")]
        )
        result = await node_fn(state, config)
        assert result == []

    @pytest.mark.asyncio
    async def test_preload_no_user_message(self, mock_store, config):
        node_fn = create_memory_preload_node(mock_store)
        state = AgentState(context=[Message.text_message("I am an assistant", role="assistant")])
        result = await node_fn(state, config)
        assert result == []

    @pytest.mark.asyncio
    async def test_preload_custom_query_builder(self, mock_store, config):
        mock_store.asearch.return_value = []
        custom_query = lambda state: "custom query"
        node_fn = create_memory_preload_node(mock_store, query_builder=custom_query)
        state = AgentState(
            context=[Message.text_message("irrelevant", role="user")]
        )
        await node_fn(state, config)
        call_args = mock_store.asearch.call_args
        assert call_args[0][1] == "custom query"

    @pytest.mark.asyncio
    async def test_preload_store_exception(self, mock_store, config):
        mock_store.asearch.side_effect = RuntimeError("connection lost")
        node_fn = create_memory_preload_node(mock_store)
        state = AgentState(
            context=[Message.text_message("hello", role="user")]
        )
        result = await node_fn(state, config)
        assert result == []

    @pytest.mark.asyncio
    async def test_preload_custom_template(self, mock_store, sample_search_results, config):
        mock_store.asearch.return_value = sample_search_results
        template = "MEMORIES:\n{memories}"
        node_fn = create_memory_preload_node(mock_store, system_prompt_template=template)
        state = AgentState(
            context=[Message.text_message("hello", role="user")]
        )
        result = await node_fn(state, config)
        text = result[0].text()
        assert text.startswith("MEMORIES:")


# ---------------------------------------------------------------------------
# get_memory_system_prompt
# ---------------------------------------------------------------------------


class TestGetMemorySystemPrompt:
    def test_no_retrieval_includes_write_instructions(self):
        prompt = get_memory_system_prompt("no_retrieval")
        assert len(prompt) > 0
        assert "do not" in prompt.lower() or "do not have access" in prompt.lower()
        assert "memory_tool" in prompt
        assert "store" in prompt.lower()

    def test_preload_includes_read_and_write(self):
        prompt = get_memory_system_prompt("preload")
        assert "memory context" in prompt.lower()
        assert "memory_tool" in prompt
        assert "store" in prompt.lower()
        assert len(prompt) > 0

    def test_postload_returns_text(self):
        prompt = get_memory_system_prompt("postload")
        assert "memory_tool" in prompt
        assert "search" in prompt
        assert "store" in prompt

    def test_default_mode_includes_write(self):
        prompt = get_memory_system_prompt()
        assert "memory_tool" in prompt

    def test_unknown_mode_returns_empty(self):
        assert get_memory_system_prompt("unknown_mode") == ""


# ---------------------------------------------------------------------------
# Tool schema generation
# ---------------------------------------------------------------------------


class TestMemoryToolSchema:
    def test_tool_decorator_metadata(self):
        assert memory_tool._py_tool_name == "memory_tool"
        assert "memory" in memory_tool._py_tool_tags

    def test_tool_description(self):
        assert "Search" in memory_tool._py_tool_description
        assert "store" in memory_tool._py_tool_description.lower()


# ---------------------------------------------------------------------------
# MemoryIntegration
# ---------------------------------------------------------------------------


class TestMemoryIntegrationInit:
    def test_default_mode_is_no_retrieval(self, mock_store):
        mi = MemoryIntegration(store=mock_store)
        assert mi.retrieval_mode == ReadMode.NO_RETRIEVAL

    def test_accepts_string_mode(self, mock_store):
        mi = MemoryIntegration(store=mock_store, retrieval_mode="preload")
        assert mi.retrieval_mode == ReadMode.PRELOAD

    def test_accepts_enum_mode(self, mock_store):
        mi = MemoryIntegration(store=mock_store, retrieval_mode=ReadMode.POSTLOAD)
        assert mi.retrieval_mode == ReadMode.POSTLOAD

    def test_invalid_string_mode_raises(self, mock_store):
        with pytest.raises(ValueError):
            MemoryIntegration(store=mock_store, retrieval_mode="invalid_mode")

    def test_store_property(self, mock_store):
        mi = MemoryIntegration(store=mock_store)
        assert mi.store is mock_store


class TestMemoryIntegrationPreloadNode:
    def test_preload_mode_creates_node(self, mock_store):
        mi = MemoryIntegration(store=mock_store, retrieval_mode="preload")
        assert mi.preload_node is not None
        assert callable(mi.preload_node)

    def test_no_retrieval_has_no_preload_node(self, mock_store):
        mi = MemoryIntegration(store=mock_store, retrieval_mode="no_retrieval")
        assert mi.preload_node is None

    def test_postload_has_no_preload_node(self, mock_store):
        mi = MemoryIntegration(store=mock_store, retrieval_mode="postload")
        assert mi.preload_node is None

    @pytest.mark.asyncio
    async def test_preload_node_runs(self, mock_store, sample_search_results, config):
        mock_store.asearch.return_value = sample_search_results
        mi = MemoryIntegration(store=mock_store, retrieval_mode="preload")
        state = AgentState(
            context=[Message.text_message("hello", role="user")]
        )
        result = await mi.preload_node(state, config)
        assert len(result) == 1
        assert result[0].role == "system"
        assert "Long-term Memory Context" in result[0].text()

    @pytest.mark.asyncio
    async def test_preload_node_custom_template(self, mock_store, sample_search_results, config):
        mock_store.asearch.return_value = sample_search_results
        mi = MemoryIntegration(
            store=mock_store,
            retrieval_mode="preload",
            preload_prompt_template="CUSTOM:\n{memories}",
        )
        state = AgentState(
            context=[Message.text_message("hello", role="user")]
        )
        result = await mi.preload_node(state, config)
        assert result[0].text().startswith("CUSTOM:")


class TestMemoryIntegrationSystemPrompt:
    def test_no_retrieval_prompt(self, mock_store):
        mi = MemoryIntegration(store=mock_store, retrieval_mode="no_retrieval")
        prompt = mi.system_prompt
        assert "memory_tool" in prompt
        assert "do NOT" in prompt

    def test_preload_prompt(self, mock_store):
        mi = MemoryIntegration(store=mock_store, retrieval_mode="preload")
        prompt = mi.system_prompt
        assert "memory context" in prompt.lower()
        assert "memory_tool" in prompt

    def test_postload_prompt(self, mock_store):
        mi = MemoryIntegration(store=mock_store, retrieval_mode="postload")
        prompt = mi.system_prompt
        assert "search" in prompt
        assert "memory_tool" in prompt


class TestMemoryIntegrationTools:
    def test_tools_always_includes_memory_tool(self, mock_store):
        for mode in ReadMode:
            mi = MemoryIntegration(store=mock_store, retrieval_mode=mode)
            assert memory_tool in mi.tools

    def test_tools_returns_list(self, mock_store):
        mi = MemoryIntegration(store=mock_store)
        assert isinstance(mi.tools, list)
        assert len(mi.tools) >= 1


class TestMemoryIntegrationWire:
    def test_wire_preload_adds_node_and_entry(self, mock_store):
        from tenxgraph.core.graph import StateGraph

        mi = MemoryIntegration(store=mock_store, retrieval_mode="preload")
        graph = StateGraph(AgentState())

        async def dummy(state, config):
            return state

        graph.add_node("main", dummy)
        mi.wire(graph, entry_to="main")

        assert graph.entry_point == "memory_preload"
        assert "memory_preload" in graph.nodes

    def test_wire_no_retrieval_sets_entry_directly(self, mock_store):
        from tenxgraph.core.graph import StateGraph

        mi = MemoryIntegration(store=mock_store, retrieval_mode="no_retrieval")
        graph = StateGraph(AgentState())

        async def dummy(state, config):
            return state

        graph.add_node("main", dummy)
        mi.wire(graph, entry_to="main")

        assert graph.entry_point == "main"
        assert "memory_preload" not in graph.nodes

    def test_wire_postload_sets_entry_directly(self, mock_store):
        from tenxgraph.core.graph import StateGraph

        mi = MemoryIntegration(store=mock_store, retrieval_mode="postload")
        graph = StateGraph(AgentState())

        async def dummy(state, config):
            return state

        graph.add_node("main", dummy)
        mi.wire(graph, entry_to="main")

        assert graph.entry_point == "main"

    def test_wire_custom_preload_name(self, mock_store):
        from tenxgraph.core.graph import StateGraph

        mi = MemoryIntegration(store=mock_store, retrieval_mode="preload")
        graph = StateGraph(AgentState())

        async def dummy(state, config):
            return state

        graph.add_node("main", dummy)
        mi.wire(graph, entry_to="main", preload_node_name="mem_load")

        assert graph.entry_point == "mem_load"
        assert "mem_load" in graph.nodes

    def test_wire_preload_creates_edge(self, mock_store):
        from tenxgraph.core.graph import StateGraph

        mi = MemoryIntegration(store=mock_store, retrieval_mode="preload")
        graph = StateGraph(AgentState())

        async def dummy(state, config):
            return state

        graph.add_node("main", dummy)
        mi.wire(graph, entry_to="main")

        # Verify edge from preload → main exists
        has_edge = any(
            e.from_node == "memory_preload" and e.to_node == "main"
            for e in graph.edges
        )
        assert has_edge


# ---------------------------------------------------------------------------
# Deduplication — _find_duplicate & _do_write dedup paths
# ---------------------------------------------------------------------------


class TestFindDuplicateByKey:
    """Tests for the _find_duplicate_by_key helper."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_match(self, mock_store):
        mock_store.asearch.return_value = []
        result = await _find_duplicate_by_key(
            mock_store, {"user_id": "u1"}, "user_name", MemoryType.EPISODIC
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_match_when_key_exists(self, mock_store):
        hit = MemorySearchResult(id="m1", content="Name is Atharv", score=0.90)
        mock_store.asearch.return_value = [hit]
        result = await _find_duplicate_by_key(
            mock_store, {"user_id": "u1"}, "user_name", MemoryType.EPISODIC
        )
        assert result is hit
        # Verify memory_key filter was passed
        call_kwargs = mock_store.asearch.call_args[1]
        assert call_kwargs["filters"] == {"memory_key": "user_name"}

    @pytest.mark.asyncio
    async def test_passes_prestripped_config_to_asearch(self, mock_store):
        """_find_duplicate_by_key passes config as-is; callers must strip thread_id first."""
        mock_store.asearch.return_value = []
        pre_stripped = {"user_id": "u1"}
        await _find_duplicate_by_key(
            mock_store,
            pre_stripped,
            "user_name",
            MemoryType.EPISODIC,
        )
        called_config = mock_store.asearch.call_args[0][0]
        assert called_config == pre_stripped

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self, mock_store):
        mock_store.asearch.side_effect = RuntimeError("boom")
        result = await _find_duplicate_by_key(
            mock_store, {"user_id": "u1"}, "user_name", MemoryType.EPISODIC
        )
        assert result is None


class TestFindDuplicateBySimilarity:
    """Tests for the _find_duplicate_by_similarity helper."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_match(self, mock_store):
        mock_store.asearch.return_value = []
        result = await _find_duplicate_by_similarity(
            mock_store, {"user_id": "u1"}, "some content", MemoryType.EPISODIC
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_match_above_threshold(self, mock_store):
        hit = MemorySearchResult(id="m1", content="same", score=0.97)
        mock_store.asearch.return_value = [hit]
        result = await _find_duplicate_by_similarity(
            mock_store, {"user_id": "u1"}, "same content", MemoryType.EPISODIC
        )
        assert result is hit

    @pytest.mark.asyncio
    async def test_passes_prestripped_config_to_asearch(self, mock_store):
        """_find_duplicate_by_similarity passes config as-is; callers must strip thread_id first."""
        mock_store.asearch.return_value = []
        pre_stripped = {"user_id": "u1"}
        await _find_duplicate_by_similarity(
            mock_store,
            pre_stripped,
            "content",
            MemoryType.EPISODIC,
        )
        called_config = mock_store.asearch.call_args[0][0]
        assert called_config == pre_stripped

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self, mock_store):
        mock_store.asearch.side_effect = RuntimeError("boom")
        result = await _find_duplicate_by_similarity(
            mock_store, {"user_id": "u1"}, "content", MemoryType.EPISODIC
        )
        assert result is None


class TestDoWriteDedup:
    """Tests for deduplication logic inside _do_write (action='store')."""

    @pytest.mark.asyncio
    async def test_store_updates_by_memory_key(self, mock_store, config):
        """When memory_key is provided and a matching record exists → update."""
        # First call (key search) returns a match; second (similarity) not reached
        mock_store.asearch.return_value = [
            MemorySearchResult(
                id="existing-1",
                content="User's name is Atharv",
                score=0.50,
                metadata={"memory_key": "user_name"},
            )
        ]
        result = await _do_write(
            mock_store, config, "store", "User's name is Prashant",
            "", MemoryType.EPISODIC, "general",
            {"memory_key": "user_name"}, "merge",
        )
        assert result["status"] == "updated_existing"
        assert result["memory_id"] == "existing-1"
        assert result["memory_key"] == "user_name"
        mock_store.aupdate.assert_awaited_once()
        mock_store.astore.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_skips_identical_duplicate(self, mock_store, config):
        """Score >= 0.95 on similarity check → skip the write entirely."""
        # No memory_key → falls through to similarity check
        mock_store.asearch.return_value = [
            MemorySearchResult(id="existing-1", content="My name is Atharv", score=0.97)
        ]
        result = await _do_write(
            mock_store, config, "store", "My name is Atharv",
            "", MemoryType.EPISODIC, "general", None, "merge",
        )
        assert result["status"] == "skipped_duplicate"
        assert result["memory_id"] == "existing-1"
        mock_store.astore.assert_not_awaited()
        mock_store.aupdate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_new_when_key_not_found(self, mock_store, config):
        """memory_key provided but no existing match → store new."""
        mock_store.asearch.return_value = []
        result = await _do_write(
            mock_store, config, "store", "User's name is Prashant",
            "", MemoryType.EPISODIC, "general",
            {"memory_key": "user_name"}, "merge",
        )
        assert result["status"] == "stored"
        mock_store.astore.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_proceeds_when_no_duplicate(self, mock_store, config):
        """No match on key or similarity → normal store."""
        mock_store.asearch.return_value = []
        result = await _do_write(
            mock_store, config, "store", "Brand new fact",
            "", MemoryType.EPISODIC, "general", None, "merge",
        )
        assert result["status"] == "stored"
        mock_store.astore.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_proceeds_when_dedup_search_fails(self, mock_store, config):
        """If both dedup searches raise, fall through to normal store."""
        mock_store.asearch.side_effect = RuntimeError("network error")
        mock_store.astore.return_value = "new-id"
        result = await _do_write(
            mock_store, config, "store", "Some content",
            "", MemoryType.EPISODIC, "general", None, "merge",
        )
        assert result["status"] == "stored"

    @pytest.mark.asyncio
    async def test_dedup_does_not_affect_update_action(self, mock_store, config):
        """Dedup only applies to 'store'; 'update' should not trigger it."""
        result = await _do_write(
            mock_store, config, "update", "new text", "m1",
            MemoryType.EPISODIC, "general", {"x": 1}, "replace",
        )
        assert result["status"] == "updated"
        mock_store.asearch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dedup_does_not_affect_delete_action(self, mock_store, config):
        """Dedup only applies to 'store'; 'delete' should not trigger it."""
        result = await _do_write(
            mock_store, config, "delete", "", "m1",
            MemoryType.EPISODIC, "general", None, "merge",
        )
        assert result["status"] == "deleted"
        mock_store.asearch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_identical_threshold_is_valid(self):
        """Ensure threshold constant is valid."""
        assert 0 < _IDENTICAL_SCORE_THRESHOLD <= 1.0
