"""Tests for tenxgraph.store.long_term_memory module."""

import asyncio
import json
from typing import get_args
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.store.long_term_memory import (
    DEFAULT_READ_MODE,
    _IDENTICAL_SCORE_THRESHOLD,
    ReadMode,
    _do_write,
    _find_duplicate_by_key,
    _find_duplicate_by_similarity,
    _format_search_results,
    _strip_thread_id,
    _validate_memory_type,
    get_agent_memory_system_prompt,
)
from tenxgraph.prebuilt.tools.memory import (
    make_agent_memory_tool,
    make_user_memory_tool,
)
from tenxgraph.storage.store.store_schema import MemoryTypeName
from tenxgraph.storage.store.memory_config import (
    AgentMemoryConfig,
    MemoryConfig,
    UserMemoryConfig,
)
from tenxgraph.storage.store.store_schema import MemorySearchResult, MemoryType
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.qa.testing import InMemoryStore
from tenxgraph.utils.background_task_manager import BackgroundTaskManager


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

    def test_invalid_type_raises_naming_valid_types(self):
        with pytest.raises(ValueError, match=r"unknown memory_type 'invalid'.*semantic"):
            _validate_memory_type("invalid")

    def test_accepts_enum_instance(self):
        assert _validate_memory_type(MemoryType.ENTITY) is MemoryType.ENTITY

    def test_literal_matches_enum(self):
        assert set(get_args(MemoryTypeName)) == {m.value for m in MemoryType}

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

        assert set(user_props) == {
            "action",
            "text",
            "memory_key",
            "memory_type",
            "category",
            "limit",
        }
        assert set(agent_props) == {"query", "memory_type", "category", "limit"}
        assert "memory_id" not in user_props
        assert user_props["action"]["enum"] == ["search", "remember", "forget"]
        assert "action" not in agent_props

    def test_preload_does_not_expose_agent_tool(self):
        memory = MemoryConfig(
            retrieval_mode="preload",
            user_memory=UserMemoryConfig(enabled=False),
            agent_memory=AgentMemoryConfig(enabled=True),
        )

        assert memory.model_facing_tools() == []


# ---------------------------------------------------------------------------
# user_memory_tool: memory_key and forget
# ---------------------------------------------------------------------------


class _KeyedStore(InMemoryStore):
    """InMemoryStore whose search honours exact metadata filters, like the real stores."""

    async def asearch(self, config, query, memory_type=None, limit=10, filters=None, **kwargs):
        hits = [
            m
            for m in self.memories.values()
            if (not memory_type or m.memory_type == memory_type)
            and all(m.metadata.get(k) == v for k, v in (filters or {}).items())
            and (filters or query.lower() in m.content.lower())
        ]
        return hits[:limit]


async def _call(tool_fn, store, task_manager, **kwargs):
    return json.loads(
        await tool_fn(store=store, task_manager=task_manager, config={"user_id": "u1"}, **kwargs)
    )


class TestUserMemoryKeyAndForget:
    @pytest.mark.asyncio
    async def test_remember_same_key_twice_keeps_one_memory_with_new_text(self):
        store = _KeyedStore()
        manager = BackgroundTaskManager()
        tool_fn = make_user_memory_tool(MemoryConfig(user_memory=UserMemoryConfig()))

        for text in ("User uses pip", "User switched to uv"):
            await _call(
                tool_fn, store, manager, action="remember", text=text, memory_key="package_manager"
            )
            await manager.wait_for_all()

        assert [m.content for m in store.memories.values()] == ["User switched to uv"]

    @pytest.mark.asyncio
    async def test_forget_by_key_removes_only_that_memory(self):
        store = _KeyedStore()
        manager = BackgroundTaskManager()
        tool_fn = make_user_memory_tool(MemoryConfig(user_memory=UserMemoryConfig()))
        await _call(
            tool_fn, store, manager, action="remember", text="Uses vim", memory_key="editor"
        )
        await _call(tool_fn, store, manager, action="remember", text="Likes tea")
        await manager.wait_for_all()

        result = await _call(tool_fn, store, manager, action="forget", memory_key="editor")

        assert result == {"status": "forgotten", "memory_key": "editor"}
        assert [m.content for m in store.memories.values()] == ["Likes tea"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", [None, "", "   "])
    async def test_forget_without_key_is_an_error_and_deletes_nothing(
        self, mock_store, mock_task_manager, key
    ):
        tool_fn = make_user_memory_tool(MemoryConfig(user_memory=UserMemoryConfig()))

        result = await _call(
            tool_fn, mock_store, mock_task_manager, action="forget", memory_key=key
        )

        assert "error" in result
        mock_store.aforget_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_forget_flushes_pending_writes_then_deletes_by_key(
        self, mock_store, mock_task_manager
    ):
        order = []
        mock_task_manager.get_task_count.return_value = 1
        mock_task_manager.wait_for_all.side_effect = lambda **kw: order.append("flush")
        mock_store.aforget_memory.side_effect = lambda *a, **kw: order.append("forget")
        tool_fn = make_user_memory_tool(MemoryConfig(user_memory=UserMemoryConfig()))

        await tool_fn(
            action="forget",
            memory_key="editor",
            store=mock_store,
            task_manager=mock_task_manager,
            config={"user_id": "u1", "thread_id": "t1"},
        )

        assert order == ["flush", "forget"]
        mock_store.aforget_memory.assert_awaited_once_with(
            {"user_id": "u1"}, filters={"memory_key": "editor"}
        )

    @pytest.mark.asyncio
    async def test_key_is_stripped(self, mock_store, mock_task_manager):
        tool_fn = make_user_memory_tool(MemoryConfig(user_memory=UserMemoryConfig()))

        result = await _call(
            tool_fn, mock_store, mock_task_manager, action="forget", memory_key="  editor \n"
        )

        assert result["memory_key"] == "editor"
        mock_store.aforget_memory.assert_awaited_once_with(
            {"user_id": "u1"}, filters={"memory_key": "editor"}
        )

    @pytest.mark.asyncio
    async def test_remember_puts_stripped_key_in_metadata(self, mock_store):
        manager = BackgroundTaskManager()
        tool_fn = make_user_memory_tool(MemoryConfig(user_memory=UserMemoryConfig()))

        await _call(tool_fn, mock_store, manager, action="remember", text="x", memory_key=" k ")
        await manager.wait_for_all()

        assert mock_store.astore.call_args.kwargs["metadata"]["memory_key"] == "k"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["remember", "forget"])
    async def test_over_long_key_is_an_error(self, mock_store, mock_task_manager, action):
        tool_fn = make_user_memory_tool(MemoryConfig(user_memory=UserMemoryConfig()))

        result = await _call(
            tool_fn, mock_store, mock_task_manager, action=action, text="x", memory_key="k" * 129
        )

        assert "error" in result
        mock_store.aforget_memory.assert_not_called()
        mock_task_manager.create_task.assert_not_called()

    def test_system_prompt_explains_keys_and_forget(self):
        prompt = get_agent_memory_system_prompt(
            MemoryConfig(retrieval_mode="postload", user_memory=UserMemoryConfig())
        )
        assert "memory_key" in prompt
        assert "action='forget'" in prompt
        assert "Do not invent memory ids" in prompt


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


class TestStrictMemoryType:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["search", "remember"])
    async def test_unknown_type_is_an_error_and_nothing_runs(
        self, mock_store, mock_task_manager, action
    ):
        tool_fn = make_user_memory_tool(MemoryConfig(user_memory=UserMemoryConfig()))

        result = await _call(
            tool_fn,
            mock_store,
            mock_task_manager,
            action=action,
            text="x",
            memory_type="preference",
        )

        assert result["error"].startswith("unknown memory_type 'preference'; use one of: ")
        mock_store.asearch.assert_not_called()
        mock_store.astore.assert_not_called()
        mock_task_manager.create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_agent_tool_unknown_type_is_an_error(self, mock_store, mock_task_manager):
        tool_fn = make_agent_memory_tool(
            MemoryConfig(agent_memory=AgentMemoryConfig(enabled=True, agent_id="a1"))
        )

        result = json.loads(
            await tool_fn(
                query="x",
                memory_type="nope",
                store=mock_store,
                task_manager=mock_task_manager,
                config={"user_id": "u1"},
            )
        )

        assert result["error"].startswith("unknown memory_type 'nope'")
        mock_store.asearch.assert_not_called()

    @pytest.mark.asyncio
    async def test_schema_has_enum_on_both_tools(self):
        memory = MemoryConfig(agent_memory=AgentMemoryConfig(enabled=True))
        schemas = await ToolNode(memory.model_facing_tools()).all_tools()
        by_name = {
            s["function"]["name"]: s["function"]["parameters"]["properties"] for s in schemas
        }
        expected = [m.value for m in MemoryType]
        for name in ("user_memory_tool", "agent_memory_tool"):
            assert sorted(by_name[name]["memory_type"]["enum"]) == sorted(expected)

    @pytest.mark.parametrize("cls", [UserMemoryConfig, AgentMemoryConfig])
    def test_config_rejects_unknown_type(self, cls):
        with pytest.raises(ValidationError):
            cls(memory_type="nope")

    def test_config_accepts_known_type(self):
        assert UserMemoryConfig(memory_type="semantic").memory_type == "semantic"
