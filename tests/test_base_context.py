"""Tests for BaseContextManager: one `trim_context` method, sync or async."""

import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tenxgraph.core.graph.agent_internal.execution import AgentExecutionMixin
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.base_context import BaseContextManager
from tenxgraph.utils.callable_utils import call_sync_or_async


_ENGINE_TRIM = AgentExecutionMixin._trim_context


def _state(n: int = 5) -> AgentState:
    return AgentState(context=[Message.text_message(f"Message {i}", "user") for i in range(n)])


class SyncManager(BaseContextManager[AgentState]):
    def trim_context(self, state: AgentState) -> AgentState:
        state.context = state.context[-2:]
        return state


class AsyncManager(BaseContextManager[AgentState]):
    async def trim_context(self, state: AgentState) -> AgentState:
        state.context = state.context[-1:]
        return state


class TestBaseContextManager:
    def test_is_abstract(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseContextManager()  # type: ignore

    def test_trim_context_is_required(self):
        class Incomplete(BaseContextManager):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Incomplete()

    def test_only_trim_context_is_needed(self):
        assert isinstance(SyncManager(), BaseContextManager)
        assert isinstance(AsyncManager(), BaseContextManager)

    @pytest.mark.asyncio
    async def test_sync_trim_context_through_helper(self):
        result = await call_sync_or_async(SyncManager().trim_context, _state())
        assert [m.text() for m in result.context] == ["Message 3", "Message 4"]

    @pytest.mark.asyncio
    async def test_async_trim_context_through_helper(self):
        result = await call_sync_or_async(AsyncManager().trim_context, _state())
        assert [m.text() for m in result.context] == ["Message 4"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("manager_cls", [SyncManager, AsyncManager])
    async def test_engine_call_site_accepts_both(self, manager_cls):
        agent = SimpleNamespace(trim_context=True)
        result = await _ENGINE_TRIM(agent, _state(), context_manager=manager_cls())
        assert 1 <= len(result.context) <= 2

    @pytest.mark.asyncio
    async def test_atrim_context_warns_and_returns_trimmed_state(self):
        with pytest.warns(DeprecationWarning, match="atrim_context is deprecated"):
            result = await SyncManager().atrim_context(_state())
        assert len(result.context) == 2
        with pytest.warns(DeprecationWarning):
            result = await AsyncManager().atrim_context(_state())
        assert len(result.context) == 1

    @pytest.mark.asyncio
    async def test_framework_does_not_call_atrim_context(self):
        manager = SyncManager()
        manager.atrim_context = AsyncMock()  # type: ignore[method-assign]
        agent = SimpleNamespace(trim_context=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await _ENGINE_TRIM(agent, _state(), context_manager=manager)
        manager.atrim_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_1_0_subclass_overriding_both_still_works(self):
        class Old(BaseContextManager[AgentState]):
            def trim_context(self, state):
                state.context = state.context[-3:]
                return state

            async def atrim_context(self, state):
                return self.trim_context(state)

        result = await call_sync_or_async(Old().trim_context, _state())
        assert len(result.context) == 3
