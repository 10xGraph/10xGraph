"""Tests for BaseContextManager abstract class."""

import pytest
from unittest.mock import AsyncMock, Mock

from tenxgraph.core.state import AgentState
from tenxgraph.core.state.base_context import BaseContextManager
from tenxgraph.core.state import Message


class TestBaseContextManager:
    """Test suite for BaseContextManager abstract class."""

    def test_base_context_manager_is_abstract(self):
        """Test that BaseContextManager cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseContextManager() # type: ignore

    def test_concrete_implementation_must_implement_trim_context(self):
        """Test that concrete implementations must implement trim_context."""
        
        class IncompleteContextManager(BaseContextManager):
            async def atrim_context(self, state):
                return state
            # Missing trim_context implementation
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteContextManager()

    def test_concrete_implementation_must_implement_atrim_context(self):
        """Test that concrete implementations must implement atrim_context."""
        
        class IncompleteContextManager(BaseContextManager):
            def trim_context(self, state):
                return state
            # Missing atrim_context implementation
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteContextManager()

    def test_concrete_implementation_works(self):
        """Test that proper concrete implementation can be instantiated."""
        
        class ConcreteContextManager(BaseContextManager[AgentState]):
            def trim_context(self, state: AgentState) -> AgentState:
                # Keep only last 3 messages
                if len(state.context) > 3:
                    state.context = state.context[-3:]
                return state
            
            async def atrim_context(self, state: AgentState) -> AgentState:
                # Async version - keep only last 2 messages
                if len(state.context) > 2:
                    state.context = state.context[-2:]
                return state
        
        # Should not raise any exception
        manager = ConcreteContextManager()
        assert isinstance(manager, BaseContextManager)

    def test_sync_trim_context_functionality(self):
        """Test synchronous trim_context functionality."""
        
        class SyncContextManager(BaseContextManager[AgentState]):
            def trim_context(self, state: AgentState) -> AgentState:
                # Keep only last 2 messages
                if len(state.context) > 2:
                    state.context = state.context[-2:]
                return state
            
            async def atrim_context(self, state: AgentState) -> AgentState:
                return self.trim_context(state)
        
        manager = SyncContextManager()
        
        # Create state with multiple messages
        messages = [
            Message.text_message("Message 1", "user"),
            Message.text_message("Message 2", "assistant"),
            Message.text_message("Message 3", "user"),
            Message.text_message("Message 4", "assistant"),
            Message.text_message("Message 5", "user"),
        ]
        
        state = AgentState(context=messages)
        assert len(state.context) == 5
        
        # Trim context
        trimmed_state = manager.trim_context(state)
        
        # Should keep only last 2 messages
        assert len(trimmed_state.context) == 2
        assert trimmed_state.context[0].text() == "Message 4"
        assert trimmed_state.context[1].text() == "Message 5"

    @pytest.mark.asyncio
    async def test_async_trim_context_functionality(self):
        """Test asynchronous atrim_context functionality."""
        
        class AsyncContextManager(BaseContextManager[AgentState]):
            def trim_context(self, state: AgentState) -> AgentState:
                if len(state.context) > 3:
                    state.context = state.context[-3:]
                return state
            
            async def atrim_context(self, state: AgentState) -> AgentState:
                # Async version - keep only last 1 message
                if len(state.context) > 1:
                    state.context = state.context[-1:]
                return state
        
        manager = AsyncContextManager()
        
        # Create state with multiple messages
        messages = [
            Message.text_message("Message 1", "user"),
            Message.text_message("Message 2", "assistant"),
            Message.text_message("Message 3", "user"),
        ]
        
        state = AgentState(context=messages)
        assert len(state.context) == 3
        
        # Trim context asynchronously
        trimmed_state = await manager.atrim_context(state)
        
        # Should keep only last 1 message
        assert len(trimmed_state.context) == 1
        assert trimmed_state.context[0].text() == "Message 3"

    def test_trim_context_with_empty_state(self):
        """Test trim_context with empty context."""
        
        class SimpleContextManager(BaseContextManager[AgentState]):
            def trim_context(self, state: AgentState) -> AgentState:
                return state
            
            async def atrim_context(self, state: AgentState) -> AgentState:
                return state
        
        manager = SimpleContextManager()
        state = AgentState(context=[])
        
        # Should not raise error with empty context
        trimmed_state = manager.trim_context(state)
        assert len(trimmed_state.context) == 0

    @pytest.mark.asyncio
    async def test_atrim_context_with_empty_state(self):
        """Test atrim_context with empty context."""
        
        class SimpleContextManager(BaseContextManager[AgentState]):
            def trim_context(self, state: AgentState) -> AgentState:
                return state
            
            async def atrim_context(self, state: AgentState) -> AgentState:
                return state
        
        manager = SimpleContextManager()
        state = AgentState(context=[])
        
        # Should not raise error with empty context
        trimmed_state = await manager.atrim_context(state)
        assert len(trimmed_state.context) == 0

    def test_generic_type_support(self):
        """Test that BaseContextManager supports generic typing."""
        
        class CustomState(AgentState):
            custom_field: str = "test"
        
        class TypedContextManager(BaseContextManager[CustomState]):
            def trim_context(self, state: CustomState) -> CustomState:
                return state
            
            async def atrim_context(self, state: CustomState) -> CustomState:
                return state
        
        manager = TypedContextManager()
        custom_state = CustomState(context=[], custom_field="custom_value")
        
        result = manager.trim_context(custom_state)
        assert isinstance(result, CustomState)
        assert result.custom_field == "custom_value"

    def test_state_modification_in_place(self):
        """Test that context managers can modify state in-place."""
        
        class ModifyingContextManager(BaseContextManager[AgentState]):
            def trim_context(self, state: AgentState) -> AgentState:
                # Modify context_summary instead of context
                state.context_summary = "Trimmed context summary"
                return state
            
            async def atrim_context(self, state: AgentState) -> AgentState:
                return self.trim_context(state)
        
        manager = ModifyingContextManager()
        state = AgentState(context=[Message.text_message("Test", "user")])
        
        result = manager.trim_context(state)
        assert result.context_summary == "Trimmed context summary"
        assert len(result.context) == 1  # Original context unchanged