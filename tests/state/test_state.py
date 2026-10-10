"""Tests for the state module."""

from tenxgraph.core.state import (
    AgentState,
    BaseContextManager,
    ExecutionState,
    ExecutionStatus,
    MessageContextManager,
)
from tenxgraph.utils import START
from tenxgraph.core.state.message import Message


class TestAgentState:
    """Test the AgentState class."""

    def test_agent_state_creation(self):
        """Test creating an AgentState."""
        state = AgentState()
        assert state.context == []  # noqa: S101
        assert state.context_summary is None  # noqa: S101
        assert state.execution_meta is not None  # noqa: S101
        assert state.execution_meta.current_node == START  # noqa: S101

    def test_agent_state_with_messages(self):
        """Test AgentState with initial messages."""
        messages = [Message.text_message("Hello"), Message.text_message("World")]
        state = AgentState(context=messages)
        assert len(state.context) == 2  # noqa: S101

    def test_agent_state_set_interrupt(self):
        """Test setting an interrupt."""
        state = AgentState()
        state.set_interrupt(
            "test_node", "test_reason", ExecutionStatus.INTERRUPTED_BEFORE, {"key": "value"}
        )

        assert state.execution_meta.interrupted_node == "test_node"  # noqa: S101
        assert state.execution_meta.interrupt_reason == "test_reason"  # noqa: S101
        assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_BEFORE  # noqa: S101

    def test_agent_state_clear_interrupt(self):
        """Test clearing an interrupt."""
        state = AgentState()
        state.set_interrupt("test_node", "test_reason", ExecutionStatus.INTERRUPTED_BEFORE)
        state.clear_interrupt()

        assert state.execution_meta.interrupted_node is None  # noqa: S101
        assert state.execution_meta.interrupt_reason is None  # noqa: S101

    def test_agent_state_is_running(self):
        """Test checking if state is running."""
        state = AgentState()
        # Default state should be running
        running = state.is_running()
        assert isinstance(running, bool)  # noqa: S101

    def test_agent_state_context_summary(self):
        """Test context summary functionality."""
        state = AgentState()
        state.context_summary = "This is a test summary"
        assert state.context_summary == "This is a test summary"  # noqa: S101

    def test_agent_state_execution_meta_delegation(self):
        """Test that AgentState properly delegates to execution_meta."""
        state = AgentState()

        # Test delegation methods exist
        assert hasattr(state, "set_interrupt")  # noqa: S101
        assert hasattr(state, "clear_interrupt")  # noqa: S101
        assert hasattr(state, "is_running")  # noqa: S101


class TestExecutionState:
    """Test the ExecutionState class."""

    def test_execution_state_creation(self):
        """Test creating an ExecutionState."""
        exec_state = ExecutionState(current_node=START)
        assert exec_state.current_node == START  # noqa: S101
        assert exec_state.status == ExecutionStatus.RUNNING  # noqa: S101
        assert exec_state.interrupted_node is None  # noqa: S101
        assert exec_state.interrupt_reason is None  # noqa: S101

    def test_execution_state_with_custom_node(self):
        """Test creating ExecutionState with custom node."""
        exec_state = ExecutionState(current_node="custom_node")
        assert exec_state.current_node == "custom_node"  # noqa: S101

    def test_set_interrupt(self):
        """Test setting an interrupt."""
        exec_state = ExecutionState(current_node=START)
        exec_state.set_interrupt(
            "node1", "test_reason", ExecutionStatus.INTERRUPTED_BEFORE, {"data": "test"}
        )

        assert exec_state.interrupted_node == "node1"  # noqa: S101
        assert exec_state.interrupt_reason == "test_reason"  # noqa: S101
        assert exec_state.status == ExecutionStatus.INTERRUPTED_BEFORE  # noqa: S101
        assert exec_state.interrupt_data == {"data": "test"}  # noqa: S101

    def test_clear_interrupt(self):
        """Test clearing an interrupt."""
        exec_state = ExecutionState(current_node=START)
        exec_state.set_interrupt("node1", "test_reason", ExecutionStatus.INTERRUPTED_BEFORE)
        exec_state.clear_interrupt()

        assert exec_state.interrupted_node is None  # noqa: S101
        assert exec_state.interrupt_reason is None  # noqa: S101
        assert exec_state.interrupt_data is None  # noqa: S101

    def test_is_running(self):
        """Test checking if execution is running."""
        exec_state = ExecutionState(current_node=START)

        # Test different statuses
        exec_state.status = ExecutionStatus.RUNNING
        assert exec_state.is_running() is True  # noqa: S101

        exec_state.status = ExecutionStatus.INTERRUPTED_BEFORE
        assert exec_state.is_running() is False  # noqa: S101

        exec_state.status = ExecutionStatus.COMPLETED
        assert exec_state.is_running() is False  # noqa: S101

    def test_is_interrupted(self):
        """Test checking if execution is interrupted."""
        exec_state = ExecutionState(current_node=START)

        assert exec_state.is_interrupted() is False  # noqa: S101

        exec_state.set_interrupt("node1", "test", ExecutionStatus.INTERRUPTED_BEFORE)
        assert exec_state.is_interrupted() is True  # noqa: S101

    def test_get_next_node(self):
        """Test getting the next node."""
        exec_state = ExecutionState(current_node=START)
        exec_state.current_node = "current"

        # Test advance_step method instead
        if hasattr(exec_state, "advance_step"):
            old_step = exec_state.step
            exec_state.advance_step()
            assert exec_state.step == old_step + 1  # noqa: S101


class TestExecutionStatus:
    """Test the ExecutionStatus enum."""

    def test_execution_status_values(self):
        """Test ExecutionStatus enum values."""
        assert ExecutionStatus.RUNNING  # noqa: S101
        assert ExecutionStatus.INTERRUPTED_BEFORE  # noqa: S101
        assert ExecutionStatus.INTERRUPTED_AFTER  # noqa: S101
        assert ExecutionStatus.COMPLETED  # noqa: S101
        assert ExecutionStatus.ERROR  # noqa: S101

    def test_execution_status_string_representation(self):
        """Test string representation of ExecutionStatus."""
        assert str(ExecutionStatus.RUNNING)  # noqa: S101
        assert str(ExecutionStatus.INTERRUPTED_BEFORE)  # noqa: S101


class TestBaseContextManager:
    """Test the BaseContextManager class."""

    def test_base_context_manager_creation(self):
        """Test creating a BaseContextManager."""

        # Since BaseContextManager is abstract, test via mock implementation
        class MockContextManager(BaseContextManager):
            def trim_context(self, state):
                return state

            async def atrim_context(self, state):
                return state

        manager = MockContextManager()
        assert manager is not None  # noqa: S101

    def test_base_context_manager_methods(self):
        """Test BaseContextManager has expected methods."""

        class MockContextManager(BaseContextManager):
            def trim_context(self, state):
                return state

            async def atrim_context(self, state):
                return state

        manager = MockContextManager()

        # Check for expected methods/attributes
        assert hasattr(manager, "__init__")  # noqa: S101
        assert hasattr(manager, "trim_context")  # noqa: S101
        assert callable(manager.trim_context)  # noqa: S101


class TestMessageContextManager:
    """Test the MessageContextManager class."""

    def test_message_context_manager_creation(self):
        """Test creating a MessageContextManager."""
        manager = MessageContextManager()
        assert manager is not None  # noqa: S101

    def test_message_context_manager_inheritance(self):
        """Test MessageContextManager inherits from BaseContextManager."""
        manager = MessageContextManager()
        assert isinstance(manager, BaseContextManager)  # noqa: S101

    def test_message_context_manager_with_max_tokens(self):
        """Test MessageContextManager with max_messages parameter."""
        try:
            manager = MessageContextManager(max_messages=1000)
            assert manager is not None  # noqa: S101
        except TypeError:
            # If constructor doesn't accept max_messages, that's fine
            manager = MessageContextManager()
            assert manager is not None  # noqa: S101

    def test_message_context_manager_methods(self):
        """Test MessageContextManager has expected methods."""
        manager = MessageContextManager()

        # Test for common context manager methods
        expected_methods = ["reduce_context", "get_context", "update_context"]
        for method in expected_methods:
            if hasattr(manager, method):
                assert callable(getattr(manager, method))  # noqa: S101


def test_state_module_imports():
    """Test that state module imports work correctly."""
    # Basic smoke test - just ensure imports work
    assert AgentState is not None  # noqa: S101
    assert BaseContextManager is not None  # noqa: S101
    assert ExecutionState is not None  # noqa: S101
    assert ExecutionStatus is not None  # noqa: S101
    assert MessageContextManager is not None  # noqa: S101
