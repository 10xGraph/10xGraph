"""
Unit tests for the TAF callback system.
"""

import pytest

from tenxgraph.utils import (
    AfterInvokeCallback,
    BeforeInvokeCallback,
    CallbackContext,
    CallbackManager,
    InvocationType,
    OnErrorCallback,
)


class TestCallbackContext:
    """Test CallbackContext dataclass."""

    def test_callback_context_creation(self):
        """Test creating a callback context."""
        context = CallbackContext(
            invocation_type=InvocationType.AI,
            node_name="test_node",
            function_name="test_function",
            metadata={"key": "value"},
        )

        assert context.invocation_type == InvocationType.AI
        assert context.node_name == "test_node"
        assert context.function_name == "test_function"
        assert context.metadata == {"key": "value"}


class MockBeforeInvokeCallback(BeforeInvokeCallback[CallbackContext, bool]):
    """Mock before invoke callback for testing."""

    def __init__(self, return_value: bool = True):
        self.return_value = return_value
        self.called = False
        self.call_context = None

    async def execute(self, context: CallbackContext) -> bool:
        """Mock execution."""
        self.called = True
        self.call_context = context
        return self.return_value


class MockAfterInvokeCallback(AfterInvokeCallback[CallbackContext, str, None]):
    """Mock after invoke callback for testing."""

    def __init__(self):
        self.called = False
        self.call_context = None
        self.call_result = None

    async def execute(self, context: CallbackContext, result: str) -> None:
        """Mock execution."""
        self.called = True
        self.call_context = context
        self.call_result = result


class MockErrorCallback(OnErrorCallback[CallbackContext, Exception, None]):
    """Mock error callback for testing."""

    def __init__(self):
        self.called = False
        self.call_context = None
        self.call_error = None

    async def execute(self, context: CallbackContext, error: Exception) -> None:
        """Mock execution."""
        self.called = True
        self.call_context = context
        self.call_error = error


class TestCallbackManager:
    """Test CallbackManager functionality."""

    def test_add_before_invoke_callback(self):
        """Test adding before invoke callbacks."""
        manager = CallbackManager()
        callback = MockBeforeInvokeCallback()

        manager.add_before_invoke_callback(callback)

        assert len(manager._before_invoke_callbacks) == 1
        assert manager._before_invoke_callbacks[0] == callback

    def test_add_after_invoke_callback(self):
        """Test adding after invoke callbacks."""
        manager = CallbackManager()
        callback = MockAfterInvokeCallback()

        manager.add_after_invoke_callback(callback)

        assert len(manager._after_invoke_callbacks) == 1
        assert manager._after_invoke_callbacks[0] == callback

    def test_add_error_callback(self):
        """Test adding error callbacks."""
        manager = CallbackManager()
        callback = MockErrorCallback()

        manager.add_error_callback(callback)

        assert len(manager._error_callbacks) == 1
        assert manager._error_callbacks[0] == callback

    @pytest.mark.asyncio
    async def test_execute_before_invoke_callbacks_all_allow(self):
        """Test before invoke callbacks when all allow execution."""
        manager = CallbackManager()
        callback1 = MockBeforeInvokeCallback(True)
        callback2 = MockBeforeInvokeCallback(True)

        manager.add_before_invoke_callback(callback1)
        manager.add_before_invoke_callback(callback2)

        context = CallbackContext(
            invocation_type=InvocationType.AI, node_name="test", function_name="test", metadata={}
        )

        result = await manager.execute_before_invoke_callbacks(context)

        assert result is True
        assert callback1.called
        assert callback2.called
        assert callback1.call_context == context
        assert callback2.call_context == context

    @pytest.mark.asyncio
    async def test_execute_before_invoke_callbacks_one_blocks(self):
        """Test before invoke callbacks when one blocks execution."""
        manager = CallbackManager()
        callback1 = MockBeforeInvokeCallback(True)
        callback2 = MockBeforeInvokeCallback(False)  # This one blocks
        callback3 = MockBeforeInvokeCallback(True)

        manager.add_before_invoke_callback(callback1)
        manager.add_before_invoke_callback(callback2)
        manager.add_before_invoke_callback(callback3)

        context = CallbackContext(
            invocation_type=InvocationType.TOOL, node_name="test", function_name="test", metadata={}
        )

        result = await manager.execute_before_invoke_callbacks(context)

        assert result is False
        assert callback1.called
        assert callback2.called
        # callback3 should not be called since callback2 blocked
        assert not callback3.called

    @pytest.mark.asyncio
    async def test_execute_after_invoke_callbacks(self):
        """Test after invoke callbacks execution."""
        manager = CallbackManager()
        callback1 = MockAfterInvokeCallback()
        callback2 = MockAfterInvokeCallback()

        manager.add_after_invoke_callback(callback1)
        manager.add_after_invoke_callback(callback2)

        context = CallbackContext(
            invocation_type=InvocationType.MCP, node_name="test", function_name="test", metadata={}
        )
        result = "test_result"

        await manager.execute_after_invoke_callbacks(context, result)

        assert callback1.called
        assert callback2.called
        assert callback1.call_context == context
        assert callback2.call_context == context
        assert callback1.call_result == result
        assert callback2.call_result == result

    @pytest.mark.asyncio
    async def test_execute_error_callbacks(self):
        """Test error callbacks execution."""
        manager = CallbackManager()
        callback1 = MockErrorCallback()
        callback2 = MockErrorCallback()

        manager.add_error_callback(callback1)
        manager.add_error_callback(callback2)

        context = CallbackContext(
            invocation_type=InvocationType.AI, node_name="test", function_name="test", metadata={}
        )
        error = Exception("test error")

        await manager.execute_error_callbacks(context, error)

        assert callback1.called
        assert callback2.called
        assert callback1.call_context == context
        assert callback2.call_context == context
        assert callback1.call_error == error
        assert callback2.call_error == error

    @pytest.mark.asyncio
    async def test_no_callbacks_registered(self):
        """Test behavior when no callbacks are registered."""
        manager = CallbackManager()

        context = CallbackContext(
            invocation_type=InvocationType.AI, node_name="test", function_name="test", metadata={}
        )

        # Should not raise exceptions
        result = await manager.execute_before_invoke_callbacks(context)
        assert result is True  # Default allow

        await manager.execute_after_invoke_callbacks(context, "result")
        await manager.execute_error_callbacks(context, Exception("error"))


class TestInvocationType:
    """Test InvocationType enum."""

    def test_invocation_type_values(self):
        """Test invocation type enum values."""
        assert InvocationType.AI.value == "AI"
        assert InvocationType.TOOL.value == "TOOL"
        assert InvocationType.MCP.value == "MCP"

    def test_invocation_type_members(self):
        """Test invocation type enum members."""
        assert len(InvocationType) == 3
        assert InvocationType.AI in InvocationType
        assert InvocationType.TOOL in InvocationType
        assert InvocationType.MCP in InvocationType


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
