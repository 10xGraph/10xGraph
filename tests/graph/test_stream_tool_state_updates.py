"""Tests for streaming tool state updates and Command/handoff handling.

Tests that ToolResult state updates and Command/handoff work correctly
in both invoke and stream modes of the node handlers.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.graph.utils.invoke_node_handler import InvokeNodeHandler
from tenxgraph.core.graph.utils.stream_node_handler import StreamNodeHandler
from tenxgraph.core.state import AgentState, Message, ToolResult
from tenxgraph.core.state.message_block import ToolCallBlock, ToolResultBlock
from tenxgraph.core.state.stream_chunks import StreamChunk
from tenxgraph.utils import CallbackManager
from tenxgraph.utils.command import Command


class CustomState(AgentState):
    """Custom state with extra fields for testing state updates."""

    status: str = "pending"
    counter: int = 0
    label: str = ""


class TestToolResultInvoke:
    """Test ToolResult state updates via InvokeNodeHandler."""

    @pytest.mark.asyncio
    async def test_single_tool_result_updates_state(self):
        """Single tool returning ToolResult updates custom state fields."""

        def update_status(tool_call_id: str | None = None) -> ToolResult:
            return ToolResult(message="Status updated", state={"status": "complete"})

        tool_node = ToolNode([update_status])
        handler = InvokeNodeHandler("tools", tool_node)

        message = Message.text_message(role="assistant", content="call tools")
        message.tools_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "update_status", "arguments": "{}"},
            }
        ]

        state = CustomState(context=[message])
        config = {"thread_id": "t1", "run_id": "r1"}

        result = await handler.invoke(config, state)

        # The merged result should be a dict with state and messages
        assert isinstance(result, dict)
        assert result["state"].status == "complete"
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], Message)

    @pytest.mark.asyncio
    async def test_multiple_tool_results_merge_state(self):
        """Multiple tools returning ToolResult should merge their state updates."""

        def update_status(tool_call_id: str | None = None) -> ToolResult:
            return ToolResult(message="Status set", state={"status": "done"})

        def update_counter(tool_call_id: str | None = None) -> ToolResult:
            return ToolResult(message="Counter set", state={"counter": 42})

        tool_node = ToolNode([update_status, update_counter])
        handler = InvokeNodeHandler("tools", tool_node)

        message = Message.text_message(role="assistant", content="call tools")
        message.tools_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "update_status", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "update_counter", "arguments": "{}"},
            },
        ]

        state = CustomState(context=[message])
        config = {"thread_id": "t1", "run_id": "r1"}

        result = await handler.invoke(config, state)

        assert isinstance(result, dict)
        # Both state fields should be updated
        assert result["state"].status == "done"
        assert result["state"].counter == 42
        assert len(result["messages"]) == 2

    @pytest.mark.asyncio
    async def test_mixed_tool_result_and_message(self):
        """Mix of ToolResult and plain Message tools should merge correctly."""

        def update_label(tool_call_id: str | None = None) -> ToolResult:
            return ToolResult(message="Label set", state={"label": "important"})

        def plain_tool(tool_call_id: str | None = None) -> Message:
            return Message.tool_message(
                content=[
                    ToolResultBlock(call_id=tool_call_id or "", output="plain result"),
                ],
            )

        tool_node = ToolNode([update_label, plain_tool])
        handler = InvokeNodeHandler("tools", tool_node)

        message = Message.text_message(role="assistant", content="call tools")
        message.tools_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "update_label", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "plain_tool", "arguments": "{}"},
            },
        ]

        state = CustomState(context=[message])
        config = {"thread_id": "t1", "run_id": "r1"}

        result = await handler.invoke(config, state)

        assert isinstance(result, dict)
        assert result["state"].label == "important"
        assert len(result["messages"]) == 2


class TestToolResultStream:
    """Test ToolResult state updates via StreamNodeHandler."""

    @pytest.mark.asyncio
    async def test_single_tool_result_state_update_in_stream(self):
        """Single tool returning ToolResult should emit state update in stream."""

        def update_status(tool_call_id: str | None = None) -> ToolResult:
            return ToolResult(message="Status updated", state={"status": "complete"})

        tool_node = ToolNode([update_status])
        handler = StreamNodeHandler("tools", tool_node)

        message = Message.text_message(role="assistant", content="call tools")
        message.tools_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "update_status", "arguments": "{}"},
            }
        ]

        state = CustomState(context=[message])
        config = {"thread_id": "t1", "run_id": "r1"}

        messages_collected = []
        state_dicts = []

        async for item in handler.stream(config, state):
            if isinstance(item, Message):
                messages_collected.append(item)
            elif isinstance(item, dict) and "is_non_streaming" in item:
                state_dicts.append(item)

        # Should have at least one message from the tool
        assert len(messages_collected) >= 1
        # Should have a state update dict
        assert len(state_dicts) == 1
        assert state_dicts[0]["state"].status == "complete"

    @pytest.mark.asyncio
    async def test_multiple_tool_results_merge_state_in_stream(self):
        """Multiple ToolResult tools merge state correctly in stream mode."""

        def update_status(tool_call_id: str | None = None) -> ToolResult:
            return ToolResult(message="Status set", state={"status": "done"})

        def update_counter(tool_call_id: str | None = None) -> ToolResult:
            return ToolResult(message="Counter set", state={"counter": 42})

        tool_node = ToolNode([update_status, update_counter])
        handler = StreamNodeHandler("tools", tool_node)

        message = Message.text_message(role="assistant", content="call tools")
        message.tools_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "update_status", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "update_counter", "arguments": "{}"},
            },
        ]

        state = CustomState(context=[message])
        config = {"thread_id": "t1", "run_id": "r1"}

        messages_collected = []
        state_dicts = []

        async for item in handler.stream(config, state):
            if isinstance(item, Message):
                messages_collected.append(item)
            elif isinstance(item, dict) and "is_non_streaming" in item:
                state_dicts.append(item)

        assert len(messages_collected) >= 2
        assert len(state_dicts) == 1
        merged = state_dicts[0]["state"]
        assert merged.status == "done"
        assert merged.counter == 42

    @pytest.mark.asyncio
    async def test_mixed_tool_result_and_message_in_stream(self):
        """Mix of ToolResult and plain Message tools works in stream mode."""

        def update_label(tool_call_id: str | None = None) -> ToolResult:
            return ToolResult(message="Label set", state={"label": "important"})

        def plain_tool(tool_call_id: str | None = None) -> Message:
            return Message.tool_message(
                content=[
                    ToolResultBlock(call_id=tool_call_id or "", output="plain result"),
                ],
            )

        tool_node = ToolNode([update_label, plain_tool])
        handler = StreamNodeHandler("tools", tool_node)

        message = Message.text_message(role="assistant", content="call tools")
        message.tools_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "update_label", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "plain_tool", "arguments": "{}"},
            },
        ]

        state = CustomState(context=[message])
        config = {"thread_id": "t1", "run_id": "r1"}

        messages_collected = []
        state_dicts = []

        async for item in handler.stream(config, state):
            if isinstance(item, Message):
                messages_collected.append(item)
            elif isinstance(item, dict) and "is_non_streaming" in item:
                state_dicts.append(item)

        # Both tools should produce messages
        assert len(messages_collected) >= 2
        # State update should be emitted (because update_label returns ToolResult)
        assert len(state_dicts) == 1
        assert state_dicts[0]["state"].label == "important"

    @pytest.mark.asyncio
    async def test_plain_tools_no_state_dict_emitted(self):
        """When no tools return ToolResult, no state update dict should be emitted."""

        def plain_tool(value: str, tool_call_id: str | None = None) -> Message:
            return Message.tool_message(
                content=[
                    ToolResultBlock(call_id=tool_call_id or "", output=f"result: {value}"),
                ],
            )

        tool_node = ToolNode([plain_tool])
        handler = StreamNodeHandler("tools", tool_node)

        message = Message.text_message(role="assistant", content="call tools")
        message.tools_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "plain_tool", "arguments": '{"value": "test"}'},
            }
        ]

        state = CustomState(context=[message])
        config = {"thread_id": "t1", "run_id": "r1"}

        state_dicts = []
        messages_collected = []

        async for item in handler.stream(config, state):
            if isinstance(item, dict) and "is_non_streaming" in item:
                state_dicts.append(item)
            elif isinstance(item, Message):
                messages_collected.append(item)

        assert len(messages_collected) == 1
        # No state update dict should be emitted for plain tools
        assert len(state_dicts) == 0


class TestCommandHandoffStream:
    """Test Command/handoff handling in streaming mode."""

    @pytest.mark.asyncio
    async def test_handoff_yields_command_in_stream(self):
        """Handoff tool should yield Command with goto in stream mode."""

        def transfer_to_specialist(tool_call_id: str | None = None) -> str:
            return "Transferring to specialist"

        tool_node = ToolNode([transfer_to_specialist])
        handler = StreamNodeHandler("tools", tool_node)

        message = Message.text_message(role="assistant", content="call tools")
        message.tools_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "transfer_to_specialist",
                    "arguments": "{}",
                },
            }
        ]

        # Patch is_handoff_tool to detect our tool as a handoff
        from unittest.mock import patch

        with patch(
            "tenxgraph.prebuilt.tools.handoff.is_handoff_tool",
            return_value=(True, "specialist_agent"),
        ):
            state = CustomState(context=[message])
            config = {"thread_id": "t1", "run_id": "r1"}

            commands = []
            state_dicts = []

            async for item in handler.stream(config, state):
                if isinstance(item, Command):
                    commands.append(item)
                elif isinstance(item, dict) and "is_non_streaming" in item:
                    state_dicts.append(item)

            # Should get a Command with goto
            assert len(commands) == 1
            assert commands[0].goto == "specialist_agent"
            # Should also get a state dict with next_node
            assert len(state_dicts) == 1
            assert state_dicts[0]["next_node"] == "specialist_agent"

    @pytest.mark.asyncio
    async def test_handoff_in_invoke_returns_command(self):
        """Handoff tool should return Command in invoke mode."""

        def transfer_to_specialist(tool_call_id: str | None = None) -> str:
            return "Transferring to specialist"

        tool_node = ToolNode([transfer_to_specialist])
        handler = InvokeNodeHandler("tools", tool_node)

        message = Message.text_message(role="assistant", content="call tools")
        message.tools_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "transfer_to_specialist",
                    "arguments": "{}",
                },
            }
        ]

        from unittest.mock import patch

        with patch(
            "tenxgraph.prebuilt.tools.handoff.is_handoff_tool",
            return_value=(True, "specialist_agent"),
        ):
            state = CustomState(context=[message])
            config = {"thread_id": "t1", "run_id": "r1"}

            result = await handler.invoke(config, state)

            assert isinstance(result, Command)
            assert result.goto == "specialist_agent"


class TestToolNodeStreamDictHandling:
    """Test ToolNode.stream() handles dict results from _internal_execute."""

    @pytest.mark.asyncio
    async def test_stream_yields_dict_for_tool_result(self):
        """ToolNode.stream should yield dict when tool returns ToolResult."""

        def stateful_tool(tool_call_id: str | None = None) -> ToolResult:
            return ToolResult(message="done", state={"status": "finished"})

        tool_node = ToolNode([stateful_tool])

        state = CustomState()
        config = {}
        callback_mgr = MagicMock(spec=CallbackManager)
        callback_mgr.execute_before_invoke = AsyncMock(side_effect=lambda ctx, data: data)
        callback_mgr.execute_after_invoke = AsyncMock(
            side_effect=lambda ctx, input_data, result: result
        )

        results = []
        async for item in tool_node.stream(
            name="stateful_tool",
            args={},
            tool_call_id="call_1",
            config=config,
            state=state,
            callback_manager=callback_mgr,
        ):
            results.append(item)

        assert len(results) == 1
        result = results[0]
        # Should be a dict with state and messages
        assert isinstance(result, dict)
        assert "state" in result
        assert "messages" in result
        assert result["state"].status == "finished"
        msg = result["messages"]
        assert isinstance(msg, Message)
        assert msg.content[0].output == "done"

    @pytest.mark.asyncio
    async def test_stream_yields_message_for_plain_tool(self):
        """ToolNode.stream should yield Message for plain tools (no ToolResult)."""

        def plain_tool(value: str, tool_call_id: str | None = None) -> str:
            return f"result: {value}"

        tool_node = ToolNode([plain_tool])

        state = AgentState()
        config = {}
        callback_mgr = MagicMock(spec=CallbackManager)
        callback_mgr.execute_before_invoke = AsyncMock(side_effect=lambda ctx, data: data)
        callback_mgr.execute_after_invoke = AsyncMock(
            side_effect=lambda ctx, input_data, result: result
        )

        results = []
        async for item in tool_node.stream(
            name="plain_tool",
            args={"value": "test"},
            tool_call_id="call_1",
            config=config,
            state=state,
            callback_manager=callback_mgr,
        ):
            results.append(item)

        assert len(results) == 1
        assert isinstance(results[0], Message)
