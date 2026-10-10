"""Tests for message conversion utilities."""

import pytest
from unittest.mock import Mock, patch

from tenxgraph.core.state import AgentState, Message, ToolResultBlock
from tenxgraph.utils.converter import _convert_dict, convert_messages


class TestConverterUtils:
    """Test suite for message conversion utilities."""

    def test_convert_dict_user_message(self):
        """Test _convert_dict with a user message."""
        message = Message.text_message("Hello world", "user")
        
        result = _convert_dict(message)
        
        expected = {
            "role": "user",
            "content": "Hello world"
        }
        assert result == expected

    def test_convert_dict_assistant_message(self):
        """Test _convert_dict with an assistant message."""
        message = Message.text_message("Hello back", "assistant")
        
        result = _convert_dict(message)
        
        expected = {
            "role": "assistant",
            "content": "Hello back"
        }
        assert result == expected

    def test_convert_dict_system_message(self):
        """Test _convert_dict with a system message."""
        message = Message.text_message("You are a helpful assistant", "system")
        
        result = _convert_dict(message)
        
        expected = {
            "role": "system",
            "content": "You are a helpful assistant"
        }
        assert result == expected

    def test_convert_dict_tool_message(self):
        """Test _convert_dict with a tool result message."""
        # Create a tool result block
        tool_result_block = ToolResultBlock(
            call_id="call_123",
            output="Tool result content"
        )
        
        message = Message(
            message_id="msg_1",
            role="tool",
            content=[tool_result_block]
        )
        
        result = _convert_dict(message)
        
        expected = {
            "role": "tool",
            "content": "Tool result content",
            "tool_call_id": "call_123"
        }
        assert result == expected

    def test_convert_dict_tool_message_no_tool_result_block(self):
        """Test _convert_dict with tool message but no ToolResultBlock."""
        message = Message.text_message("Some tool content", "tool")
        
        result = _convert_dict(message)
        
        expected = {
            "role": "tool",
            "content": "Some tool content",
            "tool_call_id": ""
        }
        assert result == expected

    def test_convert_dict_tool_message_with_none_content(self):
        """Test _convert_dict tolerates malformed tool messages with missing content."""
        message = Message.text_message("placeholder", "tool")
        message.content = None # type: ignore

        result = _convert_dict(message)

        expected = {
            "role": "tool",
            "content": "",
            "tool_call_id": ""
        }
        assert result == expected

    def test_convert_dict_assistant_with_tool_calls(self):
        """Test _convert_dict with assistant message that has tool calls."""
        tool_call_dict = {
            "id": "call_123",
            "name": "test_function",
            "arguments": {"param": "value"}
        }
        
        message = Message(
            message_id="msg_1",
            role="assistant",
            content=[],
            tools_calls=[tool_call_dict]
        )
        
        result = _convert_dict(message)
        
        expected = {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call_dict]
        }
        assert result == expected

    def test_convert_dict_assistant_with_content_and_tool_calls(self):
        """Test _convert_dict with assistant message that has both content and tool calls."""
        tool_call_dict = {
            "id": "call_123", 
            "name": "test_function",
            "arguments": {"param": "value"}
        }
        
        message = Message(
            message_id="msg_1",
            role="assistant",
            content=[Message.text_message("Calling a tool", "assistant").content[0]],
            tools_calls=[tool_call_dict]
        )
        
        result = _convert_dict(message)
        
        expected = {
            "role": "assistant",
            "content": "Calling a tool",
            "tool_calls": [tool_call_dict]
        }
        assert result == expected

    def test_convert_messages_with_none_system_prompts(self):
        """Test convert_messages raises error when system_prompts is None."""
        with pytest.raises(ValueError, match="System prompts cannot be None"):
            convert_messages(None)  # type: ignore

    def test_convert_messages_basic(self):
        """Test convert_messages with only system prompts."""
        system_prompts = [
            {"role": "system", "content": "You are helpful"},
            {"role": "system", "content": "Be concise"}
        ]
        
        result = convert_messages(system_prompts)
        
        assert result == system_prompts

    def test_convert_messages_with_state_context_summary(self):
        """Test convert_messages includes state context summary."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]
        state = AgentState(
            context=[],
            context_summary="This is a summary of previous context"
        )
        
        result = convert_messages(system_prompts, state)
        
        expected = [
            {"role": "system", "content": "You are helpful"},
            {"role": "assistant", "content": "This is a summary of previous context"}
        ]
        assert result == expected

    def test_convert_messages_with_state_context_summary_empty(self):
        """Test convert_messages with empty context summary."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]
        state = AgentState(
            context=[],
            context_summary=""
        )

        result = convert_messages(system_prompts, state)

        # Empty context_summary should not be added (condition is if state.context_summary)
        expected = [
            {"role": "system", "content": "You are helpful"}
        ]
        assert result == expected

    def test_convert_messages_with_state_context_no_summary(self):
        """Test convert_messages with state context but no summary."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]
        messages = [
            Message.text_message("Hello", "user"),
            Message.text_message("Hi there", "assistant")
        ]
        state = AgentState(context=messages)
        
        result = convert_messages(system_prompts, state)
        
        expected = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        assert result == expected

    def test_convert_messages_with_state_context_and_summary(self):
        """Test convert_messages with both context summary and messages."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]
        messages = [
            Message.text_message("New message", "user"),
        ]
        state = AgentState(
            context=messages,
            context_summary="Previous conversation summary"
        )
        
        result = convert_messages(system_prompts, state)
        
        expected = [
            {"role": "system", "content": "You are helpful"},
            {"role": "assistant", "content": "Previous conversation summary"},
            {"role": "user", "content": "New message"}
        ]
        assert result == expected

    def test_convert_messages_with_extra_messages(self):
        """Test convert_messages with extra messages."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]
        extra_messages = [
            Message.text_message("Extra message 1", "user"),
            Message.text_message("Extra response", "assistant")
        ]
        
        result = convert_messages(system_prompts, extra_messages=extra_messages)
        
        expected = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Extra message 1"},
            {"role": "assistant", "content": "Extra response"}
        ]
        assert result == expected

    def test_convert_messages_complete_scenario(self):
        """Test convert_messages with all components: system, state, and extra messages."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]
        
        state_messages = [Message.text_message("State message", "user")]
        state = AgentState(
            context=state_messages,
            context_summary="Context summary"
        )
        
        extra_messages = [Message.text_message("Extra message", "user")]
        
        result = convert_messages(system_prompts, state, extra_messages)
        
        expected = [
            {"role": "system", "content": "You are helpful"},
            {"role": "assistant", "content": "Context summary"},
            {"role": "user", "content": "State message"},
            {"role": "user", "content": "Extra message"}
        ]
        assert result == expected

    def test_convert_messages_with_tool_messages_in_state(self):
        """Test convert_messages handles tool messages in state context."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]
        
        tool_result_block = ToolResultBlock(
            call_id="call_123",
            output="Tool executed successfully"
        )
        
        tool_message = Message(
            message_id="msg_1",
            role="tool",
            content=[tool_result_block]
        )
        
        state = AgentState(context=[tool_message])
        
        result = convert_messages(system_prompts, state)
        
        expected = [
            {"role": "system", "content": "You are helpful"},
            {"role": "tool", "content": "Tool executed successfully", "tool_call_id": "call_123"}
        ]
        assert result == expected

    def test_convert_messages_with_malformed_tool_message_in_state(self):
        """Test convert_messages skips crashes for malformed persisted tool messages."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]

        tool_message = Message.text_message("placeholder", "tool")
        tool_message.content = None # type: ignore

        state = AgentState(context=[tool_message])

        result = convert_messages(system_prompts, state)

        expected = [
            {"role": "system", "content": "You are helpful"},
            {"role": "tool", "content": "", "tool_call_id": ""},
        ]
        assert result == expected

    def test_convert_messages_with_assistant_tool_calls_in_state(self):
        """Test convert_messages handles assistant messages with tool calls in state context."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]
        
        tool_call_dict = {
            "id": "call_123",
            "name": "get_weather",
            "arguments": {"location": "New York"}
        }
        
        assistant_message = Message(
            message_id="msg_1",
            role="assistant",
            content=[Message.text_message("I'll check the weather", "assistant").content[0]],
            tools_calls=[tool_call_dict]
        )
        
        state = AgentState(context=[assistant_message])
        
        result = convert_messages(system_prompts, state)
        
        expected = [
            {"role": "system", "content": "You are helpful"},
            {
                "role": "assistant",
                "content": "I'll check the weather",
                "tool_calls": [tool_call_dict]
            }
        ]
        assert result == expected

    def test_convert_messages_empty_state(self):
        """Test convert_messages with empty state."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]
        state = AgentState(context=[])
        
        result = convert_messages(system_prompts, state)
        
        assert result == system_prompts

    @patch('tenxgraph.utils.converter.logger')
    def test_convert_messages_logs_message_count(self, mock_logger):
        """Test that convert_messages logs the number of converted messages."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]
        
        result = convert_messages(system_prompts)
        
        mock_logger.debug.assert_called_once_with("Number of Converted messages: %s", 1)
        assert len(result) == 1

    @patch('tenxgraph.utils.converter.logger')
    def test_convert_messages_logs_error_on_none_system_prompts(self, mock_logger):
        """Test that convert_messages logs error when system_prompts is None."""
        with pytest.raises(ValueError):
            convert_messages(None)  # type: ignore
        
        mock_logger.error.assert_called_once_with("System prompts are None")

    def test_convert_messages_handles_none_state(self):
        """Test convert_messages handles None state gracefully."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]
        
        result = convert_messages(system_prompts, state=None)
        
        assert result == system_prompts

    def test_convert_messages_handles_none_extra_messages(self):
        """Test convert_messages handles None extra_messages gracefully."""
        system_prompts = [{"role": "system", "content": "You are helpful"}]
        
        result = convert_messages(system_prompts, extra_messages=None)
        
        assert result == system_prompts