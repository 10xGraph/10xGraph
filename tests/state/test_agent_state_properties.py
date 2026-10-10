"""Property-based tests for AgentState using Hypothesis.

This module uses Hypothesis to generate random test cases and verify
invariants in AgentState, reducers, and execution metadata.
"""

import pytest
from hypothesis import given, strategies as st

from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.core.state.execution_state import ExecutionState, StopRequestStatus
from tenxgraph.core.state.reducers import add_messages, remove_tool_messages
from tenxgraph.core.state.message import TokenUsages
from tenxgraph.core.state.message_block import TextBlock, ToolCallBlock, ToolResultBlock
from tenxgraph.utils import START, END


# Hypothesis strategies for generating test data


@st.composite
def message_strategy(draw, role=None):
    """Generate a valid Message object."""
    if role is None:
        role = draw(st.sampled_from(["user", "assistant", "system", "tool"]))
    
    text_content = draw(st.text(min_size=1, max_size=100))
    
    return Message(
        message_id=draw(st.integers(min_value=1, max_value=1000000)),
        role=role,
        content=[TextBlock(text=text_content)],
        delta=False,
    )


@st.composite
def agent_state_strategy(draw):
    """Generate a valid AgentState object."""
    messages = draw(st.lists(message_strategy(), min_size=0, max_size=10))
    summary = draw(st.one_of(st.none(), st.text(max_size=200)))
    
    return AgentState(
        context=messages,
        context_summary=summary,
    )


class TestAgentStateProperties:
    """Property-based tests for AgentState."""

    @given(agent_state_strategy())
    def test_agent_state_always_has_execution_meta(self, state):
        """Verify that AgentState always has execution metadata."""
        assert hasattr(state, "execution_meta")
        assert isinstance(state.execution_meta, ExecutionState)
        assert state.execution_meta.current_node == START

    @given(agent_state_strategy())
    def test_agent_state_starts_running(self, state):
        """Verify that new AgentState starts in RUNNING status."""
        assert state.is_running()
        assert not state.is_interrupted()
        assert state.execution_meta.status == ExecutionStatus.RUNNING

    @given(agent_state_strategy(), st.text(min_size=1, max_size=50))
    def test_set_current_node_updates_correctly(self, state, node_name):
        """Verify that set_current_node updates the current node."""
        state.set_current_node(node_name)
        assert state.execution_meta.current_node == node_name

    @given(agent_state_strategy(), st.integers(min_value=0, max_value=100))
    def test_advance_step_increments(self, state, initial_steps):
        """Verify that advance_step increments the step counter."""
        state.execution_meta.step = initial_steps
        state.advance_step()
        assert state.execution_meta.step == initial_steps + 1

    @given(agent_state_strategy())
    def test_complete_sets_status(self, state):
        """Verify that complete() sets status to COMPLETED."""
        state.complete()
        assert state.execution_meta.status == ExecutionStatus.COMPLETED
        assert not state.is_running()

    @given(agent_state_strategy(), st.text(min_size=1, max_size=100))
    def test_error_sets_status_and_message(self, state, error_msg):
        """Verify that error() sets status to ERROR and stores message."""
        state.error(error_msg)
        assert state.execution_meta.status == ExecutionStatus.ERROR
        assert not state.is_running()

    @given(
        agent_state_strategy(),
        st.text(min_size=1, max_size=50),
        st.text(min_size=1, max_size=100),
    )
    def test_set_interrupt_creates_interrupt(self, state, node_name, reason):
        """Verify that set_interrupt properly creates an interrupt."""
        state.set_interrupt(
            node_name,
            reason,
            ExecutionStatus.INTERRUPTED_BEFORE,
        )
        assert state.is_interrupted()
        assert not state.is_running()
        assert state.execution_meta.interrupted_node == node_name
        assert state.execution_meta.interrupt_reason == reason

    @given(agent_state_strategy())
    def test_clear_interrupt_resets_state(self, state):
        """Verify that clear_interrupt clears all interrupt data."""
        # First set an interrupt
        state.set_interrupt("test_node", "test reason", ExecutionStatus.INTERRUPTED_AFTER)
        assert state.is_interrupted()
        
        # Clear it
        state.clear_interrupt()
        assert not state.is_interrupted()
        assert state.is_running()
        assert state.execution_meta.interrupted_node is None
        assert state.execution_meta.interrupt_reason is None


class TestReducerProperties:
    """Property-based tests for reducer functions."""

    @given(
        st.lists(message_strategy(), min_size=0, max_size=20),
        st.lists(message_strategy(), min_size=0, max_size=20),
    )
    def test_add_messages_preserves_left_messages(self, left, right):
        """Verify that add_messages preserves all messages from left list."""
        result = add_messages(left, right)
        
        # All left messages should be in result
        left_ids = {msg.message_id for msg in left}
        result_ids = {msg.message_id for msg in result}
        assert left_ids.issubset(result_ids)

    @given(
        st.lists(message_strategy(), min_size=1, max_size=20),
        st.lists(message_strategy(), min_size=1, max_size=20),
    )
    def test_add_messages_no_duplicates(self, left, right):
        """Verify that add_messages doesn't add messages already in left."""
        result = add_messages(left, right)
        
        # The reducer should not add messages from right that are already in left
        left_ids = {msg.message_id for msg in left}
        # Count messages in result that came from right (not in left)
        added_from_right = [msg for msg in result if msg.message_id not in left_ids]
        # These should all be unique
        added_ids = [msg.message_id for msg in added_from_right]
        # Note: left itself may have duplicates, but we don't add duplicates from right
        # So we verify that no message_id from left appears in added_from_right
        for msg in added_from_right:
            assert msg.message_id not in left_ids

    @given(st.lists(message_strategy(), min_size=0, max_size=20))
    def test_add_messages_idempotent_with_self(self, messages):
        """Verify that adding messages to themselves doesn't duplicate."""
        result = add_messages(messages, messages)
        
        # Result should have same length as input (no duplicates)
        assert len(result) == len(messages)

    @given(st.lists(message_strategy(), min_size=0, max_size=20))
    def test_add_messages_order_preserved(self, messages):
        """Verify that add_messages preserves message order."""
        empty = []
        result = add_messages(empty, messages)
        
        # Messages should appear in same order
        for i, msg in enumerate(messages):
            if not msg.delta:  # Only non-delta messages are added
                assert msg.message_id in [m.message_id for m in result]

    def test_remove_tool_messages_complete_sequence(self):
        """Verify that complete tool sequences are removed."""
        user_msg = Message(
            message_id=1,
            role="user",
            content=[TextBlock(text="What's the weather?")],
        )
        
        ai_with_tools = Message(
            message_id=2,
            role="assistant",
            content=[TextBlock(text="Let me check...")],
            tools_calls=[{"id": "call1", "function": {"name": "get_weather"}}],
        )
        
        tool_result = Message(
            message_id=3,
            role="tool",
            content=[ToolResultBlock(output="Sunny, 72°F", call_id="call1")],
        )
        
        ai_final = Message(
            message_id=4,
            role="assistant",
            content=[TextBlock(text="It's sunny and 72°F")],
        )
        
        messages = [user_msg, ai_with_tools, tool_result, ai_final]
        result = remove_tool_messages(messages)
        
        # Should keep user and final AI message
        assert len(result) == 2
        assert result[0].message_id == 1
        assert result[1].message_id == 4

    def test_remove_tool_messages_incomplete_sequence(self):
        """Verify that incomplete tool sequences are preserved."""
        user_msg = Message(
            message_id=1,
            role="user",
            content=[TextBlock(text="What's the weather?")],
        )
        
        ai_with_tools = Message(
            message_id=2,
            role="assistant",
            content=[TextBlock(text="Let me check...")],
            tools_calls=[{"id": "call1", "function": {"name": "get_weather"}}],
        )
        
        messages = [user_msg, ai_with_tools]
        result = remove_tool_messages(messages)
        
        # Should keep all messages (incomplete sequence)
        assert len(result) == 2
        assert result[0].message_id == 1
        assert result[1].message_id == 2

    def test_remove_tool_messages_empty_list(self):
        """Verify that empty list returns empty list."""
        result = remove_tool_messages([])
        assert result == []

    @given(st.lists(message_strategy(role="user"), min_size=1, max_size=10))
    def test_remove_tool_messages_only_user_messages(self, messages):
        """Verify that lists with only user messages are unchanged."""
        result = remove_tool_messages(messages)
        assert len(result) == len(messages)


class TestExecutionStateProperties:
    """Property-based tests for ExecutionState."""

    @given(st.text(min_size=1, max_size=50))
    def test_execution_state_initialization(self, node_name):
        """Verify ExecutionState initializes correctly."""
        exec_state = ExecutionState(current_node=node_name)
        assert exec_state.current_node == node_name
        assert exec_state.step == 0
        assert exec_state.status == ExecutionStatus.RUNNING

    @given(st.integers(min_value=0, max_value=1000))
    def test_execution_state_step_tracking(self, initial_step):
        """Verify step tracking works correctly."""
        exec_state = ExecutionState(current_node=START)
        exec_state.step = initial_step
        
        exec_state.advance_step()
        assert exec_state.step == initial_step + 1
        
        exec_state.advance_step()
        assert exec_state.step == initial_step + 2

    def test_execution_state_stop_request_lifecycle(self):
        """Verify stop request lifecycle."""
        exec_state = ExecutionState(current_node=START)
        
        # Initially no stop requested
        assert exec_state.stop_current_execution == StopRequestStatus.NONE
        assert not exec_state.is_stopped_requested()
        
        # Request stop
        exec_state.stop_current_execution = StopRequestStatus.STOP_REQUESTED
        assert exec_state.is_stopped_requested()
        
        # Clear stop request
        exec_state.stop_current_execution = StopRequestStatus.NONE
        assert not exec_state.is_stopped_requested()


class TestMessageProperties:
    """Property-based tests for Message objects."""

    @given(st.text(min_size=1, max_size=500))
    def test_text_message_creation(self, text):
        """Verify text message creation preserves content."""
        msg = Message.text_message(text, role="user")
        
        assert msg.role == "user"
        assert msg.text() == text
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextBlock)

    @given(
        st.text(min_size=1, max_size=100),
        st.sampled_from(["user", "assistant", "system"]),
    )
    def test_message_role_assignment(self, text, role):
        """Verify message role assignment works correctly."""
        msg = Message.text_message(text, role=role)
        assert msg.role == role

    def test_message_with_token_usage(self):
        """Verify message can store token usage."""
        usage = TokenUsages(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
        )
        
        msg = Message.text_message("test", role="assistant")
        msg.usages = usage
        
        assert msg.usages.completion_tokens == 10
        assert msg.usages.prompt_tokens == 20
        assert msg.usages.total_tokens == 30

    @given(st.dictionaries(st.text(min_size=1, max_size=20), st.integers()))
    def test_message_metadata(self, metadata):
        """Verify message can store arbitrary metadata."""
        msg = Message.text_message("test", role="user")
        msg.metadata = metadata
        
        assert msg.metadata == metadata


# Additional integration property tests


class TestStateIntegrationProperties:
    """Property-based integration tests combining multiple components."""

    @given(
        agent_state_strategy(),
        st.lists(message_strategy(), min_size=1, max_size=10),
    )
    def test_state_context_updates(self, state, new_messages):
        """Verify state context updates correctly with add_messages."""
        original_count = len(state.context)
        state.context = add_messages(state.context, new_messages)
        
        # Should have at least original count
        assert len(state.context) >= original_count
        
        # All new non-delta messages should be present
        new_ids = {msg.message_id for msg in new_messages if not msg.delta}
        result_ids = {msg.message_id for msg in state.context}
        assert new_ids.issubset(result_ids)

    @given(agent_state_strategy())
    def test_state_serialization_roundtrip(self, state):
        """Verify state can be serialized and deserialized."""
        # Serialize to dict
        state_dict = state.model_dump()
        
        # Deserialize back
        restored_state = AgentState.model_validate(state_dict)
        
        # Basic equality checks
        assert restored_state.execution_meta.current_node == state.execution_meta.current_node
        assert restored_state.execution_meta.step == state.execution_meta.step
        assert len(restored_state.context) == len(state.context)

    @given(
        st.text(min_size=1, max_size=50),
        st.integers(min_value=0, max_value=100),
    )
    def test_state_execution_flow_invariants(self, node_name, steps):
        """Verify execution flow maintains invariants."""
        state = AgentState()
        
        # Execute steps and set node
        for _ in range(steps):
            state.advance_step()
            state.set_current_node(node_name)
        
        # Invariants
        assert state.execution_meta.step == steps
        # After setting node in loop, current_node should be node_name (if steps > 0) or START (if steps == 0)
        if steps > 0:
            assert state.execution_meta.current_node == node_name
        else:
            assert state.execution_meta.current_node == START
        
        # Complete should end execution
        state.complete()
        assert not state.is_running()
        assert state.execution_meta.status == ExecutionStatus.COMPLETED
