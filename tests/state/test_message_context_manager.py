import pytest
from tenxgraph.core.state.message_context_manager import MessageContextManager
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import TextBlock, ToolCallBlock, ToolResultBlock

@pytest.fixture
def system_msg():
    return Message.text_message("System prompt", role="system")

@pytest.fixture
def user_msg():
    return Message.text_message("User message", role="user")

@pytest.fixture
def assistant_msg():
    return Message.text_message("AI response", role="assistant")

@pytest.fixture
def tool_call_msg():
    return Message(
        role="assistant",
        content=[ToolCallBlock(id="call1", name="search", args={})],
        tools_calls=[{"id": "call1", "name": "search", "args": {}}],
    )

@pytest.fixture
def tool_result_msg():
    return Message.tool_message([ToolResultBlock(call_id="call1", output="result")])

def make_context(system, user_count, tool_seq=False):
    msgs = [system]
    for i in range(user_count):
        msgs.append(Message.text_message(f"User {i+1}", role="user"))
        if tool_seq:
            # Tool call
            msgs.append(
                Message(
                    role="assistant",
                    content=[ToolCallBlock(id=f"call{i+1}", name="search", args={})],
                    tools_calls=[{"id": f"call{i+1}", "name": "search", "args": {}}],
                )
            )
            # Tool result
            msgs.append(
                Message.tool_message([
                    ToolResultBlock(call_id=f"call{i+1}", output=f"result{i+1}")
                ])
            )
            # Final AI response
            msgs.append(Message.text_message(f"Final AI {i+1}", role="assistant"))
    return msgs

def test_trim_no_tool_removal(system_msg):
    ctx = make_context(system_msg, 12)
    state = AgentState(context=ctx)
    mgr = MessageContextManager(max_messages=10, remove_tool_msgs=False)
    trimmed = mgr._trim(state.context)
    assert trimmed is not None
    assert trimmed[0].role == "system"
    assert sum(1 for m in trimmed if m.role == "user") == 10

def test_trim_with_tool_removal(system_msg):
    ctx = make_context(system_msg, 3, tool_seq=True)
    state = AgentState(context=ctx)
    mgr = MessageContextManager(max_messages=2, remove_tool_msgs=True)
    trimmed = mgr._trim(state.context)
    assert trimmed is not None
    # Only complete tool sequences should be removed
    assert all(m.role != "tool" for m in trimmed)
    assert all(
        not any(isinstance(b, ToolCallBlock) for b in m.content)
        for m in trimmed if m.role == "assistant"
    )
    assert sum(1 for m in trimmed if m.role == "user") == 2

def test_trim_context_method(system_msg):
    ctx = make_context(system_msg, 5)
    state = AgentState(context=ctx)
    mgr = MessageContextManager(max_messages=3)
    new_state = mgr.trim_context(state)
    assert sum(1 for m in new_state.context if m.role == "user") == 3

@pytest.mark.asyncio
async def test_async_trim_context(system_msg):
    ctx = make_context(system_msg, 5)
    state = AgentState(context=ctx)
    mgr = MessageContextManager(max_messages=2)
    new_state = await mgr.atrim_context(state)
    assert sum(1 for m in new_state.context if m.role == "user") == 2

def test_no_trim_needed(system_msg):
    ctx = make_context(system_msg, 2)
    state = AgentState(context=ctx)
    mgr = MessageContextManager(max_messages=5)
    trimmed = mgr._trim(state.context)
    assert trimmed is None

def test_empty_context():
    mgr = MessageContextManager()
    trimmed = mgr._trim([])
    assert trimmed is None
