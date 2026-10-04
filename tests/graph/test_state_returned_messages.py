"""Messages a node appends to ``state.context`` are reported, whatever the node returns.

A node that appends to ``state.context`` and returns the same state object used to lose those
messages: they were saved in the state, but never streamed and never returned as messages,
because the "before" and "after" states were one object.
"""

import pytest

from agentflow.core.graph import StateGraph
from agentflow.core.state import AgentState, Message, StreamEvent
from agentflow.utils import ResponseGranularity
from agentflow.utils.command import Command
from agentflow.utils.constants import END


class NoteState(AgentState):
    note: str = ""


async def append_and_return_state(state: NoteState):
    state.note = "done"
    state.context.append(Message.text_message("from state", role="assistant"))
    return state


async def return_command_with_state(state: NoteState):
    state.context.append(Message.text_message("from command", role="assistant"))
    return Command(state=state, goto=END)


def _graph(node) -> StateGraph:
    graph = StateGraph(NoteState())
    graph.add_node("MAIN", node)
    graph.add_edge("MAIN", END)
    graph.set_entry_point("MAIN")
    return graph.compile()


async def _streamed_texts(app, thread_id: str) -> list[str]:
    texts = []
    async for chunk in app.astream(
        {"messages": [Message.text_message("hi")]},
        config={"thread_id": thread_id},
        response_granularity=ResponseGranularity.FULL,
    ):
        if chunk.event == StreamEvent.MESSAGE and chunk.message and chunk.message.role != "user":
            texts.append(chunk.message.text())
    return texts


@pytest.mark.asyncio
async def test_stream_includes_messages_appended_to_the_returned_state():
    texts = await _streamed_texts(_graph(append_and_return_state), "t-stream")
    assert texts == ["from state"]


@pytest.mark.asyncio
async def test_invoke_returns_messages_appended_to_the_returned_state():
    result = await _graph(append_and_return_state).ainvoke(
        {"messages": [Message.text_message("hi")]},
        config={"thread_id": "t-invoke"},
        response_granularity=ResponseGranularity.FULL,
    )
    assert [m.text() for m in result["messages"] if m.role == "assistant"] == ["from state"]
    assert result["state"].note == "done"


@pytest.mark.asyncio
async def test_command_state_streams_only_new_messages():
    texts = await _streamed_texts(_graph(return_command_with_state), "t-command")
    # The user's message was already streamed; only the node's own message is new.
    assert texts == ["from command"]
