"""A handoff answers every tool call of its assistant message before control moves on.

Providers reject a history where an assistant tool call has no tool result, so the target
agent must receive a valid history, on invoke and stream, also next to other tool calls.
"""

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.execution_state import ExecutionStatus
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.prebuilt.tools.handoff import create_handoff_tool
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END


def _calls(*names: str) -> list[dict]:
    return [
        {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": "{}"}}
        for i, n in enumerate(names)
    ]


def _unanswered(context: list[Message]) -> list[str]:
    """Ids of tool calls that no later tool message answers."""
    answered = {
        b.call_id
        for m in context
        if m.role == "tool"
        for b in m.content
        if isinstance(b, ToolResultBlock)
    }
    return [
        c["id"]
        for m in context
        if m.role == "assistant"
        for c in (m.tools_calls or [])
        if c["id"] not in answered
    ]


def _build(names: tuple[str, ...], seen: list[list[Message]], runs: list[str]):
    def lookup(x: int = 0) -> str:
        """Regular tool."""
        runs.append("lookup")
        return "found"

    def model(state: AgentState):
        msg = Message.text_message("calling", role="assistant")
        msg.tools_calls = _calls(*names)
        return msg

    def worker(state: AgentState):
        seen.append(list(state.context))
        return Message.text_message("worker done", role="assistant")

    graph = StateGraph()
    graph.add_node("MAIN", model)
    graph.add_node("TOOL", ToolNode([create_handoff_tool("worker"), lookup]))
    graph.add_node("worker", worker)
    graph.add_edge("MAIN", "TOOL")
    graph.add_edge("worker", END)
    graph.set_entry_point("MAIN")
    return graph


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [("transfer_to_worker",), ("lookup", "transfer_to_worker")])
@pytest.mark.parametrize("mode", ["invoke", "stream"])
async def test_handoff_answers_every_tool_call(names, mode):
    seen: list[list[Message]] = []
    runs: list[str] = []
    cp = InMemoryCheckpointer()
    app = _build(names, seen, runs).compile(checkpointer=cp)
    inp = {"messages": [Message.text_message("hi", role="user")]}
    config = {"thread_id": "t1"}
    if mode == "invoke":
        await app.ainvoke(inp, config=config)
    else:
        async for _ in app.astream(inp, config=config):
            pass

    assert len(seen) == 1
    assert _unanswered(seen[0]) == []
    handoff_result = next(
        b.output
        for m in seen[0]
        if m.role == "tool"
        for b in m.content
        if isinstance(b, ToolResultBlock) and b.call_id == f"call_{len(names) - 1}"
    )
    assert "Transferred to worker" in handoff_result
    assert runs == []  # a handoff turn does not run the sibling tools

    saved = await cp.aget_state(config)
    assert _unanswered(saved.context) == []
    # one answer per call, and the handoff is not replayed on a later read
    assert sum(1 for m in saved.context if m.role == "tool") == len(names)


@pytest.mark.asyncio
async def test_resume_after_answered_handoff_does_not_rerun_it():
    seen: list[list[Message]] = []
    runs: list[str] = []
    cp = InMemoryCheckpointer()
    app = _build(("transfer_to_worker",), seen, runs).compile(checkpointer=cp)
    config = {"thread_id": "t2"}
    await app.ainvoke({"messages": [Message.text_message("hi", role="user")]}, config=config)

    # Rewind to the tool node, as a crash right after the node was saved would leave it.
    state = await cp.aget_state(config)
    state.context = state.context[:3]  # user, assistant call, handoff answer
    state.execution_meta.current_node = "TOOL"
    state.execution_meta.status = ExecutionStatus.RUNNING
    await cp.aput_state(config, state)
    await app.ainvoke({"messages": []}, config=config)

    after = (await cp.aget_state(config)).context
    assert sum(1 for m in after if m.role == "tool") == 1
    assert _unanswered(after) == []
    assert len(seen) == 1  # the worker ran once, in the first run only
