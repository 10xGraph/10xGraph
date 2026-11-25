"""The durable per-step save names the node that runs next, not the one that just finished.

A process killed after that save, whose state cache is lost too (Redis restart, TTL), is
recovered from the durable copy alone. Recovery then resumes at the saved node, so a save that
still named the finished node would run it a second time (its tool calls twice).
"""

from __future__ import annotations

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END


class _DurableLog(InMemoryCheckpointer):
    """Keeps a copy of every durable state save."""

    def __init__(self) -> None:
        super().__init__()
        self.durable: list[AgentState] = []

    async def aput_state(self, config, state):
        self.durable.append(state.model_copy(deep=True))
        return await super().aput_state(config, state)


def _a_then_b(checkpointer, ran: list[str]):
    async def a(state: AgentState):
        ran.append("A")
        return Message.text_message("a done", role="assistant")

    async def b(state: AgentState):
        ran.append("B")
        return Message.text_message("b done", role="assistant")

    graph = StateGraph()
    graph.add_node("A", a)
    graph.add_node("B", b)
    graph.add_edge("A", "B")
    graph.add_edge("B", END)
    graph.set_entry_point("A")
    return graph.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_step_save_after_a_names_b():
    checkpointer = _DurableLog()
    app = _a_then_b(checkpointer, [])
    config = {"thread_id": "t-step-save", "user_id": "u1"}
    await app.ainvoke({"messages": [Message.text_message("go")]}, config)

    after_a = next(
        s
        for s in checkpointer.durable
        if s.execution_meta.status == ExecutionStatus.RUNNING and s.context[-1].text() == "a done"
    )
    assert after_a.execution_meta.current_node == "B"
    # A completed run still names its last node (the END step save does not move it).
    final = await checkpointer.aget_state(config)
    assert final.execution_meta.status == ExecutionStatus.COMPLETED
    assert final.execution_meta.current_node == "B"
    await app.aclose()


@pytest.mark.asyncio
async def test_crash_after_a_with_lost_cache_resumes_at_b():
    checkpointer = _DurableLog()
    ran: list[str] = []
    app = _a_then_b(checkpointer, ran)
    config = {"thread_id": "t-crash-lost-cache", "user_id": "u1"}
    await app.ainvoke({"messages": [Message.text_message("go")]}, config)
    after_a = next(
        s
        for s in checkpointer.durable
        if s.execution_meta.status == ExecutionStatus.RUNNING and s.context[-1].text() == "a done"
    )
    await app.aclose()

    # The process died right after A's step save and the cache is gone: only the durable copy
    # of that save is left, with no live run marker.
    crashed = InMemoryCheckpointer()
    await crashed.aput_checkpoint(config, after_a, list(after_a.context))
    await crashed.aclear_state_cache(config)
    assert await crashed.aget_state_cache(config) is None
    ran.clear()
    app = _a_then_b(crashed, ran)

    await app.ainvoke({}, config)

    assert ran == ["B"]
    state = await crashed.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert [m.text() for m in state.context] == ["go", "a done", "b done"]
    await app.aclose()
