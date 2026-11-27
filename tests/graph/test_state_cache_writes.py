"""How often a run writes the state cache and the durable store.

A node's state is written to the cache when the node starts (the running node and step, for
crash recovery and the stop endpoint) and by the per-step save after it. The node-end cache
write is skipped when the per-step save follows, since that save writes the same state.
"""

import json

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message, ToolCallBlock
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END


class _CountingCheckpointer(InMemoryCheckpointer):
    """Counts state-cache writes and durable checkpoint writes."""

    def __init__(self) -> None:
        super().__init__()
        self.cache_writes = 0
        self.checkpoint_writes = 0

    async def aput_state_cache(self, config, state):
        self.cache_writes += 1
        return await super().aput_state_cache(config, state)

    async def aput_checkpoint(self, config, state, messages):
        self.checkpoint_writes += 1
        return await super().aput_checkpoint(config, state, messages)


def _model_tool_model(checkpointer):
    """MAIN calls the tool, TOOL runs it, MAIN answers: three nodes."""

    def add(a: int, b: int) -> int:
        return a + b

    async def main(state: AgentState):
        if state.context[-1].role == "tool":
            return Message.text_message("done", role="assistant")
        args = {"a": 1, "b": 2}
        return Message(
            role="assistant",
            content=[ToolCallBlock(id="c1", name="add", args=args)],
            tools_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "add", "arguments": json.dumps(args)},
                }
            ],
        )

    def route(state: AgentState) -> str:
        last = state.context[-1]
        return "TOOL" if last.role == "assistant" and last.tools_calls else END

    graph = StateGraph()
    graph.add_node("MAIN", main)
    graph.add_node("TOOL", ToolNode([add]))
    graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")
    return graph.compile(checkpointer=checkpointer)


async def _run(durable: bool) -> _CountingCheckpointer:
    checkpointer = _CountingCheckpointer()
    app = _model_tool_model(checkpointer)
    config = {"thread_id": "t1", "durable_checkpoint_every_step": durable}
    await app.ainvoke({"messages": [Message.text_message("add 1 and 2")]}, config)
    state = await checkpointer.aget_state(config)
    assert state.context[-1].text() == "done"
    assert await checkpointer.aget_state_cache(config) == state
    return checkpointer


@pytest.mark.asyncio
async def test_durable_steps_write_the_cache_twice_per_node():
    checkpointer = await _run(durable=True)
    # Four steps (__start__, MAIN, TOOL, MAIN). Per step: node start + per-step save, then
    # the final save. The last step has no per-step save: the final save follows it. Before
    # the node-end write was dropped this was 13.
    assert checkpointer.cache_writes == 8
    assert checkpointer.checkpoint_writes == 4


@pytest.mark.asyncio
async def test_without_durable_steps_the_node_end_write_stays():
    checkpointer = await _run(durable=False)
    # Per step: node start, node end, step advance (4 steps, the last has no advance write:
    # the final save follows it), then the final save.
    assert checkpointer.cache_writes == 12
    assert checkpointer.checkpoint_writes == 1
