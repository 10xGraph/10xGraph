"""P-2: event payloads are built only when someone listens.

With no publisher and no lifecycle hooks, a run never dumps its state or messages for events.
With a publisher bound, the same run does, and the publisher receives plain dicts.
"""

import sys
from unittest.mock import patch

import pytest
from injectq import InjectQ

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ToolCallBlock
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.utils import END


def lookup(city: str) -> str:
    return f"sunny in {city}"


def _graph(publisher=None):
    def model(state: AgentState):
        if state.context and state.context[-1].role == "tool":
            return Message.text_message("done", role="assistant")
        return Message(
            role="assistant",
            content=[ToolCallBlock(id="c1", name="lookup", args={"city": "Paris"})],
            tools_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"city": "Paris"}'},
                }
            ],
        )

    def route(state: AgentState) -> str:
        last = state.context[-1]
        return "TOOL" if last.role == "assistant" and last.tools_calls else END

    graph = StateGraph(container=InjectQ(), publisher=publisher)
    graph.add_node("MAIN", model)
    graph.add_node("TOOL", ToolNode([lookup]))
    graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")
    return graph.compile()


class _Collect(BasePublisher):
    def __init__(self):
        super().__init__({})
        self.events = []

    async def publish(self, event):
        self.events.append(event)

    async def close(self):
        pass

    def sync_close(self):
        pass


# Modules that build or send events. A dump called from one of them is an event payload.
_EVENT_CODE = (
    "tenxgraph/runtime/publisher/",
    "tenxgraph/core/graph/tool_node/",
    "tenxgraph/core/graph/utils/engine.py",
    "tenxgraph/core/graph/utils/stream_handler.py",
    "tenxgraph/core/graph/utils/node_runner.py",
    "tenxgraph/core/graph/utils/handler_utils.py",
)


def _count_dumps():
    """Count AgentState / Message dumps called from event code. Other dumps (the storage
    checks before a save, the tool ledger) are not event payloads and are not counted."""
    calls = {"n": 0}
    originals = {cls: cls.model_dump for cls in (AgentState, Message)}

    def wrap(original):
        def counted(self, *args, **kwargs):
            caller = sys._getframe(1).f_code.co_filename.replace("\\", "/")
            if any(part in caller for part in _EVENT_CODE):
                calls["n"] += 1
            return original(self, *args, **kwargs)

        return counted

    patches = [patch.object(cls, "model_dump", wrap(fn)) for cls, fn in originals.items()]
    return calls, patches


async def _run(app, mode: str):
    inp = {"messages": [Message.text_message("weather?")]}
    if mode == "invoke":
        return await app.ainvoke(inp, config={"thread_id": "t1"})
    return [c async for c in app.astream(inp, config={"thread_id": "t1"})]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invoke", "stream"])
async def test_no_listener_builds_no_event_payloads(mode):
    app = _graph()
    calls, patches = _count_dumps()
    for p in patches:
        p.start()
    try:
        await _run(app, mode)
    finally:
        for p in patches:
            p.stop()
        await app.aclose()
    assert calls["n"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invoke", "stream"])
async def test_listener_gets_plain_payloads_and_one_start_end_per_tool(mode):
    pub = _Collect()
    app = _graph(publisher=pub)
    calls, patches = _count_dumps()
    for p in patches:
        p.start()
    try:
        await _run(app, mode)
        await app.aclose()
    finally:
        for p in patches:
            p.stop()
    assert calls["n"] > 0
    # An event publisher gets the state on the run-end event only, as a plain dict.
    with_state = [e for e in pub.events if "state" in e.data]
    assert [e.metadata["event_name"] for e in with_state] == ["run.completed"]
    assert isinstance(with_state[0].data["state"], dict)

    tool = [
        (e.event_type, e.node_name, e.data["function_name"])
        for e in pub.events
        if e.event == "tool_execution"
    ]
    assert tool == [("start", "TOOL", "lookup"), ("end", "TOOL", "lookup")]
    # Run events reach the publisher with no hook registered.
    names = [e.metadata.get("event_name") for e in pub.events]
    assert names[0] == "run.started" and names[-1] == "run.completed"
