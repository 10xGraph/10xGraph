"""Run correlation in logs and metrics on both paths (G-25)."""

import logging

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message, ToolCallBlock
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import metrics
from tenxgraph.utils.constants import END
from tenxgraph.utils.logging import CorrelationFilter, get_log_context


MODES = ["ainvoke", "astream"]


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.addFilter(CorrelationFilter())
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


async def run(app, mode, config):
    data = {"messages": [Message.text_message("hi")]}
    if mode == "ainvoke":
        return await app.ainvoke(data, config)
    return [c async for c in app.astream(data, config)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_logs_carry_run_ids_and_the_context_is_reset_after(mode):
    probe = logging.getLogger("tenxgraph.test_probe")

    async def work(state: AgentState):
        probe.warning("inside the node")
        return Message.text_message("ok", role="assistant")

    graph = StateGraph()
    graph.add_node("WORK", work)
    graph.add_edge("WORK", END)
    graph.set_entry_point("WORK")
    app = graph.compile(checkpointer=InMemoryCheckpointer())

    capture = Capture()
    probe.addHandler(capture)
    try:
        await run(app, mode, {"thread_id": f"t-log-{mode}", "user_id": "u-log"})
    finally:
        probe.removeHandler(capture)

    [record] = [r for r in capture.records if r.getMessage() == "inside the node"]
    assert record.thread_id == f"t-log-{mode}"
    assert record.user_id == "u-log"
    assert record.run_id
    # The ids do not outlive the run.
    assert get_log_context() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_node_and_tool_metrics_are_recorded(mode, monkeypatch):
    seen: list[str] = []
    real_counter = metrics.counter
    real_timer = metrics.timer

    def counter(name, *args, **kwargs):
        seen.append(name)
        return real_counter(name, *args, **kwargs)

    def timer(name, *args, **kwargs):
        seen.append(name)
        return real_timer(name, *args, **kwargs)

    monkeypatch.setattr(metrics, "counter", counter)
    monkeypatch.setattr(metrics, "timer", timer)

    async def work() -> str:
        return "done"

    async def model(state: AgentState):
        if state.context[-1].role == "tool":
            return Message.text_message("fin", role="assistant")
        return Message(
            role="assistant",
            content=[ToolCallBlock(id="c1", name="work", args={})],
            tools_calls=[
                {"id": "c1", "type": "function", "function": {"name": "work", "arguments": "{}"}}
            ],
        )

    def route(state: AgentState) -> str:
        last = state.context[-1]
        return "TOOL" if last.role == "assistant" and last.tools_calls else END

    graph = StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode([work]))
    graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    app = graph.compile(checkpointer=InMemoryCheckpointer())

    await run(app, mode, {"thread_id": f"t-metrics-{mode}"})

    for name in (
        "tenxgraph.node.executions",
        "tenxgraph.node.duration",
        "tenxgraph.tool.calls",
        "tenxgraph.tool.duration",
    ):
        assert name in seen, name
