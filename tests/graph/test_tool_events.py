"""Tool events in both run modes, and the event sequence recorder (engine merge, step 1).

Tool events name the graph node in ``node_name`` and the tool, as the model called it, in
``data["function_name"]``. The recorder compares each scenario's
``(event, event_type, node_name)`` sequence with ``snapshots/engine_events.json``; tool
events inside one run of them are compared as a multiset, since parallel order is not fixed.
Set ``UPDATE_EVENT_SNAPSHOT=1`` to rewrite the file after an intended change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tenxgraph.core.exceptions import ToolDenied
from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.publisher.events import Event, EventType
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import CallbackManager, InvocationType, interrupt
from tenxgraph.utils.constants import END

from .tool_guard_helpers import MODES, FakeMCPClient, RecordingPublisher, build, run


SNAPSHOT = Path(__file__).parent / "snapshots" / "engine_events.json"
PICK_COLOR = {
    "name": "pick_color",
    "description": "Ask the user to pick a color.",
    "parameters": {"type": "object", "properties": {}},
}


class InlineRecorder(RecordingPublisher):
    """Gets every event in publish order, on the caller's stack."""

    inline = True

    def publish_inline(self, event):
        self.events.append(event.model_copy(deep=True))


def alpha() -> str:
    return "a"


def beta() -> str:
    return "b"


def ask() -> str:
    return str(interrupt("ok?"))


def _context_recorder() -> tuple[CallbackManager, list[tuple[str, str]]]:
    """A manager that records ``(context.node_name, context.function_name)`` per tool call."""
    seen: list[tuple[str, str]] = []

    async def before(context, input_data):
        seen.append((context.node_name, context.function_name))
        if context.function_name == "refund":
            raise ToolDenied("Not for you.")
        return input_data

    manager = CallbackManager()
    for kind in (InvocationType.TOOL, InvocationType.MCP):
        manager.register_before_invoke(kind, before)
    return manager, seen


def _tool_app(scenario: str, publisher, manager):
    def refund() -> str:
        return "refunded"

    nodes = {
        "local_parallel": (ToolNode([alpha, beta]), [("c1", "alpha"), ("c2", "beta")]),
        "mcp": (
            ToolNode([], client=FakeMCPClient("lookup"), mcp_prefix="srv_"),
            [("c1", "srv_lookup")],
        ),
        "unknown": (ToolNode([alpha]), [("c1", "ghost")]),
        "denied": (ToolNode([refund]), [("c1", "refund")]),
        "remote": (ToolNode([alpha]), [("c1", "pick_color")]),
        "tool_interrupt": (ToolNode([ask]), [("c1", "ask")]),
    }
    tool_node, calls = nodes[scenario]
    app = build(
        tool_node,
        calls,
        checkpointer=InMemoryCheckpointer(),
        manager=manager,
        publisher=publisher,
    )
    return app, tool_node


def _node_app(scenario: str, publisher):
    holder: dict = {}

    async def first(state: AgentState, config: dict):
        if scenario == "node_error":
            raise RuntimeError("boom")
        if scenario == "stop":
            await holder["app"].astop({"thread_id": config["thread_id"]})
        return Message.text_message("first", role="assistant")

    def second(state: AgentState):
        return Message.text_message("second", role="assistant")

    graph = StateGraph(publisher=publisher)
    graph.add_node("FIRST", first)
    graph.add_node("SECOND", second)
    graph.add_edge("FIRST", "SECOND")
    graph.add_edge("SECOND", END)
    graph.set_entry_point("FIRST")
    holder["app"] = graph.compile(checkpointer=InMemoryCheckpointer())
    return holder["app"]


TOOL_SCENARIOS = ["local_parallel", "mcp", "unknown", "denied", "remote", "tool_interrupt"]
SCENARIOS = ["function_nodes", "node_error", "stop", *TOOL_SCENARIOS]


async def _record(scenario: str, mode: str):
    publisher = InlineRecorder()
    manager, seen = _context_recorder()
    config: dict = {"thread_id": f"events-{scenario}-{mode}"}
    if scenario in TOOL_SCENARIOS:
        app, tool_node = _tool_app(scenario, publisher, manager)
        await tool_node.all_tools()  # what an Agent does first; it finds the MCP tools
        if scenario == "remote":
            config["remote_tools"] = [PICK_COLOR]
    else:
        app = _node_app(scenario, publisher)
    try:
        await run(app, mode, config)
    except Exception:  # node_error: the sequence up to the failure is the point
        if scenario != "node_error":
            raise
    return publisher.events, seen


def _value(item) -> str:
    return str(getattr(item, "value", item))


def _sequence(events) -> list:
    """``event:type:node`` per event; each run of tool events becomes one sorted list."""
    out: list = []
    for event in events:
        key = f"{_value(event.event)}:{_value(event.event_type)}:{event.node_name}"
        if event.event == Event.TOOL_EXECUTION:
            if out and isinstance(out[-1], list):
                out[-1] = sorted([*out[-1], key])
            else:
                out.append([key])
        else:
            out.append(key)
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("scenario", "tools"),
    [
        ("local_parallel", ["alpha", "beta"]),
        ("mcp", ["srv_lookup"]),
        ("unknown", ["ghost"]),
        ("denied", ["refund"]),
        ("remote", ["pick_color"]),
    ],
)
async def test_tool_events_name_the_graph_node_and_the_called_tool(scenario, tools, mode):
    events, seen = await _record(scenario, mode)
    tool_events = [e for e in events if e.event == Event.TOOL_EXECUTION]

    assert tool_events
    assert {e.node_name for e in tool_events} == {"TOOL"}
    assert {e.metadata["graph_node"] for e in tool_events} == {"TOOL"}
    assert sorted({e.data["function_name"] for e in tool_events}) == tools
    # Callbacks see the graph node too (remote and unknown tools run no callback).
    assert all(node == "TOOL" for node, _ in seen)
    if scenario in ("local_parallel", "mcp", "denied"):
        assert sorted(name for _, name in seen) == tools


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_unknown_tool_sends_start_then_error(mode):
    events, _ = await _record("unknown", mode)
    types = [e.event_type for e in events if e.event == Event.TOOL_EXECUTION]
    assert types == [EventType.START, EventType.ERROR]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_event_sequence_matches_the_snapshot(scenario, mode):
    """Today's sequence per scenario and mode; step 4 of the engine merge makes them equal."""
    recorded = _sequence((await _record(scenario, mode))[0])
    snapshot = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}
    if os.environ.get("UPDATE_EVENT_SNAPSHOT"):
        snapshot.setdefault(scenario, {})[mode] = recorded
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
    assert recorded == snapshot[scenario][mode]
