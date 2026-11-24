"""Shared scaffolding for the ToolDenied / ToolGuard / approval tests: real graph runs with a
scripted model node, a recording publisher and a fake MCP client."""

from __future__ import annotations

import json

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.graph.utils.node_runner import NodeRunner
from tenxgraph.core.graph.utils.tool_runner import ToolRunner
from tenxgraph.core.state import AgentState, Message, ToolCallBlock
from tenxgraph.core.state.message_block import ErrorBlock, ToolResultBlock
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.utils import CallbackManager, ResponseGranularity
from tenxgraph.utils.constants import END


FULL = ResponseGranularity.FULL
MODES = ["ainvoke", "astream"]


class RecordingPublisher(BasePublisher):
    def __init__(self):
        super().__init__({})
        self.events: list = []

    async def publish(self, event):
        self.events.append(event.model_copy(deep=True))

    async def close(self):
        pass

    def sync_close(self):
        pass


class FakeMCPTool:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} description"
        self.input_schema = {"type": "object", "properties": {}}
        self.meta = {}


class FakeMCPClient:
    """Enough of a fastmcp client for ToolNode: list_tools, call_tool, async context."""

    def __init__(self, *names):
        self._tools = [FakeMCPTool(n) for n in names]
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, args, *, meta=None):
        self.calls.append((name, args))

        class _Res:
            content = [{"type": "text", "text": f"{name} ran"}]

        return _Res()


def assistant_call(calls: list[tuple[str, str]], args: dict | None = None) -> Message:
    return Message(
        role="assistant",
        message_id="asst-1",
        content=[ToolCallBlock(id=cid, name=name, args=args or {}) for cid, name in calls],
        tools_calls=[
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args or {})},
            }
            for cid, name in calls
        ],
    )


def build(
    tool_node: ToolNode, calls, *, args=None, checkpointer=None, manager=None, publisher=None
):
    """MODEL asks for ``calls`` once, TOOL runs them, MODEL answers "done"."""

    async def model(st: AgentState):
        if st.context[-1].role == "tool":
            return Message.text_message("done", role="assistant")
        return assistant_call(calls, args)

    def route(st: AgentState) -> str:
        last = st.context[-1]
        return "TOOL" if last.role == "assistant" and last.tools_calls else END

    graph = StateGraph(publisher=publisher)
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", tool_node)
    graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    return graph.compile(checkpointer=checkpointer, callback_manager=manager or CallbackManager())


async def run(app, mode, config, data=None):
    """Run once; returns (state, chunks). chunks is [] for ainvoke."""
    data = data or {"messages": [Message.text_message("go")]}
    if mode == "ainvoke":
        result = await app.ainvoke(data, config, response_granularity=FULL)
        return result["state"], []
    chunks = [c async for c in app.astream(data, config, response_granularity=FULL)]
    state = await app._checkpointer.aget_state(config) if app._checkpointer else None
    return state, chunks


def tool_results(state: AgentState) -> dict[str, ToolResultBlock]:
    """call_id -> result block of every tool message in the state."""
    found = {}
    for msg in state.context or []:
        for block in msg.content or []:
            if isinstance(block, ToolResultBlock):
                found[block.call_id] = block
    return found


def tool_messages(state: AgentState) -> list[Message]:
    return [m for m in state.context or [] if m.role == "tool"]


def error_codes(msg: Message) -> list[str | None]:
    return [b.code for b in msg.content or [] if isinstance(b, ErrorBlock)]


async def run_calls(tool_node, message, state, config, *, streaming=False, name="tools"):
    """Run a message's tool calls through the ToolRunner: (stream chunks, ToolRunResult)."""
    items = [
        i
        async for i in ToolRunner(name, tool_node).call_tools(
            message, state, config, streaming=streaming
        )
    ]
    return items[:-1], items[-1]


async def run_node(tool_node, config, state, *, streaming, name="tools"):
    """Run one ToolNode step through the NodeRunner: (stream chunks, NodeResult)."""
    items = [
        i
        async for i in NodeRunner(name, tool_node).run(
            config, state, CallbackManager(), streaming=streaming
        )
    ]
    return items[:-1], items[-1]
