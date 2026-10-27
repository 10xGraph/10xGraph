"""Tool metric labels: the model-emitted name is used only for tools the node can run."""

from unittest.mock import AsyncMock

import pytest

from tenxgraph.core.graph import ToolNode
from tenxgraph.core.graph.utils import invoke_node_handler as mod
from tenxgraph.core.graph.utils.invoke_node_handler import InvokeNodeHandler
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.prebuilt.tools.handoff import create_handoff_tool


def echo() -> str:
    """Echo."""
    return "ok"


_real_timer = mod.metrics.timer


class _Recorder:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.timers: list[tuple[str, dict]] = []

    def counter(self, name):
        rec = self

        class _C:
            def inc(self, amount=1, attributes=None):
                rec.calls.append((name, attributes))

        return _C()

    def timer(self, name, attributes=None):
        self.timers.append((name, attributes))
        return _real_timer(name, attributes)


@pytest.fixture
def rec(monkeypatch):
    r = _Recorder()
    monkeypatch.setattr(mod.metrics, "counter", r.counter)
    monkeypatch.setattr(mod.metrics, "timer", r.timer)
    return r


async def _call(node: ToolNode, name: str, config: dict | None = None):
    handler = InvokeNodeHandler("TOOL", node)
    call = {"id": "c1", "type": "function", "function": {"name": name, "arguments": "{}"}}
    return await handler._handle_single_tool(call, AgentState(), config or {"thread_id": "t"})


def _tools_label(rec: _Recorder) -> set[str]:
    return {a["tool"] for n, a in rec.calls if n == "tenxgraph.tool.calls"}


@pytest.mark.asyncio
async def test_local_tool_keeps_its_name(rec):
    await _call(ToolNode([echo]), "echo")
    assert _tools_label(rec) == {"echo"}
    assert all(a["tool"] == "echo" for _, a in rec.timers)


@pytest.mark.asyncio
async def test_unknown_name_is_grouped_and_still_reports_not_found(rec):
    out = await _call(ToolNode([echo]), "tool_98765")
    assert _tools_label(rec) == {"unknown"}
    assert all(a["tool"] == "unknown" for _, a in rec.timers)
    assert isinstance(out, Message)
    text = " ".join(b.output for b in out.content if isinstance(b, ToolResultBlock))
    assert "not found" in text


@pytest.mark.asyncio
async def test_handoff_tool_keeps_its_name(rec):
    node = ToolNode([create_handoff_tool("admin_node")])
    await _call(node, "transfer_to_admin_node")
    assert _tools_label(rec) == {"transfer_to_admin_node"}


@pytest.mark.asyncio
async def test_mcp_tool_keeps_its_name(rec):
    node = ToolNode([echo])
    node.mcp_tools = ["remote_search"]
    node._mcp_execute = AsyncMock(return_value=Message.tool_message(content=[]))  # type: ignore
    await _call(node, "remote_search")
    assert _tools_label(rec) == {"remote_search"}


@pytest.mark.asyncio
async def test_configured_remote_tool_keeps_its_name(rec):
    node = ToolNode([echo])
    node.set_remote_tool([{"type": "function", "function": {"name": "pick_color"}}])
    await _call(node, "pick_color")
    assert _tools_label(rec) == {"pick_color"}


@pytest.mark.asyncio
async def test_per_run_remote_tool_keeps_its_name_only_for_that_run(rec):
    node = ToolNode([echo])
    cfg = {
        "thread_id": "t",
        "remote_tools": [{"name": "geo", "description": "Location.", "parameters": {}}],
    }
    await _call(node, "geo", cfg)
    assert _tools_label(rec) == {"geo"}
    rec.calls.clear()
    await _call(node, "geo")
    assert _tools_label(rec) == {"unknown"}
