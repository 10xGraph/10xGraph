"""Optional result cap and duplicate-name warning on ToolNode (G-50)."""

import logging

import pytest

from tenxgraph.core.graph import ToolNode
from tenxgraph.core.state import AgentState
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.utils import CallbackManager


def big() -> str:
    """Return a lot of text."""
    return "x" * 1000


@pytest.mark.asyncio
async def test_result_cap_cuts_long_text():
    node = ToolNode([big], max_result_chars=100)
    msg = await node.invoke(
        name="big",
        args={},
        tool_call_id="c1",
        config={},
        state=AgentState(),
        callback_manager=CallbackManager(),
    )
    out = next(b for b in msg.content if isinstance(b, ToolResultBlock)).output
    assert len(out) < 200
    assert "output cut" in out


@pytest.mark.asyncio
async def test_no_cap_by_default():
    node = ToolNode([big])
    msg = await node.invoke(
        name="big",
        args={},
        tool_call_id="c1",
        config={},
        state=AgentState(),
        callback_manager=CallbackManager(),
    )
    assert len(next(b for b in msg.content if isinstance(b, ToolResultBlock)).output) == 1000


def test_replacing_a_tool_name_logs_a_warning(caplog):
    def one() -> str:
        return "1"

    def other() -> str:
        return "2"

    other.__name__ = "one"
    node = ToolNode([one])
    with caplog.at_level(logging.WARNING, logger="tenxgraph.graph.tool_node"):
        node.add_tool(other)
        node.add_tool(other)  # same callable again is not a replacement
    assert sum("already registered" in r.getMessage() for r in caplog.records) == 1


def test_generator_of_tools_is_accepted():
    node = ToolNode(f for f in [big])
    assert "big" in node._funcs
