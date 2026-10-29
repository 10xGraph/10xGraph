"""Model-controlled tool names must not forge log lines (G-72)."""

import logging

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils.constants import END
from tenxgraph.utils.log_safety import log_safe


def test_log_safe_escapes_control_chars():
    assert log_safe("a\r\nb\tc\x00") == "a\\r\\nb\\tc\\x00"
    assert "\n" not in log_safe("x\u2028y")


def test_log_safe_truncates():
    assert log_safe("a" * 500, limit=10) == "a" * 10 + "..."


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["ainvoke", "astream"])
async def test_tool_name_with_newline_does_not_split_log_line(caplog, method):
    def real_tool() -> str:
        """A tool."""
        return "ok"

    assistant = Message.text_message("calling", role="assistant")
    assistant.tools_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "real_tool\nFORGED LOG LINE", "arguments": "{}"},
        }
    ]

    def model(state: AgentState):
        return assistant.model_copy(deep=True, update={"message_id": "assistant_1"})

    graph = StateGraph()
    graph.add_node("MAIN", model)
    graph.add_node("TOOL", ToolNode([real_tool]))
    graph.add_edge("MAIN", "TOOL")
    graph.add_edge("TOOL", END)
    graph.set_entry_point("MAIN")
    compiled = graph.compile()

    with caplog.at_level(logging.DEBUG):
        if method == "ainvoke":
            await compiled.ainvoke({"messages": [Message.text_message("hi")]})
        else:
            async for _ in compiled.astream({"messages": [Message.text_message("hi")]}):
                pass

    relevant = [r.getMessage() for r in caplog.records if "FORGED LOG LINE" in r.getMessage()]
    assert relevant, "expected the tool name to be logged"
    assert all("\n" not in m for m in relevant)
