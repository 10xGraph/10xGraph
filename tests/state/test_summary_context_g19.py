"""G-19: summary manager keeps tool pairs together, counts results, bounds the summary."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import ImageBlock, MediaRef, ToolResultBlock
from tenxgraph.core.state.summary_context_manager import SummaryContextManager, _estimate_tokens


def _tool_result(call_id: str, text: str) -> Message:
    return Message(
        role="tool",
        content=[ToolResultBlock(call_id=call_id, output=text)],
        message_id=f"t-{call_id}",
    )


def _assistant_call(call_id: str) -> Message:
    m = Message.text_message("calling", role="assistant")
    m.tools_calls = [
        {"id": call_id, "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]
    return m


def test_kept_context_never_starts_with_a_tool_result():
    mgr = SummaryContextManager("gpt-4o-mini", keep_recent=2)
    msgs = [
        Message.text_message("q", role="user"),
        _assistant_call("c1"),
        _tool_result("c1", "r"),
        Message.text_message("done", role="assistant"),
    ]
    summarise, keep = mgr._split_context(msgs)
    assert keep[0].role != "tool"
    assert [m.role for m in summarise + keep] == [m.role for m in msgs]


def test_token_estimate_counts_tool_results():
    big = _tool_result("c1", "x" * 4000)
    assert _estimate_tokens([big]) >= 1000


def test_token_estimate_counts_images():
    img = Message(
        role="user",
        content=[ImageBlock(media=MediaRef(kind="url", url="http://x/y.png"))],
        message_id="i",
    )
    assert _estimate_tokens([img]) > 1


@pytest.mark.asyncio
async def test_cumulative_summary_is_compacted_when_it_grows_too_large():
    mgr = SummaryContextManager("gpt-4o-mini", max_messages=2, keep_recent=1, max_summary_tokens=10)
    state = AgentState()
    state.context_summary = "old " * 500
    state.context = [Message.text_message(f"m{i}", role="user") for i in range(5)]
    with patch(
        "tenxgraph.core.state.summary_context_manager.call_llm",
        AsyncMock(return_value=("short", 0, 0, 0)),
    ):
        await mgr.atrim_context(state)
    assert state.context_summary == "short"


def test_sync_trim_in_running_loop_raises_instead_of_asyncio_run():
    import asyncio

    mgr = SummaryContextManager("gpt-4o-mini")

    async def go():
        with pytest.raises(RuntimeError, match="atrim_context"):
            mgr.trim_context(AgentState())

    asyncio.run(go())
