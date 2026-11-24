"""ToolRunner: one tool path for invoke and stream (engine merge step 2).

Covers the picks of step 2 (every call answered, duplicates removed), that invoke builds no
stream objects, that a cancel reaches every worker, and that a tool's own progress chunk
comes before its result chunk.
"""

import asyncio

import pytest

from tenxgraph.core.graph import ToolNode
from tenxgraph.core.graph.utils import tool_runner
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.core.state.stream_chunks import StreamEvent
from tests.graph.tool_guard_helpers import run_calls


def _assistant(*names: str) -> Message:
    msg = Message.text_message("calling", role="assistant")
    msg.tools_calls = [
        {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": "{}"}}
        for i, n in enumerate(names)
    ]
    return msg


def _outputs(messages: list[Message]) -> list[str]:
    return [str(b.output) for m in messages for b in m.content if isinstance(b, ToolResultBlock)]


class _Escape(BaseException):
    pass


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_base_exception_in_a_tool_still_answers_the_call(streaming):
    def ok() -> str:
        return "fine"

    def escapes() -> str:
        raise _Escape("boom")

    msg = _assistant("ok", "escapes")
    _, result = await run_calls(
        ToolNode([ok, escapes]), msg, AgentState(context=[msg]), {}, streaming=streaming
    )

    assert [m.content[-1].call_id for m in result.messages] == ["call_0", "call_1"]
    assert result.messages[1].content[-1].is_error is True


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_duplicate_messages_are_kept_once(streaming):
    shared = Message.tool_message(content=[ToolResultBlock(call_id="call_0", output="same")])

    def same() -> Message:
        return shared

    msg = _assistant("same", "same")
    _, result = await run_calls(
        ToolNode([same]), msg, AgentState(context=[msg]), {}, streaming=streaming
    )

    assert [m.message_id for m in result.messages] == [shared.message_id]


@pytest.mark.asyncio
async def test_invoke_builds_no_stream_objects(monkeypatch):
    built: list[str] = []
    real_chunk, real_emitter = tool_runner.StreamChunk, tool_runner.StreamEmitter

    def chunk(*a, **k):
        built.append("chunk")
        return real_chunk(*a, **k)

    def emitter(*a, **k):
        built.append("emitter")
        return real_emitter(*a, **k)

    monkeypatch.setattr(tool_runner, "StreamChunk", chunk)
    monkeypatch.setattr(tool_runner, "StreamEmitter", emitter)

    def ok() -> str:
        return "fine"

    msg = _assistant("ok", "ok", "missing")
    chunks, result = await run_calls(ToolNode([ok]), msg, AgentState(context=[msg]), {})

    assert built == []
    assert chunks == []
    assert len(result.messages) == 3

    await run_calls(ToolNode([ok]), msg, AgentState(context=[msg]), {}, streaming=True)
    assert "chunk" in built and "emitter" in built


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_cancel_cancels_every_worker(streaming):
    started: list[str] = []
    cancelled: list[str] = []

    async def slow(tool_call_id: str = "") -> str:
        started.append(tool_call_id)
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(tool_call_id)
            raise
        return "late"

    msg = _assistant("slow", "slow")
    run = asyncio.create_task(
        run_calls(ToolNode([slow]), msg, AgentState(context=[msg]), {}, streaming=streaming)
    )
    while len(started) < 2:
        await asyncio.sleep(0.01)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert sorted(cancelled) == ["call_0", "call_1"]


@pytest.mark.asyncio
async def test_progress_chunk_comes_before_the_result_chunk():
    def reports(emit=None) -> str:
        emit.progress("halfway")
        return "done"

    msg = _assistant("reports")
    chunks, result = await run_calls(
        ToolNode([reports]), msg, AgentState(context=[msg]), {"thread_id": "t"}, streaming=True
    )

    kinds = [
        c.data.get("status") if c.event != StreamEvent.MESSAGE else "tool_result" for c in chunks
    ]
    assert kinds[0] == "invoking_tool"
    assert kinds[-1] == "tool_result"
    assert len(kinds) == 3  # invoking, the tool's progress, the result
    assert _outputs(result.messages) == ["done"]
