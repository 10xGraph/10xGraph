import asyncio

import pytest

from tenxgraph.core.state.stream_chunks import StreamEvent
from tenxgraph.core.state.stream_emitter import StreamEmitter


@pytest.mark.asyncio
async def test_progress_emits_message_chunk_with_expected_payload():
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    emitter = StreamEmitter(
        tool_name="weather",
        tool_call_id="call-1",
        node_name="TOOL",
        thread_id="t-1",
        run_id="r-1",
        queue=queue,
        loop=loop,
    )

    emitter.progress("fetching", {"attempt": 2})
    await asyncio.sleep(0)
    chunk = queue.get_nowait()

    assert chunk.event == StreamEvent.MESSAGE
    assert chunk.thread_id == "t-1"
    assert chunk.run_id == "r-1"
    assert chunk.data["tool_name"] == "weather"
    assert chunk.data["tool_call_id"] == "call-1"
    assert chunk.data["node"] == "TOOL"
    assert chunk.data["status"] == "tool_progress"
    assert chunk.data["message"] == "fetching"
    assert chunk.data["attempt"] == 2


@pytest.mark.asyncio
async def test_error_emits_error_chunk():
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    emitter = StreamEmitter(
        tool_name="search",
        tool_call_id="call-2",
        node_name="TOOL",
        thread_id=None,
        run_id=None,
        queue=queue,
        loop=loop,
    )

    emitter.error("failed", {"code": "E42"})
    await asyncio.sleep(0)
    chunk = queue.get_nowait()

    assert chunk.event == StreamEvent.ERROR
    assert chunk.data["status"] == "tool_failed"
    assert chunk.data["message"] == "failed"
    assert chunk.data["code"] == "E42"


@pytest.mark.asyncio
async def test_message_and_update_emit_expected_event_types():
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    emitter = StreamEmitter(
        tool_name="math",
        tool_call_id="call-3",
        node_name="CALC",
        thread_id="thread",
        run_id="run",
        queue=queue,
        loop=loop,
    )

    emitter.message("step")
    emitter.update({"partial": 123})
    await asyncio.sleep(0)

    first = queue.get_nowait()
    second = queue.get_nowait()

    assert first.event == StreamEvent.MESSAGE
    assert first.data["status"] == "tool_message"
    assert first.data["message"] == "step"

    assert second.event == StreamEvent.UPDATES
    assert second.data["partial"] == 123
    assert second.data["tool_name"] == "math"
    assert second.data["node"] == "CALC"
