"""Realtime tool calls run like the turn-based ones: tool_timeout, tool metrics, clean cancel.

Each call runs in a task of its own, so the receive loop keeps yielding provider events
while a tool runs, and ending the session cancels the tools still running.
"""

import asyncio
import gc
import warnings

import pytest

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.realtime.base import AudioDeltaEvent, ToolCallEvent, ToolResultEvent
from tenxgraph.core.realtime.live_agent import LiveAgent
from tenxgraph.core.realtime.queue import LiveInputQueue
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.utils import metrics
from tests.graph.tool_guard_helpers import run_calls
from tests.realtime.test_live_agent import MODEL, FakeRealtimeClient


class _OpenClient(FakeRealtimeClient):
    """Yields its events, then keeps the stream open like a live socket."""

    async def receive(self):
        for event in self.events:
            yield event
        await asyncio.Event().wait()


class _Recorder:
    def __init__(self):
        self.counts: list[tuple[str, dict]] = []
        self.timers: list[tuple[str, dict]] = []

    def install(self, monkeypatch):
        real_timer = metrics.timer
        rec = self

        class _Counter:
            def __init__(self, name):
                self.name = name

            def inc(self, amount=1, attributes=None):
                rec.counts.append((self.name, attributes))

        def timer(name, attributes=None):
            rec.timers.append((name, attributes))
            return real_timer(name, attributes)

        monkeypatch.setattr(metrics, "counter", _Counter)
        monkeypatch.setattr(metrics, "timer", timer)

    def names(self) -> list[str]:
        return [name for name, _ in self.counts]


def _closed_queue():
    queue = LiveInputQueue()
    queue.close()
    return queue


async def _run(agent, config, queue=None):
    return [e async for e in agent.arun(queue or _closed_queue(), config)]


def _agent(client, *tools):
    return LiveAgent(MODEL, tool_node=ToolNode(list(tools)), realtime_client_factory=lambda: client)


async def slow_tool() -> str:
    """Never finishes in time."""
    await asyncio.sleep(10)
    return "late"


def get_weather(city: str) -> str:
    """Weather."""
    return f"sunny in {city}"


@pytest.mark.asyncio
async def test_tool_timeout_cuts_the_call_off_with_the_invoke_path_result():
    client = FakeRealtimeClient([ToolCallEvent(id="c1", name="slow_tool", args={})])
    events = await _run(_agent(client, slow_tool), {"thread_id": "t1", "tool_timeout": 0.05})

    result = next(e for e in events if isinstance(e, ToolResultEvent)).result
    # The text the invoke path gives the model for the same timed-out call.
    msg = Message.text_message("calling", role="assistant")
    msg.tools_calls = [
        {"id": "c1", "type": "function", "function": {"name": "slow_tool", "arguments": "{}"}}
    ]
    _, ran = await run_calls(
        ToolNode([slow_tool]), msg, AgentState(context=[msg]), {"tool_timeout": 0.05}, name="LIVE"
    )
    [invoked] = ran.messages
    expected = " ".join(b.output for b in invoked.content if isinstance(b, ToolResultBlock))

    assert result == {"result": expected}
    assert "exceeded its timeout of 0.05s" in expected
    assert client.tool_responses == [("c1", "slow_tool", result)]


@pytest.mark.asyncio
async def test_no_tool_timeout_by_default():
    async def short_tool() -> str:
        """Takes a moment."""
        await asyncio.sleep(0.05)
        return "done"

    client = FakeRealtimeClient([ToolCallEvent(id="c1", name="short_tool", args={})])
    await _run(_agent(client, short_tool), {"thread_id": "t1"})
    assert client.tool_responses == [("c1", "short_tool", {"result": "done"})]


@pytest.mark.asyncio
async def test_tool_metrics_are_recorded(monkeypatch):
    rec = _Recorder()
    rec.install(monkeypatch)
    client = FakeRealtimeClient([ToolCallEvent(id="c1", name="get_weather", args={"city": "Rome"})])

    await _run(_agent(client, get_weather), {"thread_id": "t1"})

    attrs = {"node": "LIVE", "tool": "get_weather"}
    assert ("tenxgraph.tool.calls", attrs) in rec.counts
    assert ("tenxgraph.tool.duration", attrs) in rec.timers
    assert "tenxgraph.tool.timeouts" not in rec.names()


@pytest.mark.asyncio
async def test_timeout_metric_is_recorded(monkeypatch):
    rec = _Recorder()
    rec.install(monkeypatch)
    client = FakeRealtimeClient([ToolCallEvent(id="c1", name="slow_tool", args={})])

    await _run(_agent(client, slow_tool), {"thread_id": "t1", "tool_timeout": 0.05})

    assert ("tenxgraph.tool.timeouts", {"node": "LIVE", "tool": "slow_tool"}) in rec.counts


@pytest.mark.asyncio
async def test_receiving_continues_while_a_tool_runs():
    gate = asyncio.Event()

    async def wait_for_gate() -> str:
        """Finishes only once the caller has seen the audio after the call."""
        await gate.wait()
        return "opened"

    client = FakeRealtimeClient(
        [ToolCallEvent(id="c1", name="wait_for_gate", args={}), AudioDeltaEvent(data=b"\x01")]
    )
    types = []
    async for event in _agent(client, wait_for_gate).arun(_closed_queue(), {"thread_id": "t1"}):
        types.append(event.type)
        if event.type == "audio_delta":
            gate.set()

    assert types == ["tool_call", "audio_delta", "tool_result"]
    assert client.tool_responses == [("c1", "wait_for_gate", {"result": "opened"})]


@pytest.mark.asyncio
async def test_session_cancel_cancels_the_running_tool():
    started = asyncio.Event()
    outcome: list[str] = []

    async def hang() -> str:
        """Runs until cancelled."""
        started.set()
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise
        return "never"

    client = _OpenClient([ToolCallEvent(id="c1", name="hang", args={})])
    agent = _agent(client, hang)
    queue = LiveInputQueue()  # left open: only the cancel ends the session

    async def consume():
        async for _ in agent.arun(queue, {"thread_id": "t1"}):
            pass

    gc.collect()  # flush coroutines leaked by earlier tests before recording
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), timeout=2)
        session.cancel()
        with pytest.raises(asyncio.CancelledError):
            await session
        gc.collect()

    assert outcome == ["cancelled"]
    assert agent._tool_tasks == set()
    assert client.closed is True
    assert client.tool_responses == []
    leftover = [t for t in asyncio.all_tasks() if "_run_tool" in repr(t.get_coro())]
    assert leftover == []
    unawaited = [str(w.message) for w in caught if "never awaited" in str(w.message)]
    assert not [m for m in unawaited if "_run_tool" in m or "hang'" in m]


def boom() -> str:
    """Always fails."""
    raise RuntimeError("boom")


async def short_tool() -> str:
    """Takes a moment."""
    await asyncio.sleep(0.05)
    return "ok"


@pytest.mark.asyncio
async def test_tool_result_event_reports_status_and_duration():
    client = FakeRealtimeClient(
        [
            ToolCallEvent(id="c1", name="short_tool", args={}),
            ToolCallEvent(id="c2", name="boom", args={}),
            ToolCallEvent(id="c3", name="slow_tool", args={}),
            ToolCallEvent(id="c4", name="ghost", args={}),
        ]
    )
    events = await _run(
        _agent(client, short_tool, boom, slow_tool), {"thread_id": "t1", "tool_timeout": 0.2}
    )
    by_id = {e.id: e for e in events if isinstance(e, ToolResultEvent)}

    assert by_id["c1"].status == "success"
    assert by_id["c1"].duration_ms >= 40.0
    assert by_id["c2"].status == "error"
    assert by_id["c3"].status == "error"
    assert by_id["c3"].duration_ms >= 150.0
    assert by_id["c4"].status == "error"
