"""LiveAgent session isolation, cancel safety and persistence-failure handling.

Covers G-10 (concurrent sessions), G-62 (setup/teardown cancel), G-61 (thread metadata on a
failed read) and the non-timeout parts of G-60 (tool ledger, no reconnect on a storage error,
close sentinel never dropped).
"""

import asyncio

import pytest

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.realtime.base import (
    InputTranscriptEvent,
    SessionUpdateEvent,
    ToolCallEvent,
    TurnCompleteEvent,
)
from tenxgraph.core.realtime.live_agent import LiveAgent
from tenxgraph.core.realtime.queue import LiveInputQueue
from tenxgraph.prebuilt.agent import AudioAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.thread_info import ThreadInfo
from tests.realtime.test_live_agent import MODEL, FakeRealtimeClient, _drain, _factory


class HeldClient(FakeRealtimeClient):
    """Yields its scripted events, then stays open until closed (a live provider socket)."""

    def __init__(self, events=None):
        super().__init__(events)
        self.connected = asyncio.Event()
        self.receive_cancelled = False

    async def connect(self, config, resume_handle=None):
        await super().connect(config, resume_handle)
        self.connected.set()

    async def receive(self):
        for event in self.events:
            yield event
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.receive_cancelled = True
            raise


async def _wait_for(predicate, timeout=2.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class TestConcurrentSessions:
    @pytest.mark.asyncio
    async def test_two_sessions_stay_separate(self):
        client_a = HeldClient(
            [
                InputTranscriptEvent(text="from A", finished=True),
                SessionUpdateEvent(resumption_handle="HA"),
            ]
        )
        client_b = HeldClient(
            [
                InputTranscriptEvent(text="from B", finished=True),
                SessionUpdateEvent(resumption_handle="HB"),
            ]
        )
        cp = InMemoryCheckpointer()
        app = AudioAgent(MODEL, realtime_client_factory=_factory(client_a, client_b)).compile(
            checkpointer=cp
        )
        cfg_a = {"thread_id": "ta", "user_id": "ua"}
        cfg_b = {"thread_id": "tb", "user_id": "ub"}
        qa, qb = LiveInputQueue(), LiveInputQueue()

        async def consume(queue, cfg):
            return [e async for e in app.arealtime(queue, cfg)]

        task_a = asyncio.create_task(consume(qa, cfg_a))
        await client_a.connected.wait()
        task_b = asyncio.create_task(consume(qb, cfg_b))
        await client_b.connected.wait()

        qa.send_audio(b"AAAA")
        qb.send_audio(b"BBBB")
        await _wait_for(lambda: client_a.sent_audio and client_b.sent_audio)
        await asyncio.sleep(0.05)

        # A disconnects; B must stay open and keep receiving its own audio.
        qa.close()
        await asyncio.wait_for(task_a, 2)
        assert client_a.closed is True
        assert client_b.closed is False
        qb.send_audio(b"B2")
        await _wait_for(lambda: len(client_b.sent_audio) == 2)
        qb.close()
        await asyncio.wait_for(task_b, 2)

        assert [a for a, _ in client_a.sent_audio] == [b"AAAA"]
        assert [a for a, _ in client_b.sent_audio] == [b"BBBB", b"B2"]
        assert client_b.closed is True
        msgs_a = await cp.alist_messages(cfg_a)
        msgs_b = await cp.alist_messages(cfg_b)
        assert [m.content[0].text for m in msgs_a] == ["from A"]
        assert [m.content[0].text for m in msgs_b] == ["from B"]
        assert (await cp.aget_thread(cfg_a)).metadata["resumption_handle"] == "HA"
        assert (await cp.aget_thread(cfg_b)).metadata["resumption_handle"] == "HB"

    @pytest.mark.asyncio
    async def test_same_instance_refuses_a_second_concurrent_arun(self):
        client = HeldClient()
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))
        q = LiveInputQueue()
        task = asyncio.create_task(_drain(agent, q, {"thread_id": "t"}))
        await client.connected.wait()
        with pytest.raises(RuntimeError, match="new_session"):
            await _drain(agent, LiveInputQueue(), {"thread_id": "t2"})
        q.close()
        await asyncio.wait_for(task, 2)

    def test_new_session_has_fresh_state_and_shared_config(self):
        agent = LiveAgent(MODEL, realtime_client_factory=_factory())
        agent._resume_handle = "H"
        agent._input_transcript_buf = "x"
        session = agent.new_session()
        assert session is not agent
        assert session._send_lock is not agent._send_lock
        assert session._resume_handle is None
        assert session._input_transcript_buf == ""
        assert session.realtime_config is agent.realtime_config


class TestCancelSafety:
    @pytest.mark.asyncio
    async def test_failed_connect_closes_the_provider_session(self):
        class FailingClient(FakeRealtimeClient):
            async def connect(self, config, resume_handle=None):
                raise ConnectionError("boom")

        client = FailingClient()
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))
        with pytest.raises(ConnectionError):
            await _drain(agent, LiveInputQueue(), {"thread_id": "t"})
        assert client.closed is True
        assert agent._session_active is False

    @pytest.mark.asyncio
    async def test_cancel_during_setup_closes_session_and_reraises(self):
        started = asyncio.Event()

        class SlowClient(FakeRealtimeClient):
            async def connect(self, config, resume_handle=None):
                started.set()
                await asyncio.Event().wait()

        client = SlowClient()
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))
        task = asyncio.create_task(_drain(agent, LiveInputQueue(), {"thread_id": "t"}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.closed is True

    @pytest.mark.asyncio
    async def test_cancel_mid_session_stops_receive_and_pump_and_reraises(self):
        client = HeldClient()
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))
        task = asyncio.create_task(_drain(agent, LiveInputQueue(), {"thread_id": "t"}))
        await client.connected.wait()
        await asyncio.sleep(0.05)  # receive is now pending
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.receive_cancelled is True
        assert client.closed is True
        assert task.cancelled()


class _ReadFailsCheckpointer(InMemoryCheckpointer):
    async def aget_thread(self, config):
        raise ConnectionError("transient")


class _PutMessagesFailsCheckpointer(InMemoryCheckpointer):
    async def aput_messages(self, config, messages, metadata=None):
        raise ConnectionError("db down")


class TestPersistenceFailures:
    @pytest.mark.asyncio
    async def test_failed_thread_read_does_not_overwrite_thread(self):
        cp = _ReadFailsCheckpointer()
        config = {"thread_id": "t", "user_id": "u"}
        await InMemoryCheckpointer.aput_thread(
            cp,
            config,
            ThreadInfo(thread_id="t", thread_name="My chat", user_id="u", metadata={"k": "v"}),
        )
        client = FakeRealtimeClient([SessionUpdateEvent(resumption_handle="H1")])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))

        await _drain(agent, LiveInputQueue_closed(), config, checkpointer=cp)

        thread = await InMemoryCheckpointer.aget_thread(cp, config)
        assert thread.thread_name == "My chat"
        assert thread.metadata == {"k": "v"}

    @pytest.mark.asyncio
    async def test_handle_save_keeps_existing_thread_name(self):
        cp = InMemoryCheckpointer()
        config = {"thread_id": "t", "user_id": "u"}
        await cp.aput_thread(
            config, ThreadInfo(thread_id="t", thread_name="My chat", user_id="u", metadata={})
        )
        client = FakeRealtimeClient([SessionUpdateEvent(resumption_handle="H1")])
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))
        await _drain(agent, LiveInputQueue_closed(), config, checkpointer=cp)
        thread = await cp.aget_thread(config)
        assert thread.thread_name == "My chat"
        assert thread.metadata["resumption_handle"] == "H1"

    @pytest.mark.asyncio
    async def test_storage_error_does_not_reconnect_the_provider(self):
        first = FakeRealtimeClient(
            [InputTranscriptEvent(text="hi", finished=True), TurnCompleteEvent()]
        )
        second = FakeRealtimeClient()
        agent = LiveAgent(MODEL, realtime_client_factory=_factory(first, second))
        agent._reconnect_base_delay = 0

        events = await _drain(
            agent,
            LiveInputQueue_closed(),
            {"thread_id": "t"},
            checkpointer=_PutMessagesFailsCheckpointer(),
        )

        assert second.connected_with == []  # never reconnected
        assert [e.type for e in events] == ["input_transcript", "turn_complete"]


def LiveInputQueue_closed():
    q = LiveInputQueue()
    q.close()
    return q


class TestToolLedger:
    @pytest.mark.asyncio
    async def test_replayed_call_id_runs_the_tool_once(self):
        calls = []

        def charge(amount: int) -> str:
            calls.append(amount)
            return "ok"

        client = FakeRealtimeClient(
            [
                ToolCallEvent(id="c1", name="charge", args={"amount": 5}),
                ToolCallEvent(id="c1", name="charge", args={"amount": 5}),
            ]
        )
        agent = LiveAgent(
            MODEL, tool_node=ToolNode([charge]), realtime_client_factory=_factory(client)
        )
        await _drain(
            agent,
            LiveInputQueue_closed(),
            {"thread_id": "t", "user_id": "u"},
            checkpointer=InMemoryCheckpointer(),
        )
        assert calls == [5]
        assert [r[2] for r in client.tool_responses] == [{"result": "ok"}, {"result": "ok"}]

    @pytest.mark.asyncio
    async def test_new_session_same_thread_replays_recorded_call(self):
        """A crash-restart session on the same thread must not rerun a re-sent call."""
        calls = []

        def charge(amount: int) -> str:
            calls.append(amount)
            return "ok"

        cp = InMemoryCheckpointer()
        for _ in range(2):  # two separate sessions (new_session ids), one shared thread
            client = FakeRealtimeClient([ToolCallEvent(id="c1", name="charge", args={"amount": 5})])
            agent = LiveAgent(
                MODEL, tool_node=ToolNode([charge]), realtime_client_factory=_factory(client)
            )
            await _drain(
                agent,
                LiveInputQueue_closed(),
                {"thread_id": "t", "user_id": "u"},
                checkpointer=cp,
            )
            assert [r[2] for r in client.tool_responses] == [{"result": "ok"}]
        assert calls == [5]

    @pytest.mark.asyncio
    async def test_reused_call_id_with_different_args_still_runs(self):
        calls = []

        def charge(amount: int) -> str:
            calls.append(amount)
            return "ok"

        cp = InMemoryCheckpointer()
        for amount in (5, 7):
            client = FakeRealtimeClient(
                [ToolCallEvent(id="c1", name="charge", args={"amount": amount})]
            )
            agent = LiveAgent(
                MODEL, tool_node=ToolNode([charge]), realtime_client_factory=_factory(client)
            )
            await _drain(
                agent, LiveInputQueue_closed(), {"thread_id": "t", "user_id": "u"}, checkpointer=cp
            )
        assert calls == [5, 7]

    @pytest.mark.asyncio
    async def test_non_object_args_are_refused(self):
        def noop() -> str:
            return "ran"

        client = FakeRealtimeClient([ToolCallEvent(id="c1", name="noop", args={})])
        client.events[0].args = "not-a-dict"
        agent = LiveAgent(
            MODEL, tool_node=ToolNode([noop]), realtime_client_factory=_factory(client)
        )
        await _drain(agent, LiveInputQueue_closed(), {"thread_id": "t"})
        assert "error" in client.tool_responses[0][2]


class TestCloseSentinel:
    def test_close_survives_a_full_queue(self):
        q = LiveInputQueue(maxsize=2)
        q.send_audio(b"1")
        q.send_audio(b"2")
        q.close()
        kinds = []
        while True:
            try:
                kinds.append(q.get_nowait().kind)
            except asyncio.QueueEmpty:
                break
        assert kinds[-1] == "close"
