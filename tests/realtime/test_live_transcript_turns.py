"""Realtime transcripts and state are saved when a turn ends without a "finished" marker.

A live Gemini session (gemini-3.8-live, 2026-10-30) saved nothing: the thread had no
messages and no state. Transcripts were saved only on a transcription chunk marked
``finished``, which Gemini Live does not send; the turn just ends with ``turn_complete``,
so the text stayed in a buffer and was dropped at session end. The fake provider here
sends what the live session sent: partial chunks, then ``turn_complete``, no marker.
"""

import pytest

from tenxgraph.core.realtime.base import (
    InputTranscriptEvent,
    InterruptedEvent,
    OutputTranscriptEvent,
    TurnCompleteEvent,
)
from tenxgraph.core.realtime.live_agent import LiveAgent
from tenxgraph.core.realtime.queue import LiveInputQueue
from tenxgraph.core.state import AgentState, Message
from tenxgraph.prebuilt.agent import AudioAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tests.realtime.test_live_agent import FakeRealtimeClient, _closed_queue, _drain, _factory


MODEL = "gemini-3.8-live"
CONFIG = {"thread_id": "t-live", "user_id": "u1"}


def user(text: str) -> InputTranscriptEvent:
    return InputTranscriptEvent(text=text, finished=False)


def model(text: str) -> OutputTranscriptEvent:
    return OutputTranscriptEvent(text=text, finished=False)


async def run_session(events: list, state: AgentState | None = None):
    client = FakeRealtimeClient(events)
    agent = LiveAgent(MODEL, realtime_client_factory=_factory(client))
    cp = InMemoryCheckpointer()
    state = state or AgentState()
    out = await _drain(agent, _closed_queue(), dict(CONFIG), state=state, checkpointer=cp)
    return out, cp, state


def texts(messages: list[Message]) -> list[tuple[str, str]]:
    return [(m.role, m.text()) for m in messages]


@pytest.mark.asyncio
async def test_turn_without_finished_marker_is_saved():
    out, cp, _ = await run_session(
        [user("What is "), user("the weather?"), model("It is "), model("sunny."), TurnCompleteEvent()]
    )

    saved = await cp.alist_messages(CONFIG)
    assert texts(saved) == [("user", "What is the weather?"), ("assistant", "It is sunny.")]
    # The client gets one complete transcript per side before the turn_complete.
    finished = [(e.type, e.text) for e in out if getattr(e, "finished", False)]
    assert finished == [
        ("input_transcript", "What is the weather?"),
        ("output_transcript", "It is sunny."),
    ]
    assert out[-1].type == "turn_complete"


@pytest.mark.asyncio
async def test_state_checkpoint_holds_the_transcripts():
    _, cp, _ = await run_session([user("hi"), model("hello"), TurnCompleteEvent()])

    state = await cp.aget_state(CONFIG)
    assert state is not None
    assert texts(state.context) == [("user", "hi"), ("assistant", "hello")]


@pytest.mark.asyncio
async def test_two_turns_are_two_pairs_not_one_long_message():
    _, cp, _ = await run_session(
        [
            user("hi"),
            model("hello"),
            TurnCompleteEvent(),
            user("weather?"),
            model("sunny"),
            TurnCompleteEvent(),
        ]
    )

    assert texts(await cp.alist_messages(CONFIG)) == [
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "weather?"),
        ("assistant", "sunny"),
    ]


@pytest.mark.asyncio
async def test_finished_marker_then_turn_complete_saves_once():
    _, cp, _ = await run_session(
        [
            InputTranscriptEvent(text="hi", finished=True),
            OutputTranscriptEvent(text="hello", finished=True),
            TurnCompleteEvent(),
        ]
    )

    assert texts(await cp.alist_messages(CONFIG)) == [("user", "hi"), ("assistant", "hello")]


@pytest.mark.asyncio
async def test_barge_in_keeps_the_users_words_and_drops_the_cut_off_reply():
    _, cp, _ = await run_session(
        [
            user("tell me about Khulna"),
            model("Khulna is a city in"),
            InterruptedEvent(),
            user("just the weather"),
            model("It is sunny."),
            TurnCompleteEvent(),
        ]
    )

    assert texts(await cp.alist_messages(CONFIG)) == [
        ("user", "tell me about Khulna"),
        ("user", "just the weather"),
        ("assistant", "It is sunny."),
    ]


@pytest.mark.asyncio
async def test_session_end_mid_turn_saves_what_was_said():
    # The client went away before turn_complete: nothing closed the buffers.
    _, cp, _ = await run_session([user("are you"), user(" there?"), model("Yes, I am")])

    assert texts(await cp.alist_messages(CONFIG)) == [
        ("user", "are you there?"),
        ("assistant", "Yes, I am"),
    ]
    state = await cp.aget_state(CONFIG)
    assert texts(state.context) == [("user", "are you there?"), ("assistant", "Yes, I am")]


# ---------------------------------------------------------------------------
# Through CompiledGraph.arealtime: the session's start state
# ---------------------------------------------------------------------------


async def run_graph(app, config: dict) -> None:
    q = LiveInputQueue()
    q.close()
    async for _ in app.arealtime(q, config):
        pass


@pytest.mark.asyncio
async def test_session_continues_the_threads_saved_state():
    """A second session must add to the thread's state, not overwrite it."""
    cp = InMemoryCheckpointer()
    first = FakeRealtimeClient([user("hi"), model("hello"), TurnCompleteEvent()])
    second = FakeRealtimeClient([user("weather?"), model("sunny"), TurnCompleteEvent()])
    app = AudioAgent(MODEL, realtime_client_factory=_factory(first, second)).compile(
        checkpointer=cp
    )
    config = {"thread_id": "t-two-sessions", "user_id": "u1"}

    await run_graph(app, dict(config))
    await run_graph(app, dict(config))

    state = await cp.aget_state(config)
    assert texts(state.context) == [
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "weather?"),
        ("assistant", "sunny"),
    ]


@pytest.mark.asyncio
async def test_sessions_do_not_share_the_graph_state_object():
    """Sessions on different threads never see each other's transcripts."""
    cp = InMemoryCheckpointer()
    a = FakeRealtimeClient([user("I am Alice"), TurnCompleteEvent()])
    b = FakeRealtimeClient([user("I am Bob"), TurnCompleteEvent()])
    app = AudioAgent(MODEL, realtime_client_factory=_factory(a, b)).compile(checkpointer=cp)

    await run_graph(app, {"thread_id": "t-alice", "user_id": "alice"})
    await run_graph(app, {"thread_id": "t-bob", "user_id": "bob"})

    bob = await cp.aget_state({"thread_id": "t-bob", "user_id": "bob"})
    assert texts(bob.context) == [("user", "I am Bob")]
    assert app._state is None or app._state.context == []
