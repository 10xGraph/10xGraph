"""Each streamed item reaches the client as its own StreamChunk.

The stream is queued before it is consumed, so a chunk that is mutated and yielded again
would show every reader the last item. These tests stream a scripted Gemini reply through
``astream`` (mocked at ``Agent._call_google``; no real provider) and check that the delta
messages arrive in order, followed by exactly one non-delta message per model call, and
that no StreamChunk object is yielded twice.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.core.graph import Agent
from tenxgraph.core.state import Message
from tenxgraph.core.state.stream_chunks import StreamChunk, StreamEvent
from tenxgraph.prebuilt.agent import PlanActReflectAgent, ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tests.adapters.test_google_genai_converter import (
    MockCandidate,
    MockContent,
    MockFunctionCall,
    MockGenerateContentResponse,
    MockPart,
    MockUsageMetadata,
)


pytestmark = pytest.mark.asyncio


def _chunk(part, usage=False):
    return MockGenerateContentResponse(
        candidates=[MockCandidate(content=MockContent([part]))],
        usage_metadata=MockUsageMetadata(10, 100, 150, 40) if usage else None,
    )


def _text_reply(thought: str, text: str):
    """Three stream chunks: a thought, the first half of the text, the rest plus usage."""
    cut = len(text) // 2
    return [
        _chunk(MockPart(text=thought, thought=True)),
        _chunk(MockPart(text=text[:cut])),
        _chunk(MockPart(text=text[cut:]), usage=True),
    ]


def _tool_reply(name: str, args: dict):
    return [_chunk(MockPart(function_call=MockFunctionCall(name, args)), usage=True)]


def lookup(city: str) -> str:
    """Look up a city."""
    return f"sunny in {city}"


async def _stream(agent, replies: list):
    queue = list(replies)

    async def fake(self, messages, tools=None, stream=False, **kwargs):
        items = queue.pop(0)

        async def gen():
            for item in items:
                yield item

        return gen()

    with (
        patch.object(Agent, "_create_client", return_value=MagicMock()),
        patch.object(Agent, "_call_google", fake),
    ):
        app = agent.compile(checkpointer=InMemoryCheckpointer())
        return [
            c
            async for c in app.astream(
                {"messages": [Message.text_message("HI")]}, {"thread_id": "t", "user_id": "u"}
            )
        ]


def _node_messages(chunks, node):
    return [
        c.message
        for c in chunks
        if isinstance(c, StreamChunk)
        and c.event == StreamEvent.MESSAGE
        and c.message is not None
        and (c.metadata or {}).get("node") == node
        and c.message.role == "assistant"
    ]


def _assert_deltas_then_one_final(messages, calls):
    """Deltas arrive as distinct messages in order; each call ends with one non-delta."""
    finals = [m for m in messages if not m.delta]
    assert len(finals) == calls
    assert len({id(m) for m in messages}) == len(messages)
    # Every final is preceded by its deltas, never by another final first.
    seen_final = 0
    for m in messages:
        if m.delta:
            assert seen_final < calls
        else:
            seen_final += 1
    assert any(m.delta for m in messages)


def _assert_distinct(chunks):
    stream_chunks = [c for c in chunks if isinstance(c, StreamChunk)]
    assert len({id(c) for c in stream_chunks}) == len(stream_chunks)
    metas = [c.metadata for c in stream_chunks if c.metadata]
    assert len({id(m) for m in metas}) == len(metas)


async def test_agent_node_streams_deltas_then_one_final():
    agent = ReactAgent(model="gemini-2.5-flash", provider="google", tools=[])
    chunks = await _stream(agent, [_text_reply("thinking", "Hello, which city?")])
    msgs = _node_messages(chunks, "MAIN")
    _assert_deltas_then_one_final(msgs, calls=1)
    deltas = [m for m in msgs if m.delta]
    assert len(deltas) == 3
    # The deltas are different messages carrying their own content, not the final repeated.
    texts = ["".join(getattr(b, "text", "") for b in m.content) for m in deltas]
    assert any("Hello" in t for t in texts) and not all("Hello, which city?" in t for t in texts)
    assert sum(1 for m in msgs if m.usages) <= 1
    _assert_distinct(chunks)


async def test_plan_act_reflect_function_node_path():
    agent = PlanActReflectAgent(model="gemini-2.5-flash", provider="google")
    chunks = await _stream(
        agent,
        [
            _text_reply("planning", "Hello, which city?"),
            _text_reply("evaluating", "Greeting handled. [DONE]"),
        ],
    )
    for node in ("PLAN", "REFLECT"):
        msgs = _node_messages(chunks, node)
        _assert_deltas_then_one_final(msgs, calls=1)
    _assert_distinct(chunks)


async def test_tool_call_path_chunks_are_distinct():
    agent = ReactAgent(model="gemini-2.5-flash", provider="google", tools=[lookup])
    chunks = await _stream(
        agent,
        [_tool_reply("lookup", {"city": "Paris"}), _text_reply("done", "It is sunny in Paris.")],
    )
    tool_msgs = [
        c
        for c in chunks
        if isinstance(c, StreamChunk)
        and c.event == StreamEvent.MESSAGE
        and (c.data or {}).get("status") == "tool_result"
    ]
    assert len(tool_msgs) == 1
    main = _node_messages(chunks, "MAIN")
    assert len([m for m in main if not m.delta]) == 2
    _assert_distinct(chunks)
