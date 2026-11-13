"""An Anthropic stream that fails at open (429/5xx) is retried by the Agent's retry loop.

``messages.stream()`` returns a context manager that sends the request only when entered.
With no stream idle timeout set, the request used to be sent later by the converter,
outside the retry loop, so a 500 or 529 at open failed the run. These tests drive a real
``AsyncAnthropic`` client over an ``httpx2.MockTransport``.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.agent_internal import execution
from tenxgraph.core.graph.agent_internal.constants import RetryConfig
from tenxgraph.core.llm import set_default_stream_idle_timeout
from tenxgraph.core.llm.stream_idle import stream_idle_timeout_var
from tenxgraph.runtime.adapters.llm.anthropic_converter import AnthropicConverter


# The anthropic extra is optional: CI jobs without it skip this module.
anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx2")  # anthropic 1.x runs on httpx2


def _sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


_HEAD = [
    _sse(
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }
    ),
    _sse(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    ),
    _sse(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        }
    ),
]
_TAIL = [
    _sse({"type": "content_block_stop", "index": 0}),
    _sse(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        }
    ),
    _sse({"type": "message_stop"}),
]


class Body(httpx.AsyncByteStream):
    """SSE body: sends ``chunks``, then raises ``error`` or blocks forever if asked."""

    def __init__(self, chunks, error=None, block=False):
        self.chunks = chunks
        self.error = error
        self.block = block
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error
        if self.block:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def _ok(body=None):
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=body or Body(_HEAD + _TAIL),
    )


def _status(code, headers=None):
    return httpx.Response(
        code,
        headers=headers or {},
        json={"type": "error", "error": {"type": "api_error", "message": "boom"}},
    )


@pytest.fixture(autouse=True)
def _no_idle_timeout(monkeypatch):
    monkeypatch.delenv("TENXGRAPH_LLM_STREAM_IDLE_TIMEOUT", raising=False)
    set_default_stream_idle_timeout(None)
    token = stream_idle_timeout_var.set(None)
    yield
    stream_idle_timeout_var.reset(token)


@pytest.fixture
def sleeps(monkeypatch):
    """Record back-off delays instead of sleeping."""
    seen: list[float] = []

    async def fake_sleep(delay, *a, **k):
        seen.append(delay)

    monkeypatch.setattr(execution.asyncio, "sleep", fake_sleep)
    return seen


def _agent(responses, **kw):
    """Default-config Agent whose Anthropic client answers with ``responses`` in order."""
    requests: list[httpx.Request] = []

    def handler(request):
        requests.append(request)
        return responses[min(len(requests), len(responses)) - 1]

    client = anthropic.AsyncAnthropic(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with patch.object(Agent, "_create_client", return_value=client):
        agent = Agent(model="claude-sonnet-4-5", provider="anthropic", **kw)
    return agent, requests


async def _run(agent):
    result = await agent._call_llm_with_retry([{"role": "user", "content": "hi"}], stream=True)
    return [m async for m in AnthropicConverter().convert_streaming_response({}, "n", result)]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [429, 500, 529])
async def test_error_at_stream_open_is_retried_with_default_config(code, sleeps):
    agent, requests = _agent([_status(code), _status(code), _ok()])
    messages = await _run(agent)
    assert len(requests) == 3
    assert len(sleeps) == 2
    final = [m for m in messages if not m.delta]
    assert final and final[-1].text() == "hi"


@pytest.mark.asyncio
async def test_retry_after_is_honored_at_stream_open(sleeps):
    agent, requests = _agent([_status(529, {"retry-after": "3"}), _ok()])
    await _run(agent)
    assert len(requests) == 2
    assert sleeps == [3.0]


@pytest.mark.asyncio
async def test_bad_request_at_stream_open_is_not_retried(sleeps):
    agent, requests = _agent([_status(400), _ok()])
    with pytest.raises(anthropic.BadRequestError):
        await agent._call_llm_with_retry([{"role": "user", "content": "hi"}], stream=True)
    assert len(requests) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_circuit_breaker_sees_the_failure_at_open(sleeps):
    agent, _ = _agent(
        [_status(500)],
        retry_config=RetryConfig(max_retries=1, circuit_breaker_enabled=True),
    )
    with pytest.raises(anthropic.InternalServerError):
        await agent._call_llm_with_retry([{"role": "user", "content": "hi"}], stream=True)
    breaker = agent._get_circuit_breaker("anthropic", "claude-sonnet-4-5", agent.retry_config)
    assert breaker.failure_count == 1


@pytest.mark.asyncio
async def test_error_after_a_chunk_is_not_retried(sleeps):
    body = Body(_HEAD, error=httpx.ReadError("connection reset"))
    agent, requests = _agent([_ok(body), _ok()])
    result = await agent._call_llm_with_retry([{"role": "user", "content": "hi"}], stream=True)
    got = []
    with pytest.raises(Exception):  # noqa: B017
        async for m in AnthropicConverter().convert_streaming_response({}, "n", result):
            got.append(m)
    assert [m.text() for m in got if m.delta] == ["hi"]
    assert len(requests) == 1  # no second request, so no repeated output
    assert body.closed is True


@pytest.mark.asyncio
async def test_stream_closed_when_cancelled_before_the_first_event():
    body = Body([], block=True)
    agent, requests = _agent([_ok(body)])
    task = asyncio.create_task(
        agent._call_llm_with_retry([{"role": "user", "content": "hi"}], stream=True)
    )
    for _ in range(100):
        if requests:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert body.closed is True


@pytest.mark.asyncio
async def test_stream_closed_when_the_consumer_stops_early():
    body = Body(_HEAD, block=True)
    agent, _ = _agent([_ok(body)])
    result = await agent._call_llm_with_retry([{"role": "user", "content": "hi"}], stream=True)
    gen = AnthropicConverter().convert_streaming_response({}, "n", result)
    first = await gen.__anext__()
    await gen.aclose()
    assert first.text() == "hi"
    assert body.closed is True
