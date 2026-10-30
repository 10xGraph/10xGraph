"""G-59: opt-in stream_idle_timeout closes a stalled LLM stream and raises a typed error."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.core.exceptions import LLMStreamIdleTimeoutError, NodeTimeoutError
from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.agent_internal.constants import RetryConfig
from tenxgraph.core.llm import set_default_stream_idle_timeout
from tenxgraph.core.llm.stream_idle import (
    get_default_stream_idle_timeout,
    resolve_stream_idle_timeout,
    stream_idle_timeout_var,
)
from tenxgraph.runtime.adapters.llm.anthropic_converter import AnthropicConverter
from tenxgraph.runtime.adapters.llm.google_genai_converter import GoogleGenAIConverter
from tenxgraph.runtime.adapters.llm.openai_converter import OpenAIConverter
from tenxgraph.runtime.adapters.llm.openai_responses_converter import OpenAIResponsesConverter


class StallingStream:
    """Yields its items, then never produces another chunk."""

    def __init__(self, items):
        self.items = list(items)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.items:
            return self.items.pop(0)
        await asyncio.sleep(3600)
        return None

    async def aclose(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("TENXGRAPH_LLM_STREAM_IDLE_TIMEOUT", raising=False)
    set_default_stream_idle_timeout(None)
    token = stream_idle_timeout_var.set(None)
    yield
    stream_idle_timeout_var.reset(token)
    set_default_stream_idle_timeout(None)


async def _consume(gen):
    return [m async for m in gen]


def _chunk():
    return SimpleNamespace(
        model="m",
        id="c1",
        usage=None,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=None,
                delta=SimpleNamespace(content="hi", tool_calls=None, role="assistant"),
            )
        ],
    )


def test_no_default_and_settings_resolution(monkeypatch):
    assert get_default_stream_idle_timeout() is None
    assert resolve_stream_idle_timeout({}) is None
    monkeypatch.setenv("TENXGRAPH_LLM_STREAM_IDLE_TIMEOUT", "7")
    assert resolve_stream_idle_timeout({}) == 7.0
    set_default_stream_idle_timeout(3)
    assert resolve_stream_idle_timeout({}) == 3.0
    assert resolve_stream_idle_timeout({"stream_idle_timeout": 1.5}) == 1.5
    with pytest.raises(ValueError):
        set_default_stream_idle_timeout(0)


def test_error_is_a_node_timeout():
    assert issubclass(LLMStreamIdleTimeoutError, NodeTimeoutError)


@pytest.mark.asyncio
async def test_openai_chat_stall_raises_and_closes_stream():
    stream_idle_timeout_var.set(0.05)
    stream = StallingStream([_chunk()])
    with pytest.raises(LLMStreamIdleTimeoutError):
        await _consume(OpenAIConverter().convert_streaming_response({}, "n", stream))
    assert stream.closed is True


@pytest.mark.asyncio
async def test_openai_responses_stall_raises():
    stream_idle_timeout_var.set(0.05)
    stream = StallingStream([])
    with pytest.raises(LLMStreamIdleTimeoutError):
        await _consume(
            OpenAIResponsesConverter().convert_streaming_response(
                config={}, node_name="n", response=stream
            )
        )
    assert stream.closed is True


@pytest.mark.asyncio
async def test_google_stall_raises():
    stream_idle_timeout_var.set(0.05)
    stream = StallingStream([])
    with pytest.raises(LLMStreamIdleTimeoutError):
        await _consume(
            GoogleGenAIConverter().convert_streaming_response(
                config={}, node_name="n", response=stream
            )
        )
    assert stream.closed is True


@pytest.mark.asyncio
async def test_anthropic_stall_raises():
    stream_idle_timeout_var.set(0.05)
    stream = StallingStream([])
    with pytest.raises(LLMStreamIdleTimeoutError):
        await _consume(
            AnthropicConverter().convert_streaming_response(
                config={}, node_name="n", response=stream
            )
        )
    assert stream.closed is True


@pytest.mark.asyncio
async def test_unset_timeout_does_not_interrupt_a_slow_stream():
    class Slow(StallingStream):
        async def __anext__(self):
            if self.items:
                await asyncio.sleep(0.05)
                return self.items.pop(0)
            raise StopAsyncIteration

    msgs = await _consume(OpenAIConverter().convert_streaming_response({}, "n", Slow([_chunk()])))
    assert msgs


@pytest.mark.asyncio
async def test_cancel_during_the_wait_propagates():
    stream_idle_timeout_var.set(30)
    task = asyncio.create_task(
        _consume(OpenAIConverter().convert_streaming_response({}, "n", StallingStream([])))
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_agent_sets_the_per_call_value_from_llm_kwargs_then_global():
    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        agent = Agent(
            model="gpt-4o",
            provider="openai",
            retry_config=RetryConfig(max_retries=0, jitter=0.0),
            reasoning_config=None,
            stream_idle_timeout=2.5,
        )
        other = Agent(model="gpt-4o", provider="openai", reasoning_config=None)
    seen = []

    async def fake_call(self, messages, tools=None, stream=False, **kw):
        seen.append(stream_idle_timeout_var.get())
        return SimpleNamespace()

    set_default_stream_idle_timeout(9)
    with patch.object(Agent, "_call_llm", fake_call):
        await agent._call_llm_with_retry([])
        await other._call_llm_with_retry([])
    assert seen == [2.5, 9.0]


def test_search_helper_clients_get_a_request_timeout():
    from tenxgraph.core.llm import get_default_llm_timeout
    from tenxgraph.prebuilt.tools import search

    captured = []

    class FakeClient:
        def __init__(self, **kw):
            captured.append(kw["http_options"].timeout)
            self.models = SimpleNamespace(
                generate_content=lambda **k: SimpleNamespace(text="x", candidates=[])
            )

    with patch("google.genai.Client", FakeClient):
        with contextlib.suppress(Exception):
            search._google_web_search_sync("q", "gemini-2.5-flash", 100)
        with contextlib.suppress(Exception):
            search._vertex_ai_search_sync("q", "ds", "gemini-2.5-flash", 100)
    assert captured == [int(get_default_llm_timeout() * 1000)] * 2


# ---------------------------------------------------------------- retry behaviour


def _retry_agent(**kw):
    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        return Agent(
            model="gpt-4o",
            provider="openai",
            retry_config=RetryConfig(max_retries=2, initial_delay=0.0, jitter=0.0),
            reasoning_config=None,
            stream_idle_timeout=0.05,
            **kw,
        )


@pytest.mark.asyncio
async def test_stall_before_first_chunk_is_retried():
    agent = _retry_agent()
    opened = []

    async def fake_call(self, messages, tools=None, stream=False, **kw):
        opened.append(StallingStream([] if len(opened) == 0 else [_chunk()]))
        return opened[-1]

    with patch.object(Agent, "_call_llm", fake_call):
        result = await agent._call_llm_with_retry([], stream=True)
        assert len(opened) == 2
        assert opened[0].closed is True
        gen = OpenAIConverter().convert_streaming_response({}, "n", result)
        first = await gen.__anext__()  # the replayed first chunk reached the converter
        await gen.aclose()
    assert first is not None


@pytest.mark.asyncio
async def test_stall_before_first_chunk_on_a_manager_stream_is_retried():
    agent = _retry_agent()
    exits = []

    class Manager:
        def __init__(self, stream):
            self.stream = stream

        async def __aenter__(self):
            return self.stream

        async def __aexit__(self, *exc):
            exits.append(1)

    calls = []

    async def fake_call(self, messages, tools=None, stream=False, **kw):
        calls.append(1)
        return Manager(StallingStream([] if len(calls) == 1 else [_chunk()]))

    with patch.object(Agent, "_call_llm", fake_call):
        result = await agent._call_llm_with_retry([], stream=True)
    assert len(calls) == 2 and exits == [1]
    async with result as s:
        assert [c async for c in _take(s, 1)]


async def _take(stream, n):
    async for item in stream:
        yield item
        n -= 1
        if n == 0:
            return


@pytest.mark.asyncio
async def test_stall_after_a_chunk_is_not_retried():
    agent = _retry_agent()
    opened = []

    async def fake_call(self, messages, tools=None, stream=False, **kw):
        opened.append(StallingStream([_chunk()]))
        return opened[-1]

    with patch.object(Agent, "_call_llm", fake_call):
        result = await agent._call_llm_with_retry([], stream=True)
        got = []
        with pytest.raises(LLMStreamIdleTimeoutError):
            async for m in OpenAIConverter().convert_streaming_response({}, "n", result):
                got.append(m)
    assert len(opened) == 1  # no second request, so no duplicated output
    assert got
