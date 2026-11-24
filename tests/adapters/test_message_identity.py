"""Converters stamp their own timestamp, never the provider's created time."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tenxgraph.runtime.adapters.llm.anthropic_converter import AnthropicConverter
from tenxgraph.runtime.adapters.llm.google_genai_converter import GoogleGenAIConverter
from tenxgraph.runtime.adapters.llm.openai_converter import OpenAIConverter
from tenxgraph.runtime.adapters.llm.openai_responses_converter import OpenAIResponsesConverter

from .test_anthropic_converter import _delta, _FakeStream, _response, _text_block
from .test_google_genai_converter import (
    MockCandidate,
    MockContent,
    MockGenerateContentResponse,
    MockPart,
)
from .test_openai_converter import MockModelResponse
from .test_openai_responses_converter import _make_message_item, _make_response


PROVIDER_CREATED = 1234567890  # 2009: any local stamp is later
PROVIDER_ID = "provider-id-1"


def _is_local(timestamp: float) -> bool:
    return abs(timestamp - time.time()) < 60 and timestamp != PROVIDER_CREATED


def _openai_response() -> MockModelResponse:
    return MockModelResponse(
        {
            "id": PROVIDER_ID,
            "created": PROVIDER_CREATED,
            "choices": [
                {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ],
        }
    )


def _openai_chunk(content: str) -> SimpleNamespace:
    delta = SimpleNamespace(
        content=content, reasoning_content=None, tool_calls=None, audio=None, images=None
    )
    return SimpleNamespace(
        id=PROVIDER_ID, model="gpt-4o", choices=[SimpleNamespace(delta=delta)], usage=None
    )


@pytest.mark.asyncio
@patch("tenxgraph.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
async def test_openai_chat_response_timestamp():
    message = await OpenAIConverter().convert_response(_openai_response())
    assert _is_local(message.timestamp)


@pytest.mark.asyncio
@patch("tenxgraph.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
async def test_openai_chat_empty_response_timestamp():
    response = MockModelResponse({"id": PROVIDER_ID, "created": PROVIDER_CREATED, "choices": []})
    message = await OpenAIConverter().convert_response(response)
    assert _is_local(message.timestamp)


@pytest.mark.asyncio
async def test_openai_chat_stream_timestamp():
    async def stream():
        yield _openai_chunk("a")
        yield _openai_chunk("b")

    messages = [m async for m in OpenAIConverter().convert_streaming_response({}, "n", stream())]
    assert _is_local(messages[-1].timestamp)


@pytest.mark.asyncio
async def test_responses_response_timestamp():
    response = _make_response(output=[_make_message_item("hi")], resp_id=PROVIDER_ID)
    message = await OpenAIResponsesConverter().convert_response(response)
    assert _is_local(message.timestamp)  # created_at is 1700000000


@pytest.mark.asyncio
async def test_responses_stream_timestamp():
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="a"),
        SimpleNamespace(type="response.output_text.delta", delta="b"),
        SimpleNamespace(type="response.completed", response=_make_response(resp_id=PROVIDER_ID)),
    ]

    async def stream():
        for event in events:
            yield event

    messages = [
        m async for m in OpenAIResponsesConverter().convert_streaming_response({}, "n", stream())
    ]
    assert _is_local(messages[-1].timestamp)


@pytest.mark.asyncio
async def test_google_response_timestamp():
    response = MockGenerateContentResponse(
        candidates=[MockCandidate(content=MockContent(parts=[MockPart(text="hi")]))],
        response_id=PROVIDER_ID,
        create_time=datetime.now() - timedelta(days=1),
    )
    message = await GoogleGenAIConverter().convert_response(response)
    assert _is_local(message.timestamp)


@pytest.mark.asyncio
async def test_google_stream_timestamp():
    def chunk(text: str) -> MockGenerateContentResponse:
        return MockGenerateContentResponse(
            candidates=[MockCandidate(content=MockContent(parts=[MockPart(text=text)]))],
            response_id=PROVIDER_ID,
        )

    async def stream():
        yield chunk("a")
        yield chunk("b")

    messages = [
        m async for m in GoogleGenAIConverter().convert_streaming_response({}, "n", stream())
    ]
    assert _is_local(messages[-1].timestamp)


@pytest.mark.asyncio
async def test_anthropic_response_timestamp():
    message = await AnthropicConverter().convert_response(
        _response([_text_block("hi")], id=PROVIDER_ID)
    )
    assert _is_local(message.timestamp)


@pytest.mark.asyncio
async def test_anthropic_stream_timestamp():
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(id=PROVIDER_ID, model="claude", usage=None),
        ),
        _delta(0, "text_delta", text="a"),
        _delta(0, "text_delta", text="b"),
    ]
    messages = [
        m
        async for m in AnthropicConverter().convert_streaming_response(
            {"thread_id": "t"}, "n", _FakeStream(events)
        )
    ]
    assert _is_local(messages[-1].timestamp)


@pytest.mark.asyncio
@patch("tenxgraph.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
async def test_timestamps_increase_between_responses():
    converter = OpenAIConverter()
    first = await converter.convert_response(_openai_response())
    second = await converter.convert_response(_openai_response())
    assert second.timestamp >= first.timestamp
