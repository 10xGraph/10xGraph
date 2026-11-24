"""One streamed response gets one message id, shared by every delta and the final message."""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.runtime.adapters.llm.anthropic_converter import AnthropicConverter
from tenxgraph.runtime.adapters.llm.google_genai_converter import GoogleGenAIConverter
from tenxgraph.runtime.adapters.llm.openai_converter import OpenAIConverter
from tenxgraph.runtime.adapters.llm.openai_responses_converter import OpenAIResponsesConverter
from tenxgraph.utils.id_generator import BaseIDGenerator, IDType

from .test_anthropic_converter import _delta, _FakeStream
from .test_block_order import _openai_chunk
from .test_google_genai_converter import (
    MockCandidate,
    MockContent,
    MockGenerateContentResponse,
    MockPart,
)
from .test_openai_responses_converter import _make_response


class _Snowflake(BaseIDGenerator):
    """Every call returns a new integer, like a Snowflake generator."""

    def __init__(self) -> None:
        self._counter = itertools.count(1000)

    @property
    def id_type(self) -> IDType:
        return IDType.BIGINT

    def generate(self) -> int:
        return next(self._counter)


@pytest.fixture(autouse=True)
def snowflake():
    generator = _Snowflake()
    mock = MagicMock()
    mock.try_get.side_effect = lambda key, default=None: (
        generator if key is BaseIDGenerator else default
    )
    with patch("tenxgraph.core.state.message.InjectQ.get_instance", return_value=mock):
        yield generator


async def _openai():
    async def stream():
        yield _openai_chunk(content="a")
        yield _openai_chunk(content="b")

    return [m async for m in OpenAIConverter().convert_streaming_response({}, "n", stream())]


async def _responses():
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="a"),
        SimpleNamespace(type="response.output_text.delta", delta="b"),
        SimpleNamespace(type="response.completed", response=_make_response(output=[])),
    ]

    async def stream():
        for event in events:
            yield event

    converter = OpenAIResponsesConverter()
    return [m async for m in converter.convert_streaming_response({}, "n", stream())]


async def _anthropic():
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(id="msg_provider", model="m", usage=None),
        ),
        _delta(0, "text_delta", text="a"),
        _delta(0, "text_delta", text="b"),
    ]
    converter = AnthropicConverter()
    return [m async for m in converter.convert_streaming_response({}, "n", _FakeStream(events))]


async def _google():
    def chunk(text):
        part = MockPart(text=text)
        return MockGenerateContentResponse(
            candidates=[MockCandidate(content=MockContent(parts=[part]))]
        )

    async def stream():
        yield chunk("a")
        yield chunk("b")

    converter = GoogleGenAIConverter()
    return [m async for m in converter.convert_streaming_response({}, "n", stream())]


RUNNERS = {
    "openai": _openai,
    "responses": _responses,
    "anthropic": _anthropic,
    "google": _google,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", RUNNERS)
async def test_one_response_one_id(name):
    messages = await RUNNERS[name]()
    deltas = [m for m in messages if m.delta]
    assert len(deltas) >= 2
    assert not messages[-1].delta
    assert len({m.message_id for m in messages}) == 1
    assert isinstance(messages[0].message_id, int)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", RUNNERS)
async def test_two_responses_two_ids(name):
    first = await RUNNERS[name]()
    second = await RUNNERS[name]()
    assert first[-1].message_id != second[-1].message_id


@pytest.mark.asyncio
async def test_anthropic_provider_id_is_in_metadata_not_message_id():
    messages = await _anthropic()
    assert messages[-1].metadata["response_id"] == "msg_provider"
    assert messages[-1].message_id != "msg_provider"
