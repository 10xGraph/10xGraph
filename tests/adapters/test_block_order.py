"""Every converter orders blocks reasoning, text, media, tool calls."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import (
    ImageBlock,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
)
from tenxgraph.runtime.adapters.llm.anthropic_converter import AnthropicConverter
from tenxgraph.runtime.adapters.llm.google_genai_converter import GoogleGenAIConverter
from tenxgraph.runtime.adapters.llm.openai_converter import OpenAIConverter
from tenxgraph.runtime.adapters.llm.openai_responses_converter import OpenAIResponsesConverter
from tenxgraph.runtime.adapters.llm.reasoning_utils import order_blocks

from .test_anthropic_converter import _delta, _FakeStream, _response, _text_block, _tool_block
from .test_google_genai_converter import (
    MockCandidate,
    MockContent,
    MockFunctionCall,
    MockGenerateContentResponse,
    MockPart,
)
from .test_openai_converter import MockModelResponse
from .test_openai_responses_converter import (
    _make_function_call_item,
    _make_message_item,
    _make_reasoning_item,
    _make_response,
)


EXPECTED = [ReasoningBlock, TextBlock, ToolCallBlock]


def _types(message: Message) -> list[type]:
    return [type(b) for b in message.content]


def test_order_blocks_is_stable_and_ranks_media_between_text_and_tools():
    image = ImageBlock(media={"kind": "url", "url": "http://x/y.png"})
    blocks = [
        ToolCallBlock(name="a", args={}, id="1"),
        TextBlock(text="one"),
        image,
        ReasoningBlock(summary="r1"),
        TextBlock(text="two"),
        ReasoningBlock(summary="r2"),
        ToolCallBlock(name="b", args={}, id="2"),
    ]
    out = order_blocks(blocks)
    assert [b.type for b in out] == [
        "reasoning",
        "reasoning",
        "text",
        "text",
        "image",
        "tool_call",
        "tool_call",
    ]
    assert [b.summary for b in out[:2]] == ["r1", "r2"]
    assert [b.text for b in out[2:4]] == ["one", "two"]
    assert [b.name for b in out[5:]] == ["a", "b"]


# ---- Google ---------------------------------------------------------------


def _google_parts():
    # Provider order deliberately puts text and the call before the thought.
    return [
        MockPart(text="answer"),
        MockPart(function_call=MockFunctionCall("calc", {"x": 1})),
        MockPart(text="thinking", thought=True),
    ]


@pytest.mark.asyncio
async def test_google_non_stream_order():
    response = MockGenerateContentResponse(
        candidates=[MockCandidate(content=MockContent(parts=_google_parts()))]
    )
    message = await GoogleGenAIConverter().convert_response(response)
    assert _types(message) == EXPECTED


@pytest.mark.asyncio
async def test_google_stream_delta_and_final_order():
    chunk = MockGenerateContentResponse(
        candidates=[MockCandidate(content=MockContent(parts=_google_parts()))]
    )

    async def stream():
        yield chunk

    messages = [
        m
        async for m in GoogleGenAIConverter().convert_streaming_response(
            config={}, node_name="n", response=stream()
        )
    ]
    delta, final = messages[0], messages[-1]
    assert delta.delta is True and _types(delta) == EXPECTED
    assert final.delta is False and _types(final) == EXPECTED
    assert final.content[0].summary == "thinking"


# ---- OpenAI chat ----------------------------------------------------------


def _openai_chunk(content=None, reasoning=None, tool_calls=None):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
        audio=None,
        images=None,
    )
    return SimpleNamespace(id="c1", model="gpt-4o", choices=[SimpleNamespace(delta=delta)])


@pytest.mark.asyncio
@patch("tenxgraph.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
async def test_openai_non_stream_order():
    response = MockModelResponse(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "thinking",
                        "tool_calls": [
                            SimpleNamespace(
                                id="tc1",
                                type="function",
                                function=SimpleNamespace(name="calc", arguments='{"x": 1}'),
                            )
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    message = await OpenAIConverter().convert_response(response)
    assert _types(message) == EXPECTED


def test_openai_delta_with_text_and_reasoning_puts_reasoning_first():
    chunk = _openai_chunk(content="answer", reasoning="thinking")
    _, _, message = OpenAIConverter()._process_chunk(chunk, "", "", [], set())
    assert _types(message) == [ReasoningBlock, TextBlock]
    assert message.delta is True


@pytest.mark.asyncio
async def test_openai_stream_final_order():
    tool = SimpleNamespace(
        id="tc1",
        type="function",
        function=SimpleNamespace(name="calc", arguments='{"x": 1}'),
    )

    async def stream():
        yield _openai_chunk(content="answer", reasoning="thinking")
        yield _openai_chunk(tool_calls=[tool])

    messages = [m async for m in OpenAIConverter().convert_streaming_response({}, "n", stream())]
    assert _types(messages[0]) == [ReasoningBlock, TextBlock]
    assert _types(messages[1]) == [ToolCallBlock]
    final = messages[-1]
    assert final.delta is False and _types(final) == EXPECTED
    assert final.content[0].summary == "thinking"


# ---- OpenAI Responses -----------------------------------------------------


@pytest.mark.asyncio
async def test_responses_non_stream_order():
    # Provider order puts the message and call ahead of the reasoning item.
    response = _make_response(
        output=[
            _make_message_item("answer"),
            _make_function_call_item(),
            _make_reasoning_item("thinking"),
        ]
    )
    message = await OpenAIResponsesConverter().convert_response(response)
    assert _types(message) == EXPECTED


@pytest.mark.asyncio
async def test_responses_stream_delta_and_final_order():
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="answer"),
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="thinking"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="function_call", name="calc", call_id="c1"),
        ),
        SimpleNamespace(type="response.function_call_arguments.delta", delta='{"x": 1}'),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call", name="calc", call_id="c1", arguments='{"x": 1}'
            ),
        ),
        SimpleNamespace(type="response.completed", response=_make_response(output=[])),
    ]

    async def stream():
        for event in events:
            yield event

    messages = [
        m
        async for m in OpenAIResponsesConverter().convert_streaming_response(
            config={}, node_name="n", response=stream()
        )
    ]
    reasoning_deltas = [m for m in messages if m.delta and _types(m) == [ReasoningBlock]]
    assert [m.content[0].summary for m in reasoning_deltas] == ["thinking"]
    final = messages[-1]
    assert final.delta is False and _types(final) == EXPECTED


# ---- Anthropic ------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_non_stream_order():
    response = _response(
        [
            SimpleNamespace(type="thinking", thinking="thinking", signature="SIG"),
            _text_block("answer"),
            _tool_block("toolu_1", "calc", {"x": 1}),
        ],
        stop_reason="tool_use",
    )
    message = await AnthropicConverter().convert_response(response)
    assert _types(message) == EXPECTED
    assert message.content[0].signature == "SIG"


@pytest.mark.asyncio
async def test_anthropic_stream_delta_and_final_order():
    events = [
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="thinking", thinking="", signature=""),
        ),
        _delta(0, "thinking_delta", thinking="thinking"),
        _delta(0, "signature_delta", signature="SIG"),
        SimpleNamespace(type="content_block_stop", index=0),
        _delta(1, "text_delta", text="answer"),
        SimpleNamespace(
            type="content_block_start",
            index=2,
            content_block=SimpleNamespace(type="tool_use", id="t1", name="calc"),
        ),
        _delta(2, "input_json_delta", partial_json='{"x": 1}'),
        SimpleNamespace(type="content_block_stop", index=2),
    ]
    messages = [
        m
        async for m in AnthropicConverter().convert_streaming_response(
            {"thread_id": "t1"}, "agent", _FakeStream(events)
        )
    ]
    assert [type(m.content[0]) for m in messages[:2]] == [ReasoningBlock, TextBlock]
    final = messages[-1]
    assert final.delta is False and _types(final) == EXPECTED
    assert final.content[0].summary == "thinking"
    assert final.content[0].signature == "SIG"
