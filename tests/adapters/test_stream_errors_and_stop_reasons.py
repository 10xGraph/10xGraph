"""Stream error propagation, streaming usage, truncated tool calls and stop reasons.

Covers G-07, G-48, G-75, G-76 and G-77. SDK objects are lightweight stand-ins:
the converters read them by attribute only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tenxgraph.runtime.adapters.llm.anthropic_converter import AnthropicConverter
from tenxgraph.runtime.adapters.llm.google_genai_converter import GoogleGenAIConverter
from tenxgraph.runtime.adapters.llm.openai_converter import OpenAIConverter
from tenxgraph.runtime.adapters.llm.tool_arguments import parse_tool_arguments


# ---------------------------------------------------------------- helpers


def _oa_chunk(content=None, finish_reason=None, usage=None, choices=True):
    delta = SimpleNamespace(
        content=content, reasoning_content=None, tool_calls=None, audio=None, images=None
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(
        id="chat-1",
        model="gpt-4o",
        choices=[choice] if choices else [],
        usage=usage,
    )


def _oa_usage(prompt=11, completion=7):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        completion_tokens_details=None,
        prompt_tokens_details=None,
    )


async def _collect(agen):
    return [m async for m in agen]


def _an_usage():
    return SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def _an_response(content, stop_reason):
    return SimpleNamespace(
        id="msg_01",
        role="assistant",
        model="claude",
        content=content,
        stop_reason=stop_reason,
        stop_details=None,
        usage=_an_usage(),
    )


def _an_tool(block_id="tu_1", name="calc", payload=None):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=payload or {})


def _an_stream(events):
    async def gen():
        for e in events:
            yield e

    return gen()


def _an_tool_events(stop_reason, *, finish_block=True):
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(id="m1", model="claude", usage=_an_usage()),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="tool_use", id="tu_1", name="calc"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"x": 1}'),
        ),
    ]
    if finish_block:
        events.append(SimpleNamespace(type="content_block_stop", index=0))
    events.append(
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason=stop_reason, stop_details=None),
            usage=SimpleNamespace(output_tokens=9),
        )
    )
    return events


def _g_part(text=None):
    return SimpleNamespace(
        text=text, thought=False, function_call=None, inline_data=None, file_data=None
    )


def _g_candidate(text=None, finish_reason=None):
    content = SimpleNamespace(parts=[_g_part(text)] if text is not None else [])
    return SimpleNamespace(content=content, finish_reason=finish_reason)


def _g_usage(prompt=4, cand=6):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=cand,
        total_token_count=prompt + cand,
        cached_content_token_count=0,
        thoughts_token_count=0,
    )


def _g_chunk(text=None, finish_reason=None, usage=None):
    return SimpleNamespace(
        candidates=[_g_candidate(text, finish_reason)],
        usage_metadata=usage,
        model_version="gemini",
    )


# ---------------------------------------------------------------- G-77


class TestRecursionError:
    def test_deeply_nested_arguments_return_error(self):
        raw = "[" * 200_000
        args, error = parse_tool_arguments(raw, tool_name="t")
        assert args == {}
        assert error

    def test_deeply_nested_object_returns_error(self):
        raw = '{"a":' * 200_000
        args, error = parse_tool_arguments(raw, tool_name="t")
        assert args == {}
        assert error


# ---------------------------------------------------------------- G-07


class TestOpenAIStreamErrors:
    @pytest.mark.asyncio
    async def test_async_mid_stream_error_propagates(self):
        async def stream():
            yield _oa_chunk("Hel")
            yield _oa_chunk("lo")
            raise RuntimeError("connection dropped")

        seen = []
        with pytest.raises(RuntimeError, match="connection dropped"):
            async for m in OpenAIConverter().convert_streaming_response({}, "n", stream()):
                seen.append(m)
        # No final "complete" message may have been produced.
        assert all(m.delta for m in seen)

    @pytest.mark.asyncio
    async def test_sync_mid_stream_error_propagates(self):
        def stream():
            yield _oa_chunk("Hel")
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await _collect(OpenAIConverter().convert_streaming_response({}, "n", stream()))

    @pytest.mark.asyncio
    async def test_error_before_first_chunk_propagates(self):
        async def stream():
            raise RuntimeError("429")
            yield  # pragma: no cover

        with pytest.raises(RuntimeError, match="429"):
            await _collect(OpenAIConverter().convert_streaming_response({}, "n", stream()))


# ---------------------------------------------------------------- G-48


class TestStreamingUsage:
    @pytest.mark.asyncio
    async def test_openai_usage_chunk_reaches_final_message(self):
        async def stream():
            yield _oa_chunk("Hi", finish_reason="stop")
            yield _oa_chunk(choices=False, usage=_oa_usage(11, 7))

        final = (await _collect(OpenAIConverter().convert_streaming_response({}, "n", stream())))[
            -1
        ]
        assert final.usages.prompt_tokens == 11
        assert final.usages.completion_tokens == 7
        assert final.usages.total_tokens == 18
        assert final.metadata["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_google_usage_reaches_final_message(self):
        async def stream():
            yield _g_chunk("Hi")
            yield _g_chunk(" there", finish_reason="STOP", usage=_g_usage(4, 6))

        final = (
            await _collect(GoogleGenAIConverter().convert_streaming_response({}, "n", stream()))
        )[-1]
        assert final.usages.prompt_tokens == 4
        assert final.usages.completion_tokens == 6
        assert final.usages.total_tokens == 10


# ---------------------------------------------------------------- G-75


class TestAnthropicTruncatedToolCall:
    @pytest.mark.asyncio
    async def test_non_streaming_max_tokens_tool_use_not_dispatched(self):
        response = _an_response([_an_tool(payload={"x": 1})], "max_tokens")
        msg = await AnthropicConverter().convert_response(response)
        assert not msg.tools_calls
        assert not [b for b in msg.content if b.type == "tool_call"]
        assert msg.metadata["stop_issue"] == "truncated"
        assert msg.metadata["dropped_tool_calls"] == ["calc"]

    @pytest.mark.asyncio
    async def test_non_streaming_tool_use_stop_still_dispatched(self):
        response = _an_response([_an_tool(payload={"x": 1})], "tool_use")
        msg = await AnthropicConverter().convert_response(response)
        assert msg.tools_calls and msg.tools_calls[0]["id"] == "tu_1"

    @pytest.mark.asyncio
    async def test_streaming_max_tokens_complete_block_not_dispatched(self):
        stream = _an_stream(_an_tool_events("max_tokens"))
        final = (await _collect(AnthropicConverter().convert_streaming_response({}, "n", stream)))[
            -1
        ]
        assert not final.tools_calls
        assert not [b for b in final.content if b.type == "tool_call"]
        assert final.metadata["stop_issue"] == "truncated"


# ---------------------------------------------------------------- G-76


class TestStopReasons:
    @pytest.mark.asyncio
    async def test_anthropic_context_window_exceeded(self):
        response = _an_response([], "model_context_window_exceeded")
        msg = await AnthropicConverter().convert_response(response)
        assert msg.metadata["stop_issue"] == "context_window_exceeded"

    @pytest.mark.asyncio
    async def test_anthropic_stream_context_window_exceeded(self):
        stream = _an_stream(_an_tool_events("model_context_window_exceeded", finish_block=False))
        final = (await _collect(AnthropicConverter().convert_streaming_response({}, "n", stream)))[
            -1
        ]
        assert final.metadata["stop_issue"] == "context_window_exceeded"

    @pytest.mark.asyncio
    async def test_anthropic_normal_stop_has_no_issue(self):
        msg = await AnthropicConverter().convert_response(_an_response([], "end_turn"))
        assert "stop_issue" not in msg.metadata

    @pytest.mark.asyncio
    async def test_openai_content_filter_stream(self):
        async def stream():
            yield _oa_chunk("par", finish_reason="content_filter")

        final = (await _collect(OpenAIConverter().convert_streaming_response({}, "n", stream())))[
            -1
        ]
        assert final.metadata["stop_issue"] == "content_filter"

    @pytest.mark.asyncio
    async def test_openai_length_stream_marked_truncated(self):
        async def stream():
            yield _oa_chunk("par", finish_reason="length")

        final = (await _collect(OpenAIConverter().convert_streaming_response({}, "n", stream())))[
            -1
        ]
        assert final.metadata["stop_issue"] == "truncated"

    @pytest.mark.asyncio
    async def test_openai_non_streaming_content_filter(self):
        message = SimpleNamespace(
            role="assistant", content="x", tool_calls=None, audio=None, images=None
        )
        response = SimpleNamespace(
            id="c1",
            model="gpt",
            created=1,
            usage=None,
            choices=[SimpleNamespace(message=message, finish_reason="content_filter")],
            system_fingerprint=None,
            service_tier=None,
            model_dump=dict,
        )
        msg = await OpenAIConverter().convert_response(response)
        assert msg.metadata["stop_issue"] == "content_filter"

    @pytest.mark.asyncio
    async def test_google_safety_non_streaming(self):
        response = SimpleNamespace(
            candidates=[_g_candidate(None, "FinishReason.SAFETY")],
            usage_metadata=None,
            model_version="gemini",
            response_id=None,
            parsed=None,
        )
        msg = await GoogleGenAIConverter().convert_response(response)
        assert msg.metadata["stop_issue"] == "content_filter"

    @pytest.mark.asyncio
    async def test_google_safety_stream(self):
        async def stream():
            yield _g_chunk(None, finish_reason="SAFETY")

        final = (
            await _collect(GoogleGenAIConverter().convert_streaming_response({}, "n", stream()))
        )[-1]
        assert final.metadata["stop_issue"] == "content_filter"
        assert final.metadata["finish_reason"] == "SAFETY"

    @pytest.mark.asyncio
    async def test_google_stop_has_no_issue(self):
        async def stream():
            yield _g_chunk("ok", finish_reason="STOP")

        final = (
            await _collect(GoogleGenAIConverter().convert_streaming_response({}, "n", stream()))
        )[-1]
        assert "stop_issue" not in final.metadata
