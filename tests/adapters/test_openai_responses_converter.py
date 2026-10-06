"""Tests for OpenAI Responses API converter."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tenxgraph.runtime.adapters.llm.openai_responses_converter import (
    OpenAIResponsesConverter,
    is_responses_api_response,
)
from tenxgraph.core.state.message_block import ReasoningBlock, TextBlock, ToolCallBlock


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_usage(input_tokens=10, output_tokens=20, total_tokens=30,
                reasoning_tokens=5, cached_tokens=2):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )


def _make_reasoning_item(text="I think therefore I am", item_id="rs_1"):
    return SimpleNamespace(
        type="reasoning", id=item_id,
        summary=[SimpleNamespace(type="summary_text", text=text)],
    )


def _make_message_item(text="The answer is 42", item_id="msg_1"):
    return SimpleNamespace(
        type="message", id=item_id, role="assistant", status="completed",
        content=[SimpleNamespace(type="output_text", text=text, annotations=[])],
    )


def _make_function_call_item(name="get_weather", arguments='{"city":"NYC"}',
                              call_id="call_1"):
    return SimpleNamespace(
        type="function_call", name=name, arguments=arguments,
        call_id=call_id, id="fc_1",
    )


def _make_response(output=None, usage=None, model="o4-mini",
                    status="completed", resp_id="resp_1"):
    r = SimpleNamespace(
        id=resp_id, model=model, status=status, created_at=1700000000,
        output=output or [], usage=usage or _make_usage(),
    )
    r.model_dump = MagicMock(return_value={"id": resp_id, "model": model})
    return r


# ---------------------------------------------------------------------------
# Tests: is_responses_api_response
# ---------------------------------------------------------------------------


class TestResponsesAPIDetection:
    def test_responses_api_detected(self):
        assert is_responses_api_response(_make_response()) is True

    def test_chat_completion_not_detected(self):
        cc = SimpleNamespace(choices=[{"message": {}}], id="chatcmpl-1")
        assert is_responses_api_response(cc) is False


# ---------------------------------------------------------------------------
# Tests: OpenAIResponsesConverter.convert_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOpenAIResponsesConverter:
    async def test_text_only(self):
        msg = await OpenAIResponsesConverter().convert_response(
            _make_response(output=[_make_message_item("Hello!")])
        )
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextBlock)
        assert msg.content[0].text == "Hello!"

    async def test_reasoning_only(self):
        msg = await OpenAIResponsesConverter().convert_response(
            _make_response(output=[_make_reasoning_item("Deep thought")])
        )
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ReasoningBlock)
        assert msg.reasoning == "Deep thought"

    async def test_mixed_reasoning_and_text(self):
        msg = await OpenAIResponsesConverter().convert_response(
            _make_response(output=[
                _make_reasoning_item("Thinking..."),
                _make_message_item("Answer"),
            ])
        )
        assert isinstance(msg.content[0], ReasoningBlock)
        assert isinstance(msg.content[1], TextBlock)

    async def test_function_call(self):
        msg = await OpenAIResponsesConverter().convert_response(
            _make_response(output=[
                _make_function_call_item("search", '{"q":"test"}', "call_1"),
            ])
        )
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ToolCallBlock)
        assert msg.content[0].name == "search"
        assert msg.content[0].id == "call_1"

    async def test_empty_output(self):
        msg = await OpenAIResponsesConverter().convert_response(
            _make_response(output=[])
        )
        assert msg.content == []
        assert msg.reasoning == ""

    async def test_token_usage_mapping(self):
        usage = _make_usage(100, 200, 300, reasoning_tokens=50, cached_tokens=10)
        msg = await OpenAIResponsesConverter().convert_response(
            _make_response(output=[_make_message_item("Hi")], usage=usage)
        )
        assert msg.usages.prompt_tokens == 100
        assert msg.usages.completion_tokens == 200
        assert msg.usages.total_tokens == 300
        assert msg.usages.reasoning_tokens == 50
        assert msg.usages.cache_read_input_tokens == 10

    async def test_metadata_extraction(self):
        msg = await OpenAIResponsesConverter().convert_response(
            _make_response(output=[_make_message_item("Hi")], model="o4-mini",
                           resp_id="resp_abc")
        )
        assert msg.metadata["provider"] == "openai"
        assert msg.metadata["model"] == "o4-mini"


# ---------------------------------------------------------------------------
# Tests: Streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResponsesConverterStreaming:
    async def test_text_deltas_accumulated(self):
        events = [
            SimpleNamespace(type="response.output_item.added",
                            item=SimpleNamespace(type="message", id="msg_1")),
            SimpleNamespace(type="response.output_text.delta", delta="Hello "),
            SimpleNamespace(type="response.output_text.delta", delta="world!"),
            SimpleNamespace(type="response.completed",
                            response=_make_response(output=[])),
        ]

        async def _aiter():
            for e in events:
                yield e

        messages = []
        async for msg in OpenAIResponsesConverter().convert_streaming_response(
            config={}, node_name="test", response=_aiter()
        ):
            messages.append(msg)

        final = messages[-1]
        assert final.delta is False
        assert any(isinstance(b, TextBlock) for b in final.content)
        assert any(b.text == "Hello world!" for b in final.content
                   if isinstance(b, TextBlock))

    async def test_reasoning_deltas_accumulated(self):
        events = [
            SimpleNamespace(type="response.reasoning_summary_text.delta",
                            delta="Think "),
            SimpleNamespace(type="response.reasoning_summary_text.delta",
                            delta="step 1"),
            SimpleNamespace(type="response.output_text.delta", delta="Answer"),
            SimpleNamespace(type="response.completed",
                            response=_make_response(output=[])),
        ]

        async def _aiter():
            for e in events:
                yield e

        messages = []
        async for msg in OpenAIResponsesConverter().convert_streaming_response(
            config={}, node_name="test", response=_aiter()
        ):
            messages.append(msg)

        final = messages[-1]
        assert any(isinstance(b, ReasoningBlock) for b in final.content)
        assert final.reasoning == "Think step 1"


# ---------------------------------------------------------------------------
# Tests: Converter registry & auto-detection
# ---------------------------------------------------------------------------


class TestConverterRegistration:
    def test_openai_responses_key(self):
        from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
        mrc = ModelResponseConverter(response="dummy", converter="openai_responses")
        assert isinstance(mrc.converter, OpenAIResponsesConverter)

    def test_openai_key_still_works(self):
        from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
        from tenxgraph.runtime.adapters.llm.openai_converter import OpenAIConverter
        mrc = ModelResponseConverter(response="dummy", converter="openai")
        assert isinstance(mrc.converter, OpenAIConverter)

    def test_invalid_key_raises(self):
        from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
        with pytest.raises(ValueError, match="Unsupported converter"):
            ModelResponseConverter(response="dummy", converter="not_real")


@pytest.mark.asyncio
class TestAutoDetectionBridge:
    async def test_responses_api_delegated_via_openai_converter(self):
        """Responses API response passed to OpenAIConverter is auto-delegated."""
        from tenxgraph.runtime.adapters.llm.openai_converter import OpenAIConverter
        msg = await OpenAIConverter().convert_response(
            _make_response(output=[_make_message_item("Delegated!")])
        )
        assert isinstance(msg.content[0], TextBlock)
        assert msg.content[0].text == "Delegated!"
