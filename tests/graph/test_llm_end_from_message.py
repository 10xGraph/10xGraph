"""The LLM END event is built from the converted Message, for real SDK response types (CA-1).

Each test runs a real provider response through ``ModelResponseConverter`` with the Agent's
end hook attached, and checks the token counts, response id and model the END event carries.
"""

from unittest.mock import patch

import pytest

from tenxgraph.core.graph.agent_internal.execution import _make_llm_end_hook
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.runtime.publisher.events import Event, EventModel, EventType


async def _end_event(response, converter_key: str, *, stream: bool = False) -> dict:
    event = EventModel(event=Event.LLM_CALL, event_type=EventType.START)
    converter = ModelResponseConverter(response, converter=converter_key)
    converter.on_stream_end = _make_llm_end_hook(event)
    with patch("tenxgraph.runtime.publisher.publish.publish_event") as pub:
        if stream:
            async for _ in converter.stream({"thread_id": "t"}, node_name="AGENT"):
                pass
        else:
            await converter.invoke()
    assert pub.call_count == 1
    sent = pub.call_args.args[0]
    assert sent.event_type == EventType.END
    return sent.data


@pytest.mark.asyncio
async def test_google_non_stream_reads_usage_metadata():
    gt = pytest.importorskip("google.genai.types")
    response = gt.GenerateContentResponse(
        response_id="resp-g",
        model_version="gemini-2.5-flash",
        candidates=[
            gt.Candidate(
                content=gt.Content(role="model", parts=[gt.Part(text="hi")]),
                finish_reason=gt.FinishReason.STOP,
            )
        ],
        usage_metadata=gt.GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
            thoughts_token_count=3,
            cached_content_token_count=2,
            total_token_count=18,
        ),
    )
    data = await _end_event(response, "google")
    assert (data["input_tokens"], data["output_tokens"]) == (10, 5)
    assert (data["reasoning_tokens"], data["cache_read_tokens"]) == (3, 2)
    assert data["cache_creation_tokens"] == 0
    assert data["response_id"] == "resp-g"
    assert data["response_model"] == "gemini-2.5-flash"
    assert data["finish_reason"] == "STOP"
    assert data["output_response"] == "hi"


@pytest.mark.asyncio
async def test_openai_chat_non_stream():
    chat = pytest.importorskip("openai.types.chat")
    usage_types = pytest.importorskip("openai.types.completion_usage")
    response = chat.ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=0,
        model="gpt-4o-2024-08-06",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"a": 1}'},
                        }
                    ],
                },
            }
        ],
        usage=usage_types.CompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details={"cached_tokens": 4},
            completion_tokens_details={"reasoning_tokens": 2},
        ),
    )
    data = await _end_event(response, "openai")
    assert (data["input_tokens"], data["output_tokens"]) == (10, 5)
    assert (data["cache_read_tokens"], data["reasoning_tokens"]) == (4, 2)
    assert data["response_id"] == "chatcmpl-1"
    assert data["response_model"] == "gpt-4o-2024-08-06"
    assert data["finish_reason"] == "tool_calls"
    assert data["output_tool_calls"] == [{"id": "call_1", "name": "f", "arguments": '{"a": 1}'}]


@pytest.mark.asyncio
async def test_openai_responses_non_stream():
    rt = pytest.importorskip("openai.types.responses")
    response = rt.Response.model_validate(
        {
            "id": "resp_1",
            "created_at": 0,
            "model": "gpt-5",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens_details": {"reasoning_tokens": 2},
            },
        }
    )
    data = await _end_event(response, "openai_responses")
    assert (data["input_tokens"], data["output_tokens"]) == (10, 5)
    assert (data["cache_read_tokens"], data["reasoning_tokens"]) == (4, 2)
    assert data["response_id"] == "resp_1"
    assert data["response_model"] == "gpt-5"
    assert data["finish_reason"] == "completed"


@pytest.mark.asyncio
async def test_anthropic_non_stream_cache_tokens():
    at = pytest.importorskip("anthropic.types")
    response = at.Message(
        id="msg_1",
        model="claude-sonnet-4-5",
        role="assistant",
        type="message",
        content=[
            at.TextBlock(type="text", text="Hello "),
            at.ToolUseBlock(type="tool_use", id="tu_1", name="get_w", input={"city": "Paris"}),
        ],
        stop_reason="tool_use",
        usage=at.Usage(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=7,
            cache_creation_input_tokens=3,
        ),
    )
    data = await _end_event(response, "anthropic")
    assert (data["input_tokens"], data["output_tokens"]) == (10, 5)
    assert (data["cache_read_tokens"], data["cache_creation_tokens"]) == (7, 3)
    # Anthropic counts thinking inside output_tokens; there is no separate reasoning count.
    assert data["reasoning_tokens"] == 0
    assert data["response_id"] == "msg_1"
    assert data["response_model"] == "claude-sonnet-4-5"
    assert data["finish_reason"] == "tool_use"
    assert data["output_response"] == "Hello"
    assert data["output_tool_calls"] == [
        {"id": "tu_1", "name": "get_w", "arguments": '{"city": "Paris"}'}
    ]


@pytest.mark.asyncio
async def test_openai_chat_stream_usage_from_last_chunk():
    chat = pytest.importorskip("openai.types.chat")

    def chunk(**kw):
        base = {"id": "chatcmpl-s", "object": "chat.completion.chunk", "created": 0}
        return chat.ChatCompletionChunk.model_validate({**base, "model": "gpt-4o", **kw})

    chunks = [
        chunk(choices=[{"index": 0, "delta": {"role": "assistant", "content": "hi"}}]),
        chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
        chunk(
            choices=[],
            usage={
                "prompt_tokens": 8,
                "completion_tokens": 2,
                "total_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        ),
    ]

    async def stream():
        for c in chunks:
            yield c

    data = await _end_event(stream(), "openai", stream=True)
    assert (data["input_tokens"], data["output_tokens"]) == (8, 2)
    assert (data["cache_read_tokens"], data["reasoning_tokens"]) == (3, 1)
    assert data["response_model"] == "gpt-4o"
    assert data["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_conversion_error_ends_span_with_error():
    event = EventModel(event=Event.LLM_CALL, event_type=EventType.START)
    converter = ModelResponseConverter(object(), converter="openai")
    converter.on_stream_end = _make_llm_end_hook(event)
    with (
        patch("tenxgraph.runtime.publisher.publish.publish_event") as pub,
        pytest.raises(AttributeError),
    ):
        await converter.invoke()
    sent = pub.call_args.args[0]
    assert sent.event_type == EventType.ERROR
