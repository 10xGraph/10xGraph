"""Tests for tenxgraph.core.llm.caller.call_llm."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.llm.caller import (
    _call_google,
    _call_openai_chat,
    _call_openai_responses,
    call_llm,
)


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

# call_llm resolves provider *and* model together, so a recognised
# "provider/" prefix is stripped before the name reaches the SDK.
_RESOLVE = "tenxgraph.core.llm.caller.resolve_provider_and_model"
# call_llm builds its client with the Agent's rule (AgentProviderMixin._create_client).
_CREATE = "tenxgraph.core.graph.agent_internal.providers.create_llm_client"
_CALL_GOOGLE = "tenxgraph.core.llm.caller._call_google"
_CALL_RESP = "tenxgraph.core.llm.caller._call_openai_responses"
_CALL_CHAT = "tenxgraph.core.llm.caller._call_openai_chat"

_DUMMY = ("text", 10, 5, 0)


@pytest.mark.anyio
async def test_google_model_dispatches_to_google():
    with (
        patch(_RESOLVE, return_value=("google", "test-model")),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_GOOGLE, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        result = await call_llm("gemini-2.0-flash", "hello")

    mock.assert_called_once()
    assert result == _DUMMY


_CALL_ANTHROPIC = "tenxgraph.core.llm.caller._call_anthropic"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("kwargs", "expected_backend"),
    [
        ({"use_vertex_ai": True}, "vertex"),
        ({"use_vertex_ai": True, "anthropic_backend": "bedrock"}, "bedrock"),
        ({"anthropic_backend": "bedrock"}, "bedrock"),
        ({}, None),
    ],
)
async def test_claude_backend_follows_agent_rule(kwargs, expected_backend):
    with (
        patch(_CREATE, return_value=MagicMock()) as create,
        patch(_CALL_ANTHROPIC, new=AsyncMock(return_value=_DUMMY)),
    ):
        await call_llm("claude-opus-4-7", "hello", **kwargs)

    assert create.call_args.args[0] == "anthropic"
    assert create.call_args.kwargs["anthropic_backend"] == expected_backend


@pytest.mark.anyio
async def test_openai_default_dispatches_to_responses():
    with (
        patch(_RESOLVE, return_value=("openai", "test-model")),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_RESP, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        result = await call_llm("gpt-4o-mini", "hello")

    mock.assert_called_once()
    assert result == _DUMMY


@pytest.mark.anyio
async def test_openai_chat_style_dispatches_to_chat():
    with (
        patch(_RESOLVE, return_value=("openai", "test-model")),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_CHAT, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        result = await call_llm("gpt-4o-mini", "hello", api_style="chat")

    mock.assert_called_once()
    assert result == _DUMMY


@pytest.mark.anyio
async def test_openai_responses_style_explicit():
    """Explicitly passing api_style='responses' still hits the Responses path."""
    with (
        patch(_RESOLVE, return_value=("openai", "test-model")),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_RESP, new=AsyncMock(return_value=_DUMMY)) as mock_resp,
        patch(_CALL_CHAT, new=AsyncMock(return_value=_DUMMY)) as mock_chat,
    ):
        await call_llm("gpt-4o-mini", "hello", api_style="responses")

    mock_resp.assert_called_once()
    mock_chat.assert_not_called()


@pytest.mark.anyio
async def test_api_style_irrelevant_for_google():
    """api_style has no effect when the provider is Google."""
    with (
        patch(_RESOLVE, return_value=("google", "test-model")),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_GOOGLE, new=AsyncMock(return_value=_DUMMY)) as mock_google,
        patch(_CALL_CHAT, new=AsyncMock(return_value=_DUMMY)) as mock_chat,
    ):
        await call_llm("gemini-2.0-flash", "hello", api_style="chat")

    mock_google.assert_called_once()
    mock_chat.assert_not_called()


# ---------------------------------------------------------------------------
# Parameters forwarded correctly
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_system_prompt_forwarded_to_responses():
    with (
        patch(_RESOLVE, return_value=("openai", "test-model")),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_RESP, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        await call_llm("gpt-4o-mini", "hi", system_prompt="Be brief.", json_mode=True)

    _, kwargs = mock.call_args
    assert kwargs["system_prompt"] == "Be brief."
    assert kwargs["json_mode"] is True


@pytest.mark.anyio
async def test_system_prompt_forwarded_to_chat():
    with (
        patch(_RESOLVE, return_value=("openai", "test-model")),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_CHAT, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        await call_llm("gpt-4o-mini", "hi", system_prompt="Be brief.", api_style="chat")

    _, kwargs = mock.call_args
    assert kwargs["system_prompt"] == "Be brief."


# ---------------------------------------------------------------------------
# Direct private call implementations
# ---------------------------------------------------------------------------


def _google_response(text: str, **usage: int) -> object:
    from google.genai import types

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="thinking", thought=True), types.Part(text=text)],
                )
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(**usage),
    )


@pytest.mark.anyio
async def test_call_google_implementation():
    mock_client = MagicMock()
    response = _google_response(
        "  hello google  ",
        prompt_token_count=10,
        candidates_token_count=5,
        cached_content_token_count=3,
    )
    mock_client.aio.models.generate_content = AsyncMock(return_value=response)

    # 1. With system prompt, temperature, json mode
    res = await _call_google(
        mock_client,
        "gemini-2.0-flash",
        "user prompt",
        system_prompt="sys prompt",
        max_tokens=100,
        temperature=0.5,
        json_mode=True,
    )

    assert res == ("hello google", 10, 5, 3)
    mock_client.aio.models.generate_content.assert_called_once()
    _, kwargs = mock_client.aio.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.0-flash"
    [content] = kwargs["contents"]
    assert content.role == "user" and content.parts[0].text == "user prompt"
    config = kwargs["config"]
    assert config.max_output_tokens == 100
    assert config.temperature == 0.5
    assert config.response_mime_type == "application/json"
    assert config.system_instruction == "sys prompt"
    assert config.thinking_config is None

    # 2. With cached content: the system prompt is inside the cache and is not resent
    mock_client.aio.models.generate_content.reset_mock()
    await _call_google(
        mock_client,
        "gemini-2.0-flash",
        "user prompt",
        system_prompt="sys prompt",
        max_tokens=100,
        temperature=0.5,
        json_mode=False,
        cached_content="cachedContents/abc123",
    )

    _, kwargs = mock_client.aio.models.generate_content.call_args
    config = kwargs["config"]
    assert config.cached_content == "cachedContents/abc123"
    assert getattr(config, "system_instruction", None) is None
    assert len(kwargs["contents"]) == 1


def _responses_response(text: str) -> object:
    from openai.types.responses import Response

    return Response.model_validate(
        {
            "id": "resp_1",
            "created_at": 0,
            "model": "gpt-4o-mini",
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 15,
                "total_tokens": 35,
                "input_tokens_details": {"cached_tokens": 7},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


@pytest.mark.anyio
async def test_call_openai_responses_implementation():
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(
        return_value=_responses_response("  hello responses  ")
    )

    res = await _call_openai_responses(
        mock_client,
        "gpt-4o-mini",
        "user prompt",
        system_prompt="sys prompt",
        max_tokens=200,
        temperature=0.7,
        json_mode=True,
        extra_param="extra_val",
    )

    assert res == ("hello responses", 20, 15, 7)

    mock_client.responses.create.assert_called_once()
    _, kwargs = mock_client.responses.create.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["input"] == [{"role": "user", "content": "user prompt"}]
    assert kwargs["max_output_tokens"] == 200
    assert kwargs["temperature"] == 0.7
    assert kwargs["instructions"] == "sys prompt"
    assert kwargs["text"] == {"format": {"type": "json_object"}}
    assert kwargs["extra_param"] == "extra_val"
    assert "reasoning" not in kwargs


@pytest.mark.anyio
async def test_call_openai_chat_implementation():
    from openai.types.chat import ChatCompletion

    response = ChatCompletion.model_validate(
        {
            "id": "c1",
            "created": 0,
            "model": "gpt-4o-mini",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "  hello chat  "},
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 25,
                "total_tokens": 55,
                "prompt_tokens_details": {"cached_tokens": 12},
            },
        }
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    res = await _call_openai_chat(
        mock_client,
        "gpt-4o-mini",
        "user prompt",
        system_prompt="sys prompt",
        max_tokens=300,
        temperature=0.8,
        json_mode=True,
        another_param="another_val",
    )

    assert res == ("hello chat", 30, 25, 12)

    mock_client.chat.completions.create.assert_called_once()
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert kwargs["max_tokens"] == 300
    assert kwargs["temperature"] == 0.8
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["another_param"] == "another_val"


@pytest.mark.anyio
async def test_client_constructor_keys_go_to_the_client():
    """timeout reaches the client, as on the Agent, and never the request."""
    with (
        patch(_CREATE, return_value=MagicMock()) as create,
        patch(_CALL_RESP, new=AsyncMock(return_value=_DUMMY)),
    ):
        await call_llm("gpt-4o-mini", "hello", timeout=12.0)

    assert create.call_args.kwargs["timeout"] == 12.0


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

_CALL_ANTHROPIC = "tenxgraph.core.llm.caller._call_anthropic"


@pytest.mark.anyio
async def test_claude_model_dispatches_to_anthropic():
    with (
        patch(_RESOLVE, return_value=("anthropic", "claude-opus-5")),
        patch(_CREATE, return_value=MagicMock()),
        patch(_CALL_ANTHROPIC, new=AsyncMock(return_value=_DUMMY)) as mock,
    ):
        result = await call_llm("claude-opus-5", "hello")

    mock.assert_called_once()
    assert result == _DUMMY


@pytest.mark.anyio
async def test_anthropic_drops_temperature_for_rejecting_models():
    """call_llm defaults temperature=0.3; current Claude models 400 on it."""
    from tenxgraph.core.llm.caller import _call_anthropic

    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=3, output_tokens=1, cache_read_input_tokens=0),
        )
    )

    await _call_anthropic(
        client,
        "claude-opus-5",
        "hello",
        system_prompt=None,
        max_tokens=100,
        temperature=0.3,
        json_mode=False,
    )
    assert "temperature" not in client.messages.create.call_args.kwargs


@pytest.mark.anyio
async def test_anthropic_keeps_temperature_for_older_models():
    from tenxgraph.core.llm.caller import _call_anthropic

    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(content=[], stop_reason="end_turn", usage=None)
    )

    await _call_anthropic(
        client,
        "claude-haiku-4-5",
        "hello",
        system_prompt=None,
        max_tokens=100,
        temperature=0.3,
        json_mode=False,
    )
    assert client.messages.create.call_args.kwargs["temperature"] == 0.3


@pytest.mark.anyio
async def test_anthropic_returns_text_and_usage():
    from tenxgraph.core.llm.caller import _call_anthropic

    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="ignored"),
                SimpleNamespace(type="text", text="  answer  "),
            ],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=42, output_tokens=7, cache_read_input_tokens=30),
        )
    )

    text, inp, out, cache = await _call_anthropic(
        client,
        "claude-opus-5",
        "hello",
        system_prompt="be terse",
        max_tokens=100,
        temperature=0.3,
        json_mode=False,
    )
    assert text == "answer"
    assert (inp, out, cache) == (42, 7, 30)
    sent = client.messages.create.call_args.kwargs
    assert sent["system"] == [{"type": "text", "text": "be terse"}]
    # call_llm sends no reasoning and, unless asked, no prompt caching.
    assert "thinking" not in sent and "cache_control" not in sent
