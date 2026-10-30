"""OpenAI Agent defaults to the Responses API; structured output and reasoning routing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.agent_internal.openai import _REASONING_REJECTED_MODELS
from tenxgraph.core.state.message import Message
from tenxgraph.runtime.adapters.llm.openai_converter import OpenAIConverter
from tenxgraph.runtime.adapters.llm.openai_responses_converter import OpenAIResponsesConverter


MESSAGES = [{"role": "user", "content": "Hi"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
USAGE = SimpleNamespace(
    input_tokens=1,
    output_tokens=2,
    total_tokens=3,
    output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    input_tokens_details=SimpleNamespace(cached_tokens=0),
)


class Answer(BaseModel):
    answer: str


def _responses_obj(parsed=None):
    return SimpleNamespace(
        id="resp_1",
        model="gpt-6-luna",
        status="completed",
        created_at=1700000000,
        output=[],
        usage=USAGE,
        output_parsed=parsed,
        model_dump=MagicMock(return_value={"id": "resp_1"}),
    )


def _chat_obj(parsed=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant", content="Hello", tool_calls=None, parsed=parsed
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=1, completion_tokens=2, total_tokens=3, completion_tokens_details=None
        ),
        model="gpt-4o",
        id="chatcmpl-1",
        model_dump=MagicMock(return_value={"id": "chatcmpl-1"}),
    )


def _client():
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=_responses_obj())
    client.responses.parse = AsyncMock(return_value=_responses_obj(Answer(answer="42")))
    client.chat.completions.create = AsyncMock(return_value=_chat_obj())
    client.chat.completions.parse = AsyncMock(return_value=_chat_obj(Answer(answer="42")))
    return client


def _api_error(status, message, param=None):
    """A real OpenAI SDK status error, as the client raises it."""
    import httpx2
    import openai

    request = httpx2.Request("POST", "http://localhost:8000/v1/responses")
    response = httpx2.Response(status, request=request)
    cls = {
        400: openai.BadRequestError,
        404: openai.NotFoundError,
        429: openai.RateLimitError,
    }.get(status, openai.APIStatusError)
    body = {"message": message, "param": param, "type": "invalid_request_error"}
    return cls(message, response=response, body=body)


@pytest.fixture(autouse=True)
def _clear_reasoning_cache():
    _REASONING_REJECTED_MODELS.clear()
    yield
    _REASONING_REJECTED_MODELS.clear()


def _agent(client=None, model="gpt-6-luna", **kwargs):
    client = client or _client()
    with patch.object(Agent, "_create_client", return_value=client):
        agent = Agent(model=model, provider="openai", **kwargs)
    agent.client = client
    return agent


class TestDefaultStyle:
    def test_no_base_url_defaults_to_responses(self):
        assert _agent().api_style == "responses"

    def test_base_url_defaults_to_chat(self):
        assert _agent(base_url="http://localhost:11434/v1").api_style == "chat"

    def test_explicit_chat_wins_without_base_url(self):
        assert _agent(api_style="chat").api_style == "chat"

    def test_explicit_responses_wins_with_base_url(self):
        agent = _agent(api_style="responses", base_url="http://localhost:11434/v1")
        assert agent.api_style == "responses"


@pytest.mark.asyncio
class TestMaintainerCase:
    async def test_tools_and_default_reasoning_use_responses_with_reasoning(self):
        client = _client()
        agent = _agent(client)
        await agent._call_llm(MESSAGES, tools=TOOLS, stream=False)

        client.chat.completions.create.assert_not_called()
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["reasoning"] == {"effort": "minimal", "summary": "auto"}
        assert "reasoning_effort" not in kwargs
        assert kwargs["tools"][0]["name"] == "get_weather"


@pytest.mark.asyncio
class TestChatReasoning:
    async def test_default_reasoning_not_sent_on_chat(self):
        client = _client()
        agent = _agent(client, api_style="chat")
        await agent._call_llm(MESSAGES, tools=TOOLS, stream=False)
        assert "reasoning_effort" not in client.chat.completions.create.call_args.kwargs

    async def test_default_reasoning_not_sent_on_chat_with_base_url(self):
        client = _client()
        agent = _agent(client, base_url="http://localhost:8000/v1")
        await agent._call_llm(MESSAGES, tools=None, stream=False)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in kwargs
        assert "extra_body" not in kwargs

    async def test_explicit_reasoning_sent_on_chat(self):
        client = _client()
        agent = _agent(client, api_style="chat", reasoning_config={"effort": "high"})
        await agent._call_llm(MESSAGES, tools=TOOLS, stream=False)
        assert client.chat.completions.create.call_args.kwargs["reasoning_effort"] == "high"

    async def test_explicit_reasoning_extra_body_with_base_url(self):
        client = _client()
        agent = _agent(
            client, base_url="http://localhost:8000/v1", reasoning_config={"effort": "low"}
        )
        await agent._call_llm(MESSAGES, tools=None, stream=False)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["extra_body"]["reasoning"] == {"effort": "low"}

    async def test_explicit_true_counts_as_explicit(self):
        client = _client()
        agent = _agent(client, api_style="chat", reasoning_config=True)
        await agent._call_llm(MESSAGES, tools=None, stream=False)
        assert client.chat.completions.create.call_args.kwargs["reasoning_effort"] == "minimal"


@pytest.mark.asyncio
class TestStickyFallback:
    async def test_probe_once_then_chat(self, caplog):
        client = _client()
        client.responses.create = AsyncMock(side_effect=_api_error(404, "Not Found"))
        agent = _agent(client, api_style="responses", base_url="http://localhost:8000/v1")

        with caplog.at_level("WARNING", logger="tenxgraph.agent"):
            await agent._call_llm(MESSAGES, tools=None, stream=False)
            await agent._call_llm(MESSAGES, tools=None, stream=False)
            await agent._call_llm(MESSAGES, tools=None, stream=False)

        assert client.responses.create.await_count == 1
        assert client.chat.completions.create.await_count == 3
        warnings = [r for r in caplog.records if "Responses API not supported" in r.getMessage()]
        assert len(warnings) == 1

    async def test_sticky_state_shared_with_attempt_copies(self):
        client = _client()
        client.responses.create = AsyncMock(side_effect=_api_error(404, "Not Found"))
        agent = _agent(client, api_style="responses", base_url="http://localhost:8000/v1")
        bound = agent._bind_attempt(agent.model, "openai", client, agent.base_url)

        await bound._call_llm(MESSAGES, tools=None, stream=False)
        await agent._call_llm(MESSAGES, tools=None, stream=False)

        assert client.responses.create.await_count == 1

    async def test_no_base_url_error_propagates_and_is_not_sticky(self):
        client = _client()
        client.responses.create = AsyncMock(side_effect=Exception("boom"))
        agent = _agent(client)
        for _ in range(2):
            with pytest.raises(Exception, match="boom"):
                await agent._call_llm(MESSAGES, tools=None, stream=False)
        assert client.responses.create.await_count == 2
        client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
class TestFallbackOnlyOnUnsupported:
    """Only 404/405/501 switch a base_url to chat; transient errors raise and do not stick."""

    @pytest.mark.parametrize("status", [404, 405, 501])
    async def test_unsupported_status_switches_to_chat(self, status):
        client = _client()
        client.responses.create = AsyncMock(side_effect=_api_error(status, "unsupported"))
        agent = _agent(client, api_style="responses", base_url="http://localhost:8000/v1")

        await agent._call_llm(MESSAGES, tools=None, stream=False)

        assert agent.base_url in agent._responses_unsupported_urls
        client.chat.completions.create.assert_awaited_once()

    async def test_rate_limit_raises_and_does_not_switch(self):
        client = _client()
        client.responses.create = AsyncMock(
            side_effect=[_api_error(429, "Rate limit reached"), _responses_obj()]
        )
        agent = _agent(client, api_style="responses", base_url="http://localhost:8000/v1")

        with pytest.raises(Exception, match="Rate limit"):
            await agent._call_llm(MESSAGES, tools=None, stream=False)
        await agent._call_llm(MESSAGES, tools=None, stream=False)

        assert agent._responses_unsupported_urls == set()
        assert client.responses.create.await_count == 2
        client.chat.completions.create.assert_not_called()
        assert agent._effective_api_style == "responses"

    async def test_timeout_raises_and_does_not_switch(self):
        import httpx2
        import openai

        timeout = openai.APITimeoutError(request=httpx2.Request("POST", "http://x/v1/responses"))
        client = _client()
        client.responses.create = AsyncMock(side_effect=timeout)
        agent = _agent(client, api_style="responses", base_url="http://localhost:8000/v1")

        with pytest.raises(openai.APITimeoutError):
            await agent._call_llm(MESSAGES, tools=None, stream=False)

        assert agent._responses_unsupported_urls == set()
        client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
class TestReasoningSummaryDefault:
    async def test_explicit_effort_gets_auto_summary(self):
        client = _client()
        config = {"effort": "high"}
        agent = _agent(client, reasoning_config=config)
        await agent._call_llm(MESSAGES, tools=None, stream=False)

        assert client.responses.create.call_args.kwargs["reasoning"] == {
            "effort": "high",
            "summary": "auto",
        }
        assert config == {"effort": "high"}  # the user's dict is not mutated

    async def test_explicit_summary_is_kept(self):
        client = _client()
        agent = _agent(client, reasoning_config={"effort": "low", "summary": "concise"})
        await agent._call_llm(MESSAGES, tools=None, stream=False)
        assert client.responses.create.call_args.kwargs["reasoning"]["summary"] == "concise"

    async def test_reasoning_off_sends_no_reasoning(self):
        client = _client()
        agent = _agent(client, reasoning_config=None)
        await agent._call_llm(MESSAGES, tools=None, stream=False)
        assert "reasoning" not in client.responses.create.call_args.kwargs


@pytest.mark.asyncio
class TestDefaultReasoningRejected:
    """Non-reasoning models 400 on the default reasoning; retry once and remember the model."""

    async def test_default_retries_without_reasoning_and_remembers(self, caplog):
        client = _client()
        rejected = _api_error(
            400, "Unsupported parameter: 'reasoning.effort'", param="reasoning.effort"
        )
        client.responses.create = AsyncMock(
            side_effect=[rejected, _responses_obj(), _responses_obj()]
        )
        agent = _agent(client, model="gpt-4o")

        with caplog.at_level("WARNING", logger="tenxgraph.agent"):
            await agent._call_llm(MESSAGES, tools=None, stream=False)
            await agent._call_llm(MESSAGES, tools=None, stream=False)

        calls = client.responses.create.call_args_list
        assert len(calls) == 3
        assert "reasoning" in calls[0].kwargs
        assert "reasoning" not in calls[1].kwargs
        assert "reasoning" not in calls[2].kwargs  # remembered: no second failed probe
        assert "gpt-4o" in _REASONING_REJECTED_MODELS
        assert sum("rejected the default reasoning" in r.getMessage() for r in caplog.records) == 1

    async def test_true_counts_as_default(self):
        client = _client()
        rejected = _api_error(400, "reasoning is not supported", param="reasoning")
        client.responses.create = AsyncMock(side_effect=[rejected, _responses_obj()])
        agent = _agent(client, model="gpt-4o", reasoning_config=True)

        await agent._call_llm(MESSAGES, tools=None, stream=False)
        assert "reasoning" not in client.responses.create.call_args.kwargs

    async def test_explicit_dict_still_raises(self):
        import openai

        client = _client()
        rejected = _api_error(400, "Unsupported parameter: 'reasoning.effort'", "reasoning.effort")
        client.responses.create = AsyncMock(side_effect=rejected)
        agent = _agent(client, model="gpt-4o", reasoning_config={"effort": "high"})

        with pytest.raises(openai.BadRequestError):
            await agent._call_llm(MESSAGES, tools=None, stream=False)
        assert client.responses.create.await_count == 1
        assert _REASONING_REJECTED_MODELS == set()

    async def test_unrelated_400_still_raises(self):
        import openai

        client = _client()
        client.responses.create = AsyncMock(side_effect=_api_error(400, "Invalid input", "input"))
        agent = _agent(client, model="gpt-4o")

        with pytest.raises(openai.BadRequestError):
            await agent._call_llm(MESSAGES, tools=None, stream=False)
        assert client.responses.create.await_count == 1

    async def test_400_about_reasoning_items_in_input_still_raises(self):
        import openai

        client = _client()
        message = "Item of type 'reasoning' was provided without its required following item."
        rejected = _api_error(400, message, param="input")
        client.responses.create = AsyncMock(side_effect=rejected)
        agent = _agent(client, model="gpt-6-luna")

        with pytest.raises(openai.BadRequestError):
            await agent._call_llm(MESSAGES, tools=None, stream=False)
        assert _REASONING_REJECTED_MODELS == set()

    async def test_cache_is_per_model(self):
        _REASONING_REJECTED_MODELS.add("gpt-4o")
        client = _client()
        agent = _agent(client, model="gpt-6-luna")
        await agent._call_llm(MESSAGES, tools=None, stream=False)
        assert client.responses.create.call_args.kwargs["reasoning"]["effort"] == "minimal"

    async def test_chat_path_retries_without_reasoning_effort(self):
        client = _client()
        rejected = _api_error(400, "Unrecognized request argument: reasoning_effort")
        client.chat.completions.create = AsyncMock(side_effect=[rejected, _chat_obj()])
        agent = _agent(client, model="gpt-4o", api_style="chat", reasoning_config=True)

        await agent._call_llm(MESSAGES, tools=None, stream=False)

        calls = client.chat.completions.create.call_args_list
        assert calls[0].kwargs["reasoning_effort"] == "minimal"
        assert "reasoning_effort" not in calls[1].kwargs


@pytest.mark.asyncio
class TestStructuredOutput:
    async def test_responses_uses_parse_with_text_format(self):
        client = _client()
        agent = _agent(client, output_schema=Answer)
        result = await agent._call_llm(MESSAGES, tools=TOOLS, stream=True)

        client.responses.create.assert_not_called()
        client.chat.completions.parse.assert_not_called()
        kwargs = client.responses.parse.call_args.kwargs
        assert kwargs["text_format"] is Answer
        assert kwargs["stream"] is False
        assert kwargs["reasoning"] == {"effort": "minimal", "summary": "auto"}
        assert agent._effective_api_style == "responses"

        message = await OpenAIResponsesConverter().convert_response(result)
        assert message.parsed_content == Answer(answer="42")

    async def test_chat_uses_non_beta_parse(self):
        client = _client()
        client.beta = None  # any use of client.beta would raise AttributeError
        agent = _agent(client, api_style="chat", output_schema=Answer)
        result = await agent._call_llm(MESSAGES, tools=None, stream=False)

        client.chat.completions.parse.assert_awaited_once()
        kwargs = client.chat.completions.parse.call_args.kwargs
        assert kwargs["response_format"] is Answer
        assert kwargs["stream"] is False
        assert "reasoning_effort" not in kwargs

        message = await OpenAIConverter().convert_response(result)
        assert message.parsed_content == Answer(answer="42")

    async def test_chat_with_schema_and_explicit_reasoning(self):
        client = _client()
        agent = _agent(
            client, api_style="chat", output_schema=Answer, reasoning_config={"effort": "low"}
        )
        await agent._call_llm(MESSAGES, tools=None, stream=False)
        assert client.chat.completions.parse.call_args.kwargs["reasoning_effort"] == "low"

    async def test_responses_streaming_converter_handles_whole_response(self):
        parsed = Answer(answer="7")
        out = [
            m
            async for m in OpenAIResponsesConverter().convert_streaming_response(
                {}, "node", _responses_obj(parsed)
            )
        ]
        assert len(out) == 1
        assert isinstance(out[0], Message)
        assert out[0].parsed_content == parsed

    async def test_plain_response_has_no_parsed_content(self):
        message = await OpenAIResponsesConverter().convert_response(_responses_obj())
        assert message.parsed_content is None


@pytest.mark.asyncio
class TestMediaOutput:
    async def test_image_output_ignores_responses_default(self):
        client = _client()
        client.images.generate = AsyncMock(return_value=MagicMock())
        agent = _agent(client, model="gpt-image-1", output_type="image")
        await agent._call_llm(MESSAGES, tools=None, stream=False)
        client.images.generate.assert_awaited_once()
        client.responses.create.assert_not_called()
