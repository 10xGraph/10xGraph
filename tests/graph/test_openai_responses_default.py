"""OpenAI Agent defaults to the Responses API; structured output and reasoning routing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from tenxgraph.core.graph.agent import Agent
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
        assert kwargs["reasoning"] == {"effort": "medium"}
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
        assert client.chat.completions.create.call_args.kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
class TestStickyFallback:
    async def test_probe_once_then_chat(self, caplog):
        client = _client()
        client.responses.create = AsyncMock(side_effect=Exception("404 Not Found"))
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
        client.responses.create = AsyncMock(side_effect=Exception("404"))
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
        assert kwargs["reasoning"] == {"effort": "medium"}
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
