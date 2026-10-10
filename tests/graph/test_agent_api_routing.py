"""Tests for Agent API style routing and responses→chat fallback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.graph.agent import Agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client():
    """Create a minimal mock OpenAI-style client."""
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=SimpleNamespace(
        id="resp_1", model="o4-mini", status="completed", created_at=1700000000,
        output=[], usage=SimpleNamespace(
            input_tokens=10, output_tokens=20, total_tokens=30,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
        model_dump=MagicMock(return_value={"id": "resp_1"}),
    ))
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(role="assistant", content="Hello",
                                    tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20,
                               total_tokens=30, completion_tokens_details=None),
        model="gpt-4o", id="chatcmpl-1",
        model_dump=MagicMock(return_value={"id": "chatcmpl-1"}),
    ))
    return client


def _create_agent(api_style="chat", reasoning_config=None, provider="openai",
                  model="gpt-4o", base_url=None, client=None):
    mock_client = client or _mock_client()
    with patch.object(Agent, "_create_client", return_value=mock_client):
        agent = Agent(model=model, provider=provider, api_style=api_style,
                      reasoning_config=reasoning_config, base_url=base_url)
    agent.client = mock_client
    return agent


# ---------------------------------------------------------------------------
# Tests: Agent construction
# ---------------------------------------------------------------------------


class TestAgentAPIStyle:
    def test_default_is_chat(self):
        agent = _create_agent(api_style="chat")
        assert agent.api_style == "chat"

    def test_responses_stored(self):
        agent = _create_agent(api_style="responses", model="o4-mini")
        assert agent.api_style == "responses"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid api_style"):
            with patch.object(Agent, "_create_client", return_value=MagicMock()):
                Agent(model="gpt-4o", provider="openai", api_style="invalid")


# ---------------------------------------------------------------------------
# Tests: _call_llm routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCallLLMRouting:
    async def test_chat_routes_to_completions(self):
        client = _mock_client()
        agent = _create_agent(api_style="chat", client=client)
        await agent._call_llm([{"role": "user", "content": "Hi"}],
                              tools=None, stream=False)
        client.chat.completions.create.assert_called_once()
        client.responses.create.assert_not_called()

    async def test_responses_routes_to_responses_api(self):
        client = _mock_client()
        agent = _create_agent(api_style="responses", model="o4-mini",
                              client=client)
        await agent._call_llm([{"role": "user", "content": "Hi"}],
                              tools=None, stream=False)
        client.responses.create.assert_called_once()
        client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Responses → Chat fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResponsesFallback:
    async def test_fallback_triggered_with_base_url(self):
        """base_url set + responses.create fails → falls back to chat."""
        client = _mock_client()
        client.responses.create = AsyncMock(
            side_effect=Exception("404: Not Found"))
        agent = _create_agent(api_style="responses",
                              base_url="http://localhost:11434/v1",
                              client=client)

        await agent._call_llm([{"role": "user", "content": "Hi"}],
                              tools=None, stream=False)

        client.chat.completions.create.assert_called_once()
        assert agent._effective_api_style == "chat"

    async def test_no_fallback_without_base_url(self):
        """Without base_url, failure propagates."""
        client = _mock_client()
        client.responses.create = AsyncMock(
            side_effect=Exception("Server error"))
        agent = _create_agent(api_style="responses", base_url=None,
                              client=client)

        with pytest.raises(Exception, match="Server error"):
            await agent._call_llm([{"role": "user", "content": "Hi"}],
                                  tools=None, stream=False)
        client.chat.completions.create.assert_not_called()

    async def test_effective_api_style_set_after_fallback(self):
        """_effective_api_style becomes 'chat' after fallback."""
        client = _mock_client()
        client.responses.create = AsyncMock(
            side_effect=Exception("Not supported"))
        agent = _create_agent(api_style="responses",
                              base_url="http://localhost:8000/v1",
                              client=client)

        await agent._call_llm([{"role": "user", "content": "Hi"}],
                              tools=None, stream=False)
        assert agent._effective_api_style == "chat"


# ---------------------------------------------------------------------------
# Tests: Converter key selection
# ---------------------------------------------------------------------------


class TestConverterKeySelection:
    def test_after_fallback_key_is_openai(self):
        """After fallback, effective style is 'chat' → converter key 'openai'."""
        agent = _create_agent(api_style="responses",
                              base_url="http://localhost:8000/v1")
        # Simulate post-fallback state
        agent._effective_api_style = "chat"
        effective = getattr(agent, "_effective_api_style", agent.api_style)
        key = "openai_responses" if (agent.provider == "openai"
                                     and effective == "responses") else agent.provider
        assert key == "openai"

    def test_without_fallback_key_is_openai_responses(self):
        """Without fallback, effective style stays 'responses'."""
        agent = _create_agent(api_style="responses",
                              base_url="http://localhost:8000/v1")
        agent._effective_api_style = "responses"
        effective = getattr(agent, "_effective_api_style", agent.api_style)
        key = "openai_responses" if (agent.provider == "openai"
                                     and effective == "responses") else agent.provider
        assert key == "openai_responses"
