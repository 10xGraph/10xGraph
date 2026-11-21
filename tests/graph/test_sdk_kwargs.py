"""Kwargs are forwarded to an SDK call only when the SDK defines them."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, create_autospec

import anthropic
import openai
import pytest

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.agent_internal import sdk_kwargs
from tenxgraph.core.graph.agent_internal.sdk_kwargs import accepted_params, filter_kwargs


@pytest.fixture(autouse=True)
def _reset_warned():
    sdk_kwargs._warned.clear()
    yield
    sdk_kwargs._warned.clear()


def _autospec(method):
    """AsyncMock with the real SDK method's signature."""
    mock = create_autospec(method, instance=False)
    mock.side_effect = None
    return mock


def _openai_agent(**kwargs) -> Agent:
    agent = Agent(model="gpt-4o", provider="openai", api_key="x", reasoning_config=None, **kwargs)
    agent.client = openai.AsyncOpenAI(api_key="x")
    return agent


class TestHelper:
    def test_real_sdk_signature_has_no_var_keyword(self):
        client = openai.AsyncOpenAI(api_key="x")
        params = accepted_params(client.responses.create)
        assert "prompt_cache_key" in params
        assert "seed" not in params
        assert "self" not in params

    def test_var_keyword_or_mock_is_not_filtered(self):
        def takes_kwargs(a, **kw): ...

        assert accepted_params(takes_kwargs) is None
        assert accepted_params(MagicMock()) is None
        assert filter_kwargs({"x": 1}, None, "p") == {"x": 1}

    def test_filters_and_warns_once(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tenxgraph.agent"):
            assert filter_kwargs({"a": 1, "b": 2}, {"a"}, "openai") == {"a": 1}
            filter_kwargs({"b": 2}, {"a"}, "openai")
        warnings = [r for r in caplog.records if "'b'" in r.getMessage()]
        assert len(warnings) == 1
        assert "openai" in warnings[0].getMessage()


@pytest.mark.asyncio
class TestOpenAI:
    async def test_responses_drops_foreign_key_keeps_real_one(self, caplog):
        agent = _openai_agent(api_style="responses", prompt_cache_key="k", seed=1)
        agent.client.responses.create = _autospec(agent.client.responses.create)
        agent.client.responses.create.side_effect = AsyncMock(return_value=MagicMock())
        with caplog.at_level(logging.WARNING, logger="tenxgraph.agent"):
            await agent._call_openai_responses([{"role": "user", "content": "hi"}])
        sent = agent.client.responses.create.call_args.kwargs
        assert sent["prompt_cache_key"] == "k"
        assert "seed" not in sent
        assert any("'seed'" in r.getMessage() for r in caplog.records)

    async def test_chat_drops_foreign_key(self):
        agent = _openai_agent(seed=3, prompt_cache_key="k", not_an_arg=1)
        create = agent.client.chat.completions.create
        agent.client.chat.completions.create = _autospec(create)
        agent.client.chat.completions.create.side_effect = AsyncMock(return_value=MagicMock())
        await agent._call_openai([{"role": "user", "content": "hi"}])
        sent = agent.client.chat.completions.create.call_args.kwargs
        assert sent["seed"] == 3
        assert "not_an_arg" not in sent


def _google_agent(**kwargs) -> Agent:
    return Agent(
        model="gemini-2.0-flash",
        provider="google",
        api_key="x",
        reasoning_config=None,
        **kwargs,
    )


class TestGoogle:
    def test_config_fields_forwarded_foreign_dropped(self, caplog):
        agent = _google_agent()
        call_kwargs = {
            "seed": 7,
            "top_k": 5,
            "temperature": 0.2,
            "prompt_cache_key": "k",
            "timeout": 3,
        }
        with caplog.at_level(logging.WARNING, logger="tenxgraph.agent"):
            config = agent._build_google_config("sys", None, call_kwargs)
        assert (config.seed, config.top_k, config.temperature) == (7, 5, 0.2)
        assert config.system_instruction == "sys"
        assert any("'prompt_cache_key'" in r.getMessage() for r in caplog.records)
        assert not any("'timeout'" in r.getMessage() for r in caplog.records)

    def test_framework_value_wins(self):
        agent = _google_agent()
        config = agent._build_google_config("framework", None, {"system_instruction": "mine"})
        assert config.system_instruction == "framework"

    def test_fallback_keys_do_not_cross_providers(self):
        google_config = _google_agent()._build_google_config(
            None, None, {"prompt_cache_key": "k", "seed": 1}
        )
        assert google_config.seed == 1
        assert "prompt_cache_key" not in google_config.model_dump(exclude_none=True)
        assert "seed" not in accepted_params(openai.AsyncOpenAI(api_key="x").responses.create)


def _anthropic_agent(**llm_kwargs) -> Agent:
    agent = Agent(
        model="claude-haiku-4-5",
        provider="anthropic",
        api_key="x",
        reasoning_config=None,
        **llm_kwargs,
    )
    agent.client = anthropic.AsyncAnthropic(api_key="x")
    messages = agent.client.messages
    messages.create = _autospec(messages.create)
    messages.create.side_effect = AsyncMock(return_value=MagicMock())
    messages.count_tokens = _autospec(messages.count_tokens)
    messages.count_tokens.side_effect = AsyncMock(return_value=MagicMock(input_tokens=5))
    return agent


@pytest.mark.asyncio
class TestAnthropic:
    async def test_foreign_key_dropped_with_one_warning_real_key_kept(self, caplog):
        agent = _anthropic_agent(prompt_cache_key="k", service_tier="auto")
        with caplog.at_level(logging.WARNING, logger="tenxgraph.agent"):
            await agent._call_anthropic([{"role": "user", "content": "hi"}])
            await agent._call_anthropic([{"role": "user", "content": "hi"}])
        sent = agent.client.messages.create.call_args.kwargs
        assert sent["service_tier"] == "auto"
        assert "prompt_cache_key" not in sent
        warnings = [r for r in caplog.records if "'prompt_cache_key'" in r.getMessage()]
        assert len(warnings) == 1

    async def test_count_tokens_gets_no_generation_params_and_no_warning(self, caplog):
        agent = _anthropic_agent(max_tokens=100)
        with caplog.at_level(logging.WARNING, logger="tenxgraph.agent"):
            assert await agent._count_tokens_anthropic([{"role": "user", "content": "hi"}]) == 5
        sent = agent.client.messages.count_tokens.call_args.kwargs
        assert "max_tokens" not in sent
        assert "stream" not in sent
        assert not [r for r in caplog.records if "SDK call" in r.getMessage()]

    async def test_real_methods_have_no_var_keyword(self):
        client = anthropic.AsyncAnthropic(api_key="x")
        for name in ("create", "stream", "count_tokens"):
            assert accepted_params(getattr(client.messages, name)) is not None


@pytest.mark.asyncio
class TestFrameworkKwargs:
    async def test_stream_idle_timeout_is_silent_on_every_provider(self, caplog):
        openai_agent = _openai_agent(stream_idle_timeout=5, seed=1)
        openai_agent.client.chat.completions.create = _autospec(
            openai_agent.client.chat.completions.create
        )
        openai_agent.client.chat.completions.create.side_effect = AsyncMock(
            return_value=MagicMock()
        )
        anthropic_agent = _anthropic_agent(stream_idle_timeout=5, anthropic_cache=False)
        google_agent = _google_agent(stream_idle_timeout=5)

        with caplog.at_level(logging.WARNING, logger="tenxgraph.agent"):
            await openai_agent._call_openai([{"role": "user", "content": "hi"}])
            await anthropic_agent._call_anthropic([{"role": "user", "content": "hi"}])
            google_agent._build_google_config(
                None, None, {"stream_idle_timeout": 5, "anthropic_cache": False}
            )

        sent = openai_agent.client.chat.completions.create.call_args.kwargs
        assert "stream_idle_timeout" not in sent
        assert "stream_idle_timeout" not in anthropic_agent.client.messages.create.call_args.kwargs
        assert not [r for r in caplog.records if "SDK call" in r.getMessage()]
