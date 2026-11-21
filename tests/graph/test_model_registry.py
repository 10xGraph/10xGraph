"""Tests for ``Agent(model_registry=...)``: named models picked per run from the run config."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.exceptions import ModelNotFoundError
from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.agent_internal.constants import RetryConfig
from tenxgraph.core.state import AgentState


EXEC = "tenxgraph.core.graph.agent_internal.execution"

REGISTRY = {
    "openai": {"model": "gpt-5-mini", "api_key": "sk-openai"},
    "gemini": {"model": "gemini-2.5-flash", "safety_settings": [{"category": "x"}]},
    "vllm": {
        "provider": "openai",
        "model": "deepseek-r1",
        "base_url": "http://gpu1:8000/v1",
        "api_key": "vllm-key",
    },
    "claude": {"model": "claude-sonnet-5", "anthropic_backend": "vertex"},
}


class FakeClient:
    """A client built by the patched ``_create_client``; remembers what built it."""

    def __init__(self, provider: str, base_url: str | None, llm_kwargs: dict[str, Any]):
        self.provider = provider
        self.base_url = base_url
        self.api_key = llm_kwargs.get("api_key")
        self.anthropic_backend = llm_kwargs.get("anthropic_backend")


@pytest.fixture
def created() -> Any:
    """Patch client creation; yields the list of clients built."""
    clients: list[FakeClient] = []

    def fake_create(self, provider, base_url=None, use_vertex_ai=False):
        client = FakeClient(provider, base_url, self.llm_kwargs)
        clients.append(client)
        return client

    with patch.object(Agent, "_create_client", fake_create):
        yield clients


@pytest.fixture
def calls() -> Any:
    """Patch ``Agent._call_llm``; yields one record per LLM call with the bound settings."""
    records: list[dict[str, Any]] = []

    async def fake_call(self, messages, tools=None, stream=False, **kwargs):
        records.append(
            {
                "model": self.model,
                "provider": self.provider,
                "client": self.client,
                "base_url": self.base_url,
                "api_style": self.api_style,
                "llm_kwargs": dict(self.llm_kwargs),
                "reasoning": self.reasoning_config,
            }
        )
        await asyncio.sleep(0.01)  # let concurrent runs interleave
        if getattr(self, "_fail_models", None) and self.model in self._fail_models:
            raise ValueError(f"{self.model} failed")
        return SimpleNamespace(model=self.model)

    with patch.object(Agent, "_call_llm", fake_call):
        yield records


def _agent(**kwargs: Any) -> Agent:
    kwargs.setdefault("model", "openai")
    kwargs.setdefault("model_registry", REGISTRY)
    kwargs.setdefault("retry_config", None)
    return Agent(**kwargs)


async def _run(agent: Agent, config: dict[str, Any]) -> MagicMock:
    """Run ``agent.execute``; returns the ModelResponseConverter mock it built."""
    config = {"_node_name": "MAIN", **config}
    with (
        patch(f"{EXEC}.convert_messages", return_value=[{"role": "user", "content": "hi"}]),
        patch("tenxgraph.runtime.publisher.publish.publish_event"),
        patch(f"{EXEC}.ModelResponseConverter") as converter,
    ):
        await agent.execute(AgentState(), config)
    return converter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_entry_sets_agent_attributes(self, created):
        agent = _agent()
        assert (agent.model, agent.provider, agent.default_model_name) == (
            "gpt-5-mini",
            "openai",
            "openai",
        )
        assert agent.client is created[0]
        assert created[0].api_key == "sk-openai"
        # Only the default client is built at construction.
        assert len(created) == 1

    def test_entry_provider_resolved_from_model(self, created):
        agent = _agent()
        entries = agent._model_entries
        assert entries["gemini"].provider == "google"
        assert entries["claude"].provider == "anthropic"
        assert entries["vllm"].api_style == "chat"  # base_url default
        assert entries["openai"].api_style == "responses"

    def test_model_not_in_registry_refused(self, created):
        with pytest.raises(ValueError, match="must name a model_registry entry"):
            _agent(model="gpt-4o")

    @pytest.mark.parametrize(
        "key",
        ["provider", "base_url", "api_key", "api_style", "use_vertex_ai", "anthropic_backend"],
    )
    def test_conflicting_agent_kwargs_refused(self, created, key):
        with pytest.raises(ValueError, match="cannot be set on the Agent"):
            _agent(**{key: "x"})

    @pytest.mark.parametrize(
        "registry",
        [{}, {"a": "gpt-4o"}, {"a": {"provider": "openai"}}, {"a": {"model": ""}}, "x"],
    )
    def test_invalid_registry_refused(self, created, registry):
        with pytest.raises(ValueError):
            _agent(model="a", model_registry=registry)

    def test_entry_output_type_validated(self, created):
        with pytest.raises(ValueError, match="Anthropic provider doesn't support"):
            _agent(output_type="image")


# ---------------------------------------------------------------------------
# Picking an entry per run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSelection:
    async def test_default_entry_used_without_config_key(self, created, calls):
        agent = _agent()
        await _run(agent, {})
        assert calls[0]["model"] == "gpt-5-mini"
        assert calls[0]["client"] is agent.client

    async def test_config_picks_entry(self, created, calls):
        agent = _agent()
        await _run(agent, {"model": "vllm"})
        call = calls[0]
        assert (call["model"], call["provider"], call["base_url"], call["api_style"]) == (
            "deepseek-r1",
            "openai",
            "http://gpu1:8000/v1",
            "chat",
        )
        # The shared Agent is unchanged.
        assert (agent.model, agent.base_url) == ("gpt-5-mini", None)

    async def test_entry_gets_own_client_built_once(self, created, calls):
        agent = _agent()
        for _ in range(3):
            await _run(agent, {"model": "vllm"})
            await _run(agent, {"model": "claude"})
        # default + one per used entry
        assert len(created) == 3
        vllm, claude = created[1], created[2]
        assert (vllm.provider, vllm.base_url, vllm.api_key) == (
            "openai",
            "http://gpu1:8000/v1",
            "vllm-key",
        )
        # A cross-provider entry never sees another entry's key.
        assert (claude.provider, claude.api_key, claude.anthropic_backend) == (
            "anthropic",
            None,
            "vertex",
        )
        assert {id(c["client"]) for c in calls if c["model"] == "deepseek-r1"} == {id(vllm)}

    async def test_unknown_name_raises(self, created, calls):
        agent = _agent()
        with pytest.raises(ModelNotFoundError) as info:
            await _run(agent, {"model": "gpt-4o"})
        message = str(info.value)
        assert "model 'gpt-4o' is not registered on agent node 'MAIN'" in message
        assert "registered: openai, gemini, vllm, claude" in message
        assert isinstance(info.value, ValueError)
        assert calls == []

    async def test_non_string_name_raises(self, created, calls):
        agent = _agent()
        with pytest.raises(ModelNotFoundError):
            await _run(agent, {"model": 3})

    async def test_agent_without_registry_ignores_config_model(self, created, calls):
        agent = Agent(model="gpt-4o", provider="openai", retry_config=None)
        await _run(agent, {"model": "something-else"})
        assert calls[0]["model"] == "gpt-4o"

    async def test_model_config_key(self, created, calls):
        agent = _agent(model_config_key="worker_model")
        await _run(agent, {"model": "nope", "worker_model": "gemini"})
        assert calls[0]["model"] == "gemini-2.5-flash"

    async def test_converter_and_event_follow_entry(self, created, calls):
        agent = _agent()
        with (
            patch("tenxgraph.runtime.publisher.publish.publish_event") as publish,
            patch(f"{EXEC}.convert_messages", return_value=[{"role": "user", "content": "hi"}]),
            patch(f"{EXEC}.ModelResponseConverter") as converter,
        ):
            await agent.execute(AgentState(), {"_node_name": "MAIN", "model": "gemini"})
        assert converter.call_args.kwargs["converter"] == "google"
        event = publish.call_args_list[0].args[0]
        assert (event.data["model"], event.data["provider"]) == ("gemini-2.5-flash", "google")

    async def test_concurrent_runs_on_different_entries(self, created, calls):
        agent = _agent()
        names = ["openai", "gemini", "vllm", "claude"] * 3
        seen: list[tuple[str, str]] = []

        def converter(response, converter):
            seen.append((response.model, converter))
            return MagicMock()

        with (
            patch(f"{EXEC}.convert_messages", return_value=[{"role": "user", "content": "hi"}]),
            patch("tenxgraph.runtime.publisher.publish.publish_event"),
            patch(f"{EXEC}.ModelResponseConverter", side_effect=converter),
        ):
            await asyncio.gather(
                *(agent.execute(AgentState(), {"_node_name": "MAIN", "model": n}) for n in names)
            )
        expected = {
            "gpt-5-mini": ("openai", "openai_responses"),
            "gemini-2.5-flash": ("google", "google"),
            "deepseek-r1": ("openai", "openai"),
            "claude-sonnet-5": ("anthropic", "anthropic"),
        }
        assert sorted(m for m, _ in seen) == sorted(m for m in expected for _ in range(3))
        for model, key in seen:
            assert key == expected[model][1]
        assert len(calls) == len(names)
        for call in calls:
            assert call["provider"] == expected[call["model"]][0]
            assert call["client"].provider == call["provider"]
        assert agent.model == "gpt-5-mini"


# ---------------------------------------------------------------------------
# Settings per entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEntrySettings:
    async def test_entry_only_key_reaches_only_that_entry(self, created, calls):
        agent = _agent()
        await _run(agent, {"model": "gemini"})
        await _run(agent, {"model": "openai"})
        gemini, openai_call = calls
        assert gemini["llm_kwargs"]["safety_settings"] == [{"category": "x"}]
        assert "safety_settings" not in openai_call["llm_kwargs"]
        assert "safety_settings" not in agent.llm_kwargs

    async def test_entry_overrides_agent_default(self, created, calls):
        registry = {
            "a": {"model": "gpt-4o", "temperature": 0.9, "reasoning_config": {"effort": "high"}},
            "b": {"model": "gpt-4o-mini"},
        }
        agent = _agent(model="a", model_registry=registry, temperature=0.1, max_tokens=50)
        await _run(agent, {})
        await _run(agent, {"model": "b"})
        a, b = calls
        assert (a["llm_kwargs"]["temperature"], a["llm_kwargs"]["max_tokens"]) == (0.9, 50)
        assert a["reasoning"] == {"effort": "high"}
        assert (b["llm_kwargs"]["temperature"], b["llm_kwargs"]["max_tokens"]) == (0.1, 50)
        assert b["reasoning"] == {"effort": "minimal"}  # the Agent default, per provider

    async def test_agent_default_filtered_per_entry_sdk(self, created):
        """``user`` is an OpenAI chat parameter; Google's config has no such field."""
        sent: dict[str, Any] = {}

        async def chat_create(*, model, messages, stream=False, user=None, temperature=None):
            sent["openai"] = {"model": model, "user": user, "temperature": temperature}
            return SimpleNamespace(model=model)

        registry = {
            "chat": {"model": "gpt-4o", "api_style": "chat"},
            "gemini": {"model": "gemini-2.5-flash"},
        }
        agent = _agent(model="chat", model_registry=registry, user="u1", temperature=0.2)
        agent.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create))
        )
        await agent._call_llm_with_retry([{"role": "user", "content": "hi"}])
        assert sent["openai"] == {"model": "gpt-4o", "user": "u1", "temperature": 0.2}

        google_client = MagicMock()
        google_client.aio.models.generate_content = AsyncMock(return_value=MagicMock())
        agent._entry_clients["gemini"] = google_client
        await agent._call_llm_with_retry(
            [{"role": "user", "content": "hi"}], model_entry=agent._model_entries["gemini"]
        )
        call = google_client.aio.models.generate_content.await_args.kwargs
        assert call["model"] == "gemini-2.5-flash"
        assert call["config"].temperature == pytest.approx(0.2)
        assert not hasattr(call["config"], "user")


# ---------------------------------------------------------------------------
# Fallbacks and retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFallbacks:
    async def test_fallback_by_name_uses_entry(self, created, calls):
        agent = _agent(fallback_models=["vllm"])
        agent._fail_models = {"gpt-5-mini"}
        await _run(agent, {})
        assert [c["model"] for c in calls] == ["gpt-5-mini", "deepseek-r1"]
        assert calls[1]["client"].api_key == "vllm-key"
        assert calls[1]["base_url"] == "http://gpu1:8000/v1"

    async def test_fallback_after_picked_entry(self, created, calls):
        agent = _agent(fallback_models=["openai"])
        agent._fail_models = {"gemini-2.5-flash"}
        await _run(agent, {"model": "gemini"})
        assert [c["model"] for c in calls] == ["gemini-2.5-flash", "gpt-5-mini"]
        assert calls[1]["client"] is agent.client  # the default entry's own client

    async def test_old_fallback_forms_unchanged(self, created, calls):
        agent = _agent(fallback_models=["gpt-4o-mini", ("gemini-2.0-flash", "google")])
        agent._fail_models = {"gpt-5-mini", "gpt-4o-mini"}
        await _run(agent, {})
        assert [(c["model"], c["provider"]) for c in calls] == [
            ("gpt-5-mini", "openai"),
            ("gpt-4o-mini", "openai"),
            ("gemini-2.0-flash", "google"),
        ]
        # The cross-provider fallback gets none of the default's credentials.
        assert calls[2]["client"].api_key is None

    async def test_old_fallback_forms_without_registry(self, created, calls):
        agent = Agent(
            model="gpt-4o",
            provider="openai",
            retry_config=None,
            fallback_models=["gpt-4o-mini", ("gemini-2.0-flash", "google")],
        )
        agent._fail_models = {"gpt-4o", "gpt-4o-mini"}
        await _run(agent, {"model": "ignored"})
        assert [(c["model"], c["provider"]) for c in calls] == [
            ("gpt-4o", "openai"),
            ("gpt-4o-mini", "openai"),
            ("gemini-2.0-flash", "google"),
        ]

    async def test_entry_retry_config_and_breaker_key(self, created, calls):
        registry = {
            "a": {"model": "gpt-4o"},
            "b": {
                "model": "gpt-4o-mini",
                "retry_config": RetryConfig(max_retries=0, circuit_breaker_enabled=True),
            },
        }
        agent = _agent(model="a", model_registry=registry)
        agent._fail_models = {"gpt-4o-mini"}
        with pytest.raises(ValueError, match="gpt-4o-mini failed"):
            await _run(agent, {"model": "b"})
        assert len(calls) == 1
        assert ("openai", "gpt-4o-mini") in agent._circuit_breakers
