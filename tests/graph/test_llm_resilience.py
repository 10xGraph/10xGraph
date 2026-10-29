"""Group 2 regressions: fallback isolation (G-08, G-45), retry policy (G-44), breaker (G-46)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.agent_internal import execution
from tenxgraph.core.graph.agent_internal.circuit_breaker import CircuitBreaker, CircuitState
from tenxgraph.core.graph.agent_internal.constants import RetryConfig


class _StatusError(Exception):
    def __init__(self, status: int, message: str = "error", headers=None, body=None):
        super().__init__(message)
        self.status_code = status
        self.body = body
        if headers is not None:
            self.response = SimpleNamespace(status_code=status, headers=headers)


def _agent(retry: RetryConfig | None = None, fallback=None, **kw) -> Agent:
    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        agent = Agent(
            model="gpt-4o",
            provider="openai",
            retry_config=retry or RetryConfig(max_retries=0, jitter=0.0),
            fallback_models=fallback,
            reasoning_config=None,
            **kw,
        )
    agent.client = "primary-client"
    return agent


# ---------------------------------------------------------------- G-08


@pytest.mark.asyncio
async def test_fallback_does_not_rewrite_shared_agent_across_awaits():
    agent = _agent(fallback=[("gemini-2.0-flash", "google")])
    seen_by_other_run: list[tuple[str, str]] = []
    release = asyncio.Event()

    async def fake_call(self, messages, tools=None, stream=False, **kw):
        if self.provider == "openai":
            raise _StatusError(503)
        # fallback is now suspended mid-call; a concurrent run observes the agent
        await release.wait()
        return SimpleNamespace(who=self.model)

    async def observer():
        await asyncio.sleep(0.01)
        seen_by_other_run.append((agent.model, agent.provider))
        release.set()

    with (
        patch.object(Agent, "_call_llm", fake_call),
        patch.object(Agent, "_create_client", return_value="fb-client"),
    ):
        result, _ = await asyncio.gather(agent._call_llm_with_retry([]), observer())

    assert result.who == "gemini-2.0-flash"
    assert seen_by_other_run == [("gpt-4o", "openai")]
    assert agent.client == "primary-client"


@pytest.mark.asyncio
async def test_cross_provider_fallback_response_uses_fallback_converter():
    agent = _agent(fallback=[("gemini-2.0-flash", "google")])

    async def fake_call(self, messages, tools=None, stream=False, **kw):
        if self.provider == "openai":
            raise _StatusError(503)
        return SimpleNamespace()

    with (
        patch.object(Agent, "_call_llm", fake_call),
        patch.object(Agent, "_create_client", return_value="fb-client"),
    ):
        execution.converter_key_var.set(None)
        await agent._call_llm_with_retry([])
    assert execution.converter_key_var.get() == "google"


# ---------------------------------------------------------------- G-45


@pytest.mark.asyncio
async def test_cross_provider_fallback_does_not_get_primary_credentials():
    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        agent = Agent(
            model="gpt-4o",
            provider="openai",
            api_key="sk-primary",
            base_url="http://primary.example/v1",
            retry_config=RetryConfig(max_retries=0, jitter=0.0),
            fallback_models=[("claude-sonnet-5", "anthropic")],
            reasoning_config=None,
        )
    built: list[dict] = []

    def fake_create(self, provider, base_url=None, use_vertex_ai=False):
        built.append(
            {"provider": provider, "base_url": base_url, "api_key": self.llm_kwargs.get("api_key")}
        )
        return "fb-client"

    async def fake_call(self, messages, tools=None, stream=False, **kw):
        if self.provider == "openai":
            raise _StatusError(503)
        return SimpleNamespace()

    with (
        patch.object(Agent, "_call_llm", fake_call),
        patch.object(Agent, "_create_client", fake_create),
    ):
        await agent._call_llm_with_retry([])

    assert built == [{"provider": "anthropic", "base_url": None, "api_key": None}]


# ---------------------------------------------------------------- G-44


@pytest.mark.asyncio
async def test_retry_delay_has_jitter():
    cfg = RetryConfig(max_retries=1, initial_delay=10.0, jitter=0.5)
    agent = _agent(retry=cfg)
    agent._call_llm = AsyncMock(side_effect=[_StatusError(503), SimpleNamespace()])
    delays: list[float] = []

    async def cap(d):
        delays.append(d)

    with (
        patch.object(execution.asyncio, "sleep", cap),
        patch.object(execution.random, "random", return_value=1.0),
    ):
        await agent._call_llm_with_retry([])
    assert delays == [5.0]


@pytest.mark.asyncio
async def test_retry_after_header_is_honoured():
    cfg = RetryConfig(max_retries=1, initial_delay=1.0, jitter=0.0)
    agent = _agent(retry=cfg)
    agent._call_llm = AsyncMock(
        side_effect=[_StatusError(429, headers={"retry-after": "5"}), SimpleNamespace()]
    )
    delays: list[float] = []

    async def cap(d):
        delays.append(d)

    with patch.object(execution.asyncio, "sleep", cap):
        await agent._call_llm_with_retry([])
    assert delays == [5.0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        _StatusError(
            429, "You exceeded your current quota", body={"error": {"code": "insufficient_quota"}}
        ),
        _StatusError(429, "You have reached your specified API usage limits"),
    ],
)
async def test_spend_cap_429_is_not_retried(exc):
    agent = _agent(retry=RetryConfig(max_retries=3, jitter=0.0))
    agent._call_llm = AsyncMock(side_effect=exc)
    with (
        patch.object(execution.asyncio, "sleep", AsyncMock()) as sleep,
        pytest.raises(_StatusError),
    ):
        await agent._call_llm_with_retry([])
    assert agent._call_llm.await_count == 1
    sleep.assert_not_called()


@pytest.mark.asyncio
async def test_sdk_retries_do_not_multiply_ours():
    class FakeOpenAIClient:
        __module__ = "openai._client"

        def __init__(self, opts):
            self.opts = opts

        def with_options(self, **kw):
            return FakeOpenAIClient({**self.opts, **kw})

    agent = _agent(retry=RetryConfig(max_retries=0, jitter=0.0))
    agent.client = FakeOpenAIClient({})
    clients = []

    async def fake_call(self, messages, tools=None, stream=False, **kw):
        clients.append(self.client)
        return SimpleNamespace()

    with patch.object(Agent, "_call_llm", fake_call):
        await agent._call_llm_with_retry([])
    assert clients[0].opts == {"max_retries": 0}


# ---------------------------------------------------------------- G-46


def test_half_open_lets_only_one_trial_through():
    now = [0.0]
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=10.0, time_func=lambda: now[0])
    cb.record_failure()
    now[0] = 11.0
    assert cb.allow() is True
    assert cb.state is CircuitState.half_open
    assert cb.allow() is False
    cb.record_success()
    assert cb.allow() is True


def test_half_open_trial_that_never_reports_is_retried_after_timeout():
    now = [0.0]
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=10.0, time_func=lambda: now[0])
    cb.record_failure()
    now[0] = 11.0
    assert cb.allow() is True
    now[0] = 25.0
    assert cb.allow() is True


@pytest.mark.asyncio
async def test_max_concurrent_calls_limits_in_flight_requests():
    agent = _agent(retry=RetryConfig(max_retries=0, jitter=0.0, max_concurrent_calls=2))
    active = 0
    peak = 0

    async def fake_call(self, messages, tools=None, stream=False, **kw):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return SimpleNamespace()

    with patch.object(Agent, "_call_llm", fake_call):
        await asyncio.gather(*[agent._call_llm_with_retry([]) for _ in range(6)])
    assert peak == 2


# ---------------------------------------------------------------- G-48 follow-up


@pytest.mark.asyncio
async def test_openai_chat_stream_requests_usage():
    agent = _agent()
    create = AsyncMock(return_value=MagicMock())
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    msgs = [{"role": "user", "content": "hi"}]
    await agent._call_openai(msgs, None, True)
    assert create.call_args.kwargs["stream_options"] == {"include_usage": True}
    await agent._call_openai(msgs, None, False)
    assert "stream_options" not in create.call_args.kwargs
    await agent._call_openai(msgs, None, True, stream_options={"include_usage": False})
    assert create.call_args.kwargs["stream_options"] == {"include_usage": False}
