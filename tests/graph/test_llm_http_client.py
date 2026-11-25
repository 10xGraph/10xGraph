"""A custom http_client (mTLS, proxy) reaches the Google client and survives fallbacks."""

from __future__ import annotations

import pytest


genai = pytest.importorskip("google.genai")
try:
    import httpx
except ImportError:  # google-genai 3.x depends on httpx2 only
    import httpx2 as httpx

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.agent_internal.model_registry import entry_attempt
from tenxgraph.core.llm import client_factory


@pytest.fixture(autouse=True)
def _google_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")


def _async_client_of(client):
    return client._api_client._async_httpx_client


@pytest.mark.parametrize("vertex", [False, True])
def test_google_client_uses_custom_http_client(vertex):
    custom = httpx.AsyncClient()
    client = client_factory.create_llm_client("google", use_vertex_ai=vertex, http_client=custom)
    assert _async_client_of(client) is custom


def test_google_client_without_http_client_builds_its_own():
    client = client_factory.create_llm_client("google")
    assert client._api_client._http_options.httpx_async_client is None
    assert _async_client_of(client) is not None


def test_google_client_httpx2_needs_google_genai_3():
    """google-genai 1.x/2.x take only httpx; 3.x also takes httpx2."""
    httpx2 = pytest.importorskip("httpx2")
    custom = httpx2.AsyncClient()
    if int(genai.__version__.split(".")[0]) >= 3:
        client = client_factory.create_llm_client("google", http_client=custom)
        assert _async_client_of(client) is custom
    else:
        with pytest.raises(ValueError, match="httpx_async_client"):
            client_factory.create_llm_client("google", http_client=custom)


def test_google_client_rejects_other_types():
    with pytest.raises(ValueError, match="httpx_async_client"):
        client_factory.create_llm_client("google", http_client=object())


def test_google_client_uses_explicit_api_key(monkeypatch):
    seen = {}
    monkeypatch.setattr(genai, "Client", lambda **kw: seen.update(kw))
    client_factory.create_llm_client("google", api_key="explicit")
    assert seen["api_key"] == "explicit"


def test_agent_passes_http_client_to_google():
    custom = httpx.AsyncClient()
    agent = Agent(model="gemini-2.5-flash", provider="google", http_client=custom)
    assert _async_client_of(agent.client) is custom


def test_same_provider_fallback_reuses_http_client():
    custom = httpx.AsyncClient()
    agent = Agent(
        model="gemini-2.5-flash",
        provider="google",
        http_client=custom,
        fallback_models=["gemini-2.0-flash", ("gemini-2.5-pro", "google")],
    )
    assert _async_client_of(agent._fallback_client("google", True, None)) is custom


def test_cross_provider_fallback_with_http_client_raises():
    with pytest.raises(ValueError, match="model_registry"):
        Agent(
            model="gemini-2.5-flash",
            provider="google",
            http_client=httpx.AsyncClient(),
            fallback_models=[("gpt-4o-mini", "openai")],
        )


def test_cross_provider_fallback_without_http_client_is_allowed():
    agent = Agent(
        model="gemini-2.5-flash",
        provider="google",
        fallback_models=[("gpt-4o-mini", "openai")],
    )
    assert agent.fallback_models == [("gpt-4o-mini", "openai")]


def test_registry_fallback_gets_its_own_http_client(monkeypatch):
    httpx2 = pytest.importorskip("httpx2")
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    google_http = httpx.AsyncClient()
    claude_http = httpx2.AsyncClient()
    agent = Agent(
        model="gemini",
        model_registry={
            "gemini": {
                "model": "gemini-2.5-flash",
                "provider": "google",
                "http_client": google_http,
            },
            "claude": {"model": "claude-haiku-4-5", "http_client": claude_http},
        },
        fallback_models=["claude"],
    )
    assert _async_client_of(agent.client) is google_http
    bound = entry_attempt(agent, agent._model_entries["claude"], None)
    assert bound.client._client is claude_http


@pytest.mark.asyncio
async def test_registry_google_entry_uploads_media_with_its_own_client(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from tenxgraph.core.graph.agent_internal.model_registry import bind_entry, entry_client
    from tenxgraph.storage.media.resolver import MediaRefResolver

    entry_http = httpx.AsyncClient()
    agent = Agent(
        model="other",
        model_registry={
            "other": {"model": "gemini-2.5-pro", "provider": "google", "http_client": entry_http}
        },
    )
    entry = agent._model_entries["other"]
    seen = []

    async def fake_resolve(self, ref, model=None):
        seen.append(self.google_client)
        return MagicMock(spec=["file_data"], file_data=MagicMock(file_uri="u"))

    monkeypatch.setattr(MediaRefResolver, "resolve_for_google", fake_resolve)
    run = bind_entry(agent, entry, None)
    run.client = entry_client(agent, entry)
    run.media_store = MagicMock()
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "graph://media/k"}}]}
    ]
    await run._resolve_media_in_messages(messages)
    assert seen == [agent._entry_clients["other"]]
    assert _async_client_of(seen[0]) is entry_http
    assert entry_client(agent, entry) is seen[0]
