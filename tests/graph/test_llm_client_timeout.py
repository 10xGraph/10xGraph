"""G-59: the LLM client timeout has a TENXGRAPH_ name, the old name still works."""

from __future__ import annotations

import pytest

from tenxgraph.core.llm import client_factory


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("TENXGRAPH_LLM_TIMEOUT", raising=False)
    monkeypatch.delenv("AGENTFLOW_LLM_TIMEOUT", raising=False)
    client_factory.set_default_llm_timeout(None)
    yield
    client_factory.set_default_llm_timeout(None)


def test_tenxgraph_env_name_sets_timeout(monkeypatch):
    monkeypatch.setenv("TENXGRAPH_LLM_TIMEOUT", "42")
    assert client_factory.get_default_llm_timeout() == 42.0


def test_agentflow_env_name_is_fallback(monkeypatch):
    monkeypatch.setenv("AGENTFLOW_LLM_TIMEOUT", "17")
    assert client_factory.get_default_llm_timeout() == 17.0


def test_new_name_wins_over_old(monkeypatch):
    monkeypatch.setenv("TENXGRAPH_LLM_TIMEOUT", "5")
    monkeypatch.setenv("AGENTFLOW_LLM_TIMEOUT", "99")
    assert client_factory.get_default_llm_timeout() == 5.0


def test_google_client_honours_explicit_timeout(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = client_factory.create_llm_client("google", timeout=12)
    assert client._api_client._http_options.timeout == 12000


def test_google_client_pins_v1beta(monkeypatch):
    """google-genai 3.x defaults to "v1", which rejects function calling."""
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("GOOGLE_GENAI_API_VERSION", raising=False)
    client = client_factory.create_llm_client("google")
    assert client._api_client._http_options.api_version == "v1beta"


def test_google_client_keeps_env_api_version(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_GENAI_API_VERSION", "v1alpha")
    from google import genai

    seen = {}
    monkeypatch.setattr(genai, "Client", lambda **kw: seen.update(kw))
    client_factory.create_llm_client("google")
    # Left to the SDK, which reads the env variable (3.x).
    assert seen["http_options"].api_version is None
