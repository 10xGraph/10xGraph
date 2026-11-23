"""Shared single-turn LLM call utility.

Used by SummaryContextManager, the eval judge (LLMCallerMixin) and UserSimulator.

Returns a plain 4-tuple so callers can choose how much they consume:

    text, *_ = await call_llm(...)              # only text
    text, inp, out, cache = await call_llm(...)  # text + token counts

OpenAI supports two API styles:
- ``"responses"`` (default) — ``client.responses.create()``, the current
  recommended API with ``input`` / ``instructions`` / ``max_output_tokens``.
- ``"chat"`` — ``client.chat.completions.create()``, the legacy style
  required by older or third-party-hosted models (e.g. some Chinese models).

The request is built by the same provider code the Agent uses (client creation, request
assembly, SDK keyword filtering, Claude capability rules) and the reply is read by the same
response converters, so a provider fix reaches both. Reasoning is not sent, and Anthropic
prompt caching is off unless ``anthropic_cache`` is passed.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Literal

from tenxgraph.core.llm.client_factory import resolve_provider_and_model


logger = logging.getLogger("tenxgraph.llm.caller")

_JSON_INSTRUCTION = "\n\nRespond with valid JSON only, no prose and no code fences."


async def call_llm(
    model: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    json_mode: bool = False,
    use_vertex_ai: bool = False,
    api_style: Literal["responses", "chat"] = "responses",
    **llm_kwargs: Any,
) -> tuple[str, int, int, int]:
    """Single-turn LLM call with provider auto-detection.

    Args:
        model: Model identifier (e.g. ``"gemini-2.0-flash"``, ``"gpt-4o-mini"``).
            Provider is inferred from the name via ``detect_provider``.
        prompt: The user-turn content to send.
        system_prompt: Optional system instruction prepended to the request.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        json_mode: When ``True``, instructs the provider to return valid JSON.
        use_vertex_ai: When ``True``, force Google Vertex AI client.
        api_style: OpenAI only. ``"responses"`` (default) uses the current
            Responses API (``client.responses.create``). Use ``"chat"`` for
            models that only support the legacy Chat Completions endpoint
            (e.g. older or self-hosted Chinese models).
        **llm_kwargs: Provider-specific parameters, handled as the Agent handles its
            ``llm_kwargs``. Examples:

            - Google: ``cached_content="cachedContents/abc123"`` — attaches an
              explicit Gemini context cache created via the Google SDK.
            - OpenAI: ``prompt_cache_key="my-agent-v1"`` — improves cache hit
              rates across requests sharing the same long system-prompt prefix.
            - OpenAI: ``prompt_cache_retention="24h"`` — extends cache retention
              for gpt-5.5+ models (default is in-memory, ~5-10 min).

    Returns:
        ``(text, input_tokens, output_tokens, cache_read_tokens)`` — plain tuple.
        Token counts are 0 when the provider does not report them.
    """
    # resolve_provider_and_model (not detect_provider) so a recognised
    # ``provider/`` prefix is stripped before the name reaches the SDK. Sending
    # "gemini/gemini-2.5-flash" or "anthropic/claude-opus-5" verbatim fails.
    provider, model = resolve_provider_and_model(model, use_vertex_ai=use_vertex_ai)
    # The Agent's client rules: use_vertex_ai selects the Vertex backend for Claude, client
    # constructor keys (timeout, max_retries, ...) go to the client.
    client = _request(provider, model, None, llm_kwargs)._create_client(
        provider, llm_kwargs.get("base_url"), use_vertex_ai
    )

    if provider == "anthropic":
        call = _call_anthropic
    elif provider == "google":
        call = _call_google
    elif api_style == "chat":
        call = _call_openai_chat
    else:
        call = _call_openai_responses
    return await call(
        client,
        model,
        prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
        **llm_kwargs,
    )


@functools.cache
def _request_class() -> type:
    # Imported lazily: the agent package imports this package's client factory.
    from tenxgraph.core.graph.agent_internal.anthropic import AgentAnthropicMixin
    from tenxgraph.core.graph.agent_internal.google import AgentGoogleMixin
    from tenxgraph.core.graph.agent_internal.openai import AgentOpenAIMixin
    from tenxgraph.core.graph.agent_internal.providers import AgentProviderMixin

    class SingleTurnRequest(
        AgentAnthropicMixin, AgentGoogleMixin, AgentOpenAIMixin, AgentProviderMixin
    ):
        """The attributes the Agent's provider mixins read, for one text request."""

        output_type = "text"
        output_schema = None
        reasoning_config = None
        base_url = None
        media_policy = None

        def __init__(self, provider: str, model: str, client: Any, llm_kwargs: dict) -> None:
            self.provider = provider
            self.model = model
            self.client = client
            self.llm_kwargs = llm_kwargs

    return SingleTurnRequest


def _request(provider: str, model: str, client: Any, llm_kwargs: dict[str, Any]) -> Any:
    return _request_class()(provider, model, client, llm_kwargs)


def _messages(prompt: str, system_prompt: str | None) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    return [*messages, {"role": "user", "content": prompt}]


async def _reply(converter_key: str, response: Any) -> tuple[str, int, int, int]:
    """Text and token counts read by the Agent's response converter for ``converter_key``."""
    from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter

    message = await ModelResponseConverter(response, converter_key).converter.convert_response(
        response
    )
    usage = message.usages
    if usage is None:
        return message.text().strip(), 0, 0, 0
    cache = usage.cache_read_input_tokens or 0
    if cache:
        logger.debug("Cache hit: %d cached tokens (%s)", cache, converter_key)
    return message.text().strip(), usage.prompt_tokens, usage.completion_tokens, cache


async def _call_anthropic(
    client: Any,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    **llm_kwargs: Any,
) -> tuple[str, int, int, int]:
    from tenxgraph.core.graph.agent_internal.anthropic import rejects_sampling

    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    # call_llm defaults temperature for every caller; leave it out for the Claude models that
    # reject it instead of logging a dropped parameter on each call.
    if not rejects_sampling(model):
        kwargs["temperature"] = temperature
    if json_mode:
        # Anthropic has no response_mime_type; steer with an instruction, which
        # is what the Messages API supports without a full JSON schema.
        system_prompt = (system_prompt or "") + _JSON_INSTRUCTION

    request = _request("anthropic", model, client, {"anthropic_cache": False, **llm_kwargs})
    response = await request._call_anthropic(_messages(prompt, system_prompt), **kwargs)

    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        logger.warning(
            "Anthropic refused the request (category=%s)",
            getattr(details, "category", None),
        )
    return await _reply("anthropic", response)


async def _call_google(
    client: Any,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    **llm_kwargs: Any,
) -> tuple[str, int, int, int]:
    kwargs: dict[str, Any] = {"max_output_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    if llm_kwargs.get("cached_content"):
        # The system prompt is inside the cache; do not resend it.
        system_prompt = None

    request = _request("google", model, client, llm_kwargs)
    response = await request._call_google(_messages(prompt, system_prompt), **kwargs)
    return await _reply("google", response)


async def _call_openai_responses(
    client: Any,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    **llm_kwargs: Any,
) -> tuple[str, int, int, int]:
    """Call the OpenAI Responses API (client.responses.create)."""
    kwargs: dict[str, Any] = {"max_output_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}

    request = _request("openai", model, client, llm_kwargs)
    response = await request._call_openai_responses(_messages(prompt, system_prompt), **kwargs)
    return await _reply("openai_responses", response)


async def _call_openai_chat(
    client: Any,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    **llm_kwargs: Any,
) -> tuple[str, int, int, int]:
    """Call the OpenAI Chat Completions API (client.chat.completions.create)."""
    kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    request = _request("openai", model, client, llm_kwargs)
    response = await request._call_openai(_messages(prompt, system_prompt), **kwargs)
    return await _reply("openai", response)
