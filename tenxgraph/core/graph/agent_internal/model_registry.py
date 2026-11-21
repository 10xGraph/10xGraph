"""Named model entries for one Agent, and the per-run choice between them.

``Agent(model="openai", model_registry={...})`` registers several models; the run config
names one (``config["model"]`` by default). Each entry is resolved once at construction with
the same rules ``Agent(model=..., provider=..., base_url=...)`` uses, gets its own client
(built once, on first use) and is bound per attempt on a private copy of the Agent, so
concurrent runs on different entries never share state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tenxgraph.core.exceptions import ModelNotFoundError
from tenxgraph.core.llm.client_factory import resolve_provider_and_model

from .constants import (
    DEFAULT_RETRY_CONFIG,
    REASONING_DEFAULT,
    RetryConfig,
    default_reasoning_config,
)


# Agent keys that describe one model's endpoint and credentials. With a registry they belong
# in the entries; on the Agent as well they are refused.
ENTRY_ONLY_KEYS = (
    "provider",
    "base_url",
    "api_key",
    "api_style",
    "use_vertex_ai",
    "anthropic_backend",
)

# Entry keys 10xGraph reads itself; every other key is a provider kwarg (llm_kwargs).
_ENTRY_SETTINGS = frozenset(
    {
        "model",
        "provider",
        "base_url",
        "api_style",
        "use_vertex_ai",
        "reasoning_config",
        "retry_config",
    }
)


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One model resolved for an Agent: what an attempt on it needs."""

    name: str | None
    model: str
    provider: str
    base_url: str | None
    api_style: str
    use_vertex_ai: bool
    llm_kwargs: dict[str, Any]
    reasoning_config: dict[str, Any] | None
    reasoning_explicit: bool
    reasoning_default: bool
    retry_config: RetryConfig | None


def _env_use_vertex_ai() -> bool:
    return os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"


def _retry(value: Any) -> RetryConfig | None:
    if value is True:
        return DEFAULT_RETRY_CONFIG
    return value if isinstance(value, RetryConfig) else None


def resolve_entry(
    name: str | None,
    spec: dict[str, Any],
    shared_kwargs: dict[str, Any],
    reasoning_config: Any,
    retry_config: Any,
) -> ModelEntry:
    """Resolve one entry. ``shared_kwargs``, ``reasoning_config`` and ``retry_config`` are the
    Agent's values; the entry's own keys win over them."""
    model = spec["model"]
    base_url = spec.get("base_url")
    # "responses" for OpenAI, "chat" behind a custom base_url (most OpenAI-compatible servers
    # only speak Chat Completions). An explicit value wins.
    api_style = spec.get("api_style") or ("chat" if base_url else "responses")
    if api_style not in ("chat", "responses"):
        raise ValueError(f"Invalid api_style '{api_style}'. Supported: 'chat', 'responses'")
    use_vertex_ai = spec["use_vertex_ai"] if "use_vertex_ai" in spec else _env_use_vertex_ai()

    provider = spec.get("provider")
    if provider is not None:
        provider = provider.lower()
    else:
        # A recognised ``provider/`` prefix selects the provider and is stripped; unknown
        # prefixes resolve to ``openai`` and keep the full model name.
        provider, model = resolve_provider_and_model(model, use_vertex_ai=use_vertex_ai)

    # True or sentinel = reasoning on at the provider's lowest effort; False or None = off;
    # a dict is sent as given. The chat path only sends an explicit choice.
    reasoning = spec.get("reasoning_config", reasoning_config)
    reasoning_explicit = reasoning is not REASONING_DEFAULT
    reasoning_default = reasoning is REASONING_DEFAULT or reasoning is True
    if reasoning_default:
        reasoning = default_reasoning_config(provider)

    return ModelEntry(
        name=name,
        model=model,
        provider=provider,
        base_url=base_url,
        api_style=api_style,
        use_vertex_ai=use_vertex_ai,
        llm_kwargs={
            **shared_kwargs,
            **{k: v for k, v in spec.items() if k not in _ENTRY_SETTINGS},
        },
        reasoning_config=None if reasoning is False or reasoning is None else reasoning,
        reasoning_explicit=reasoning_explicit,
        reasoning_default=reasoning_default,
        retry_config=_retry(spec.get("retry_config", retry_config)),
    )


def build_registry(
    registry: Any,
    default_name: str,
    agent_kwargs: dict[str, Any],
    reasoning_config: Any,
    retry_config: Any,
) -> dict[str, ModelEntry]:
    """Validate ``model_registry`` and resolve every entry."""
    if not isinstance(registry, dict) or not registry:
        raise ValueError("model_registry must be a non-empty dict of name -> entry dict")
    conflicts = [key for key in ENTRY_ONLY_KEYS if key in agent_kwargs]
    if conflicts:
        raise ValueError(
            f"{', '.join(conflicts)} cannot be set on the Agent together with model_registry; "
            "set them in the registry entries instead"
        )
    for name, spec in registry.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"model_registry names must be non-empty strings, got {name!r}")
        if not isinstance(spec, dict):
            raise ValueError(f"model_registry entry '{name}' must be a dict")
        if not isinstance(spec.get("model"), str) or not spec["model"]:
            raise ValueError(f"model_registry entry '{name}' needs a 'model' string")
    if default_name not in registry:
        raise ValueError(
            f"model='{default_name}' must name a model_registry entry; "
            f"registered: {', '.join(registry)}"
        )
    return {
        name: resolve_entry(name, spec, agent_kwargs, reasoning_config, retry_config)
        for name, spec in registry.items()
    }


def select_entry(agent: Any, config: dict[str, Any]) -> ModelEntry | None:
    """The entry ``config`` picks for this run, or None for the Agent's default model.

    An Agent without a registry ignores the key. A raw model string is never accepted.
    """
    entries: dict[str, ModelEntry] | None = getattr(agent, "_model_entries", None)
    if not entries:
        return None
    name = config.get(getattr(agent, "model_config_key", "model"))
    if name is None or name == agent.default_model_name:
        return None
    if not isinstance(name, str) or name not in entries:
        raise ModelNotFoundError(name, config.get("_node_name"), list(entries))
    return entries[name]


def named_entry(agent: Any, model: str, provider: str | None) -> ModelEntry | None:
    """The registry entry a fallback string names, or None for a plain model fallback."""
    entries: dict[str, ModelEntry] | None = getattr(agent, "_model_entries", None)
    if not entries or provider is not None:
        return None
    return entries.get(model)


def bind_entry(agent: Any, entry: ModelEntry, client: Any) -> Any:
    """A private copy of ``agent`` configured for ``entry``; the shared Agent is unchanged."""
    bound = agent._bind_attempt(
        entry.model, entry.provider, client, entry.base_url, llm_kwargs=entry.llm_kwargs
    )
    bound.api_style = entry.api_style
    bound.use_vertex_ai = entry.use_vertex_ai
    bound.reasoning_config = entry.reasoning_config
    bound._reasoning_explicit = entry.reasoning_explicit
    bound._reasoning_default = entry.reasoning_default
    bound.retry_config = entry.retry_config
    # The default model's last API style must not leak into another entry's converter key.
    bound.__dict__.pop("_effective_api_style", None)
    return bound


def entry_attempt(agent: Any, entry: ModelEntry, retry_cfg: RetryConfig | None) -> Any:
    """Bind ``entry`` with its own client, built once per entry and cached on the Agent.

    The client gets only that entry's key, endpoint and backend, never the Agent's.
    """
    cache: dict[str | None, Any] = agent.__dict__.setdefault("_entry_clients", {})
    client = cache.get(entry.name)
    if client is None:
        helper = bind_entry(agent, entry, None)
        client = helper._create_client(entry.provider, entry.base_url, entry.use_vertex_ai)
        cache[entry.name] = client
    bound = bind_entry(agent, entry, client)
    bound.client = bound._without_sdk_retries(client, retry_cfg)
    return bound
