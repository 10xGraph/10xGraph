"""Shared constants for Agent internals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tenxgraph.core.llm.client_factory import CLIENT_CONSTRUCTOR_KWARGS


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Configuration for LLM call retry and fallback behaviour.

    Attributes:
        max_retries: Maximum number of retry attempts for the *primary* model
            before moving to the next fallback (default ``3``).
        initial_delay: Delay in seconds before the first retry (default ``1.0``).
        max_delay: Upper-bound cap on exponential back-off delay (default ``30.0``).
        backoff_factor: Multiplier applied after each retry (default ``2.0``).
        retryable_status_codes: HTTP status codes considered transient/retryable.
        circuit_breaker_enabled: When True, track failures per (provider, model)
            and skip a target whose circuit is open, moving straight to the next
            fallback instead of retrying a known-dead provider (default ``False``).
        circuit_breaker_threshold: Consecutive failures that open a circuit
            (default ``5``).
        circuit_breaker_reset_timeout: Seconds a circuit stays open before a
            single half-open trial is allowed (default ``30.0``).
        jitter: Fraction of each back-off delay that is randomised, so parallel
            runs do not retry in lock-step: the sleep is
            ``delay * (1 - jitter * random())`` (default ``0.25``; ``0`` disables).
        max_concurrent_calls: Opt-in cap on in-flight LLM requests per provider
            for one Agent instance (default ``None``, no cap). For streaming calls the
            cap covers opening the stream, not reading it.
        distributed_max_concurrent_calls: Opt-in cap on in-flight LLM requests per
            (provider, model) shared by every process that uses the same Redis
            (default ``None``, off). Needs ``distributed_redis`` or
            ``distributed_redis_url``. Applies on top of ``max_concurrent_calls``. Fails
            open with a WARNING when Redis is unreachable.
        distributed_redis: An async Redis client for the shared cap.
        distributed_redis_url: Redis URL used to build a client when none is passed.
        distributed_lease_seconds: How long a slot is held before it counts as abandoned
            (default ``300``). A running call renews its lease every third of this time, so it
            only has to outlast a crashed process; that process frees its slots after this
            long.
        distributed_key_prefix: Prefix for the Redis key, like the checkpointer's
            ``redis_key_prefix`` (default ``""``).

    A ``Retry-After`` header on a retryable error is honoured when it is longer than
    the computed delay. When it is longer than ``max_delay`` the model is not retried
    and the next fallback is tried. A 429 that reports a permanent quota or spend cap
    is never retried.
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 529}),
    )
    circuit_breaker_enabled: bool = False
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_timeout: float = 30.0
    jitter: float = 0.25
    max_concurrent_calls: int | None = None
    distributed_max_concurrent_calls: int | None = None
    distributed_redis: Any = None
    distributed_redis_url: str | None = None
    distributed_lease_seconds: float = 300.0
    distributed_key_prefix: str = ""


DEFAULT_RETRY_CONFIG = RetryConfig()

# Keys 10xGraph reads itself from ``llm_kwargs``; no SDK call defines them.
FRAMEWORK_KWARGS = frozenset({"stream_idle_timeout", "anthropic_cache", "anthropic_backend"})

# One model's credentials and endpoint. Passed to its client explicitly, never in the
# constructor kwargs, and never handed to a fallback on another provider.
ENDPOINT_KWARGS = frozenset({"api_key", "base_url", "anthropic_backend"})

# Keys that must never be forwarded to request calls.
CALL_EXCLUDED_KWARGS = CLIENT_CONSTRUCTOR_KWARGS | FRAMEWORK_KWARGS | ENDPOINT_KWARGS

VALID_OUTPUT_TYPES = ("text", "image", "video", "audio", "json")
GOOGLE_OUTPUT_TYPES = ("text", "image", "video", "audio", "json")
OPENAI_OUTPUT_TYPES = ("text", "image", "audio", "json")
# Anthropic's Messages API generates text and tool calls only. Structured output
# is text constrained by a JSON schema, so "json" is supported; there is no
# image/audio/video generation endpoint.
ANTHROPIC_OUTPUT_TYPES = ("text", "json")

# Models that reject temperature / top_p / top_k with a 400. Sampling params are
# stripped for these before the request is sent. Older Claude models still accept
# them, so this is a per-model set rather than a blanket strip.
ANTHROPIC_NO_SAMPLING_MODELS = frozenset(
    {
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
    }
)

# Claude families before 4.6. They reject thinking={"type": "adaptive"} with a 400 and need
# thinking={"type": "enabled", "budget_tokens": N}. Matched as a prefix of the bare model id, so
# dated ids (-20251001), Vertex dates (@20251101) and -latest aliases all match. Unknown and
# newer models take the adaptive path.
ANTHROPIC_LEGACY_THINKING_PREFIXES = (
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-opus-4-0",
    "claude-sonnet-4-0",
    "claude-3-7-sonnet",
    "claude-sonnet-3-7",
    "claude-opus-4-2025",
    "claude-sonnet-4-2025",
)

# Only Opus 4.5 among the legacy models accepts output_config.effort, low/medium/high only.
ANTHROPIC_LEGACY_EFFORT_PREFIXES = ("claude-opus-4-5",)

# Thinking budget (tokens) for legacy models when reasoning_config gives only an effort.
ANTHROPIC_LEGACY_BUDGET_BY_EFFORT = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 16384,
}
ANTHROPIC_MIN_THINKING_BUDGET = 1024

# Anthropic requires max_tokens on every request. Streaming gets a larger default:
# a large max_tokens without streaming risks an HTTP timeout.
ANTHROPIC_DEFAULT_MAX_TOKENS = 16000
ANTHROPIC_DEFAULT_MAX_TOKENS_STREAMING = 64000

GOOGLE_THINKING_BUDGET_BY_EFFORT = {
    "low": 512,
    "medium": 8192,
    "high": 24576,
}

# What ``reasoning_config=True`` (and the default) means on each provider: reasoning on at
# the lowest effort that still reasons. Resolved per request from the attempt's provider, so a
# cross-provider fallback gets its own provider's value. Allowed values, from the SDK types:
# - openai: ``ReasoningEffort`` in openai/types/shared/reasoning_effort.py
#   ("none" turns reasoning off, so "minimal" is the lowest that reasons).
# - anthropic: ``OutputConfigParam.effort`` in anthropic/types/output_config_param.py.
# - google: the effort path sends ``thinking_budget``; "low" maps to 512 tokens below, the
#   lowest budget every Gemini 2.5 family accepts (Gemini 3 also takes a budget).
# Anthropic also gets ``display: "summarized"`` (see ``default_reasoning_config``): current
# Claude models default to "omitted", which returns thinking blocks with an empty summary.
DEFAULT_REASONING_EFFORT_BY_PROVIDER: dict[str, str] = {
    "openai": "minimal",
    "anthropic": "low",
    "google": "low",
}

# Sentinel that distinguishes the default reasoning config from an explicit None.
REASONING_DEFAULT: object = object()

# Models that answered the default reasoning parameter with a 400 (non-reasoning models such as
# gpt-4o). Process-local: later calls with the default config skip reasoning for them. An
# explicit reasoning_config is never affected. Filled by the OpenAI route only.
REASONING_REJECTED_MODELS: set[str] = set()


def reasoning_mode(value: Any) -> str:
    """Classify a ``reasoning_config`` argument.

    ``"default"`` (not passed), ``"on"`` (``True``), ``"off"`` (``False`` or ``None``) or
    ``"explicit"`` (a dict, sent as given). "default" and "on" both mean the provider's lowest
    effort; they differ only on OpenAI Chat Completions, which sends "on" but not "default".
    """
    if value is REASONING_DEFAULT:
        return "default"
    if value is True:
        return "on"
    if value is False or value is None:
        return "off"
    return "explicit"


def default_reasoning_config(provider: str) -> dict[str, Any]:
    """Return the request config ``reasoning_config=True`` stands for on ``provider``.

    Anthropic adds ``display: "summarized"`` so the reasoning summary is readable.
    """
    config = {"effort": DEFAULT_REASONING_EFFORT_BY_PROVIDER.get(provider, "low")}
    if provider == "anthropic":
        config["display"] = "summarized"
    return config


def is_default_reasoning(agent: Any) -> bool:
    """True when ``agent``'s reasoning comes from the default (``True`` or not passed)."""
    return getattr(agent, "_reasoning_mode", "explicit") in ("default", "on")


def reasoning_for(agent: Any, api: str) -> dict[str, Any] | None:
    """The reasoning config to send on this request, or None. The one place that decides it.

    ``api`` is ``"anthropic"``, ``"google"``, ``"openai_responses"`` or ``"openai_chat"``.
    An explicit dict is returned as given. The default is resolved from the attempt's
    ``provider`` here rather than at construction, so a fallback model on another provider
    gets that provider's lowest effort. On OpenAI the default is skipped for a model in
    ``REASONING_REJECTED_MODELS``, and Chat Completions skips the implicit default because
    many models reject ``reasoning_effort`` there, notably together with function tools.
    A config set on an agent that was built without a default counts as explicit.
    """
    config = getattr(agent, "reasoning_config", None)
    if not config:
        return None
    mode = getattr(agent, "_reasoning_mode", "explicit")
    if mode not in ("default", "on"):
        return config
    if api.startswith("openai"):
        if api == "openai_chat" and mode == "default":
            return None
        if getattr(agent, "model", None) in REASONING_REJECTED_MODELS:
            return None
    return default_reasoning_config(getattr(agent, "provider", ""))
