"""Shared constants for Agent internals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

CLIENT_CONSTRUCTOR_KWARGS = frozenset(
    {
        "organization",
        "project",
        "timeout",
        "max_retries",
        "default_headers",
        "default_query",
        "http_client",
    }
)

# Keys that must never be forwarded to request calls.
CALL_EXCLUDED_KWARGS = CLIENT_CONSTRUCTOR_KWARGS | frozenset(
    {
        "api_key",
        "base_url",
    }
)

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
DEFAULT_REASONING_EFFORT_BY_PROVIDER: dict[str, str] = {
    "openai": "minimal",
    "anthropic": "low",
    "google": "low",
}

# Sentinel that distinguishes the default reasoning config from an explicit None.
REASONING_DEFAULT: object = object()


def default_reasoning_config(provider: str) -> dict[str, Any]:
    """Return the request config ``reasoning_config=True`` stands for on ``provider``."""
    return {"effort": DEFAULT_REASONING_EFFORT_BY_PROVIDER.get(provider, "low")}


def resolve_reasoning_config(agent: Any) -> dict[str, Any] | None:
    """Return the reasoning config to send for ``agent``'s current provider, or None.

    An explicit dict is returned as given. The default (``True`` or unset) is resolved from
    the attempt's ``provider`` here rather than at construction, so a fallback model on
    another provider gets that provider's lowest effort.
    """
    config = getattr(agent, "reasoning_config", None)
    if not config:
        return None
    if getattr(agent, "_reasoning_default", False):
        return default_reasoning_config(getattr(agent, "provider", ""))
    return config
