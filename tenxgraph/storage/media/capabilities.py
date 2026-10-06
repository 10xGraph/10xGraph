"""Provider-aware media capability matrix.

Defines the internal capability system that determines how media (images,
documents, audio, video) should be transported to different AI providers
and models.

This module is **not** a user-facing API. It drives the resolution fallback
chain used internally by the media resolver and provider converters.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger("agentflow.media.capabilities")


class MediaTransportMode(str, Enum):
    """How media should be delivered to a provider/model."""

    remote_url = "remote_url"
    """Send a public/signed URL directly (e.g. OpenAI image_url)."""

    provider_file = "provider_file"
    """Upload via provider-native file API (e.g. Google File API)."""

    inline_bytes = "inline_bytes"
    """Send raw bytes inline (e.g. base64 data URI, Part.from_bytes)."""

    unsupported = "unsupported"
    """The model/provider does not support this media type at all."""


@dataclass(frozen=True)
class ModelMediaCapabilities:
    """Describes what media transports a provider/model supports.

    Attributes:
        provider: Provider identifier (e.g. "openai", "google").
        model_pattern: Glob-style pattern matching model names
            (e.g. "gpt-4o*", "gemini-*").
        transport_order: Ordered list of preferred transport modes per
            media type.  The resolver tries these in order until one
            succeeds.  If the list contains only ``unsupported``, the
            model cannot handle that media type.
        accepts_external_urls: Whether the provider accepts arbitrary
            external ``https://`` URLs directly.
        supports_provider_file: Whether the provider has a native file
            upload API (e.g. Google File API).
        can_convert_internal_to_remote: Whether internal storage URLs
            (``agentflow://media/...``) can be converted to public/signed
            remote URLs.
    """

    provider: str
    model_pattern: str
    transport_order: dict[str, list[MediaTransportMode]] = field(
        default_factory=dict,
    )
    accepts_external_urls: bool = False
    supports_provider_file: bool = False
    can_convert_internal_to_remote: bool = False

    def supports_media_type(self, media_type: str) -> bool:
        """Return True if this model supports the given media type."""
        modes = self.transport_order.get(media_type)
        if modes is None:
            return False
        return modes != [MediaTransportMode.unsupported]

    def get_transport_order(self, media_type: str) -> list[MediaTransportMode]:
        """Return the transport preference order for a media type."""
        return self.transport_order.get(media_type, [MediaTransportMode.unsupported])


# ---------------------------------------------------------------------------
# Default capability matrix
# ---------------------------------------------------------------------------

_DEFAULT_CAPABILITIES: list[ModelMediaCapabilities] = [
    # OpenAI vision-capable models
    ModelMediaCapabilities(
        provider="openai",
        model_pattern="gpt-4o*",
        transport_order={
            "image": [MediaTransportMode.remote_url, MediaTransportMode.inline_bytes],
        },
        accepts_external_urls=True,
        supports_provider_file=True,
        can_convert_internal_to_remote=True,
    ),
    ModelMediaCapabilities(
        provider="openai",
        model_pattern="gpt-4-vision*",
        transport_order={
            "image": [MediaTransportMode.remote_url, MediaTransportMode.inline_bytes],
        },
        accepts_external_urls=True,
        supports_provider_file=True,
        can_convert_internal_to_remote=True,
    ),
    ModelMediaCapabilities(
        provider="openai",
        model_pattern="o1*",
        transport_order={
            "image": [MediaTransportMode.remote_url, MediaTransportMode.inline_bytes],
        },
        accepts_external_urls=True,
        supports_provider_file=True,
        can_convert_internal_to_remote=True,
    ),
    ModelMediaCapabilities(
        provider="openai",
        model_pattern="o3*",
        transport_order={
            "image": [MediaTransportMode.remote_url, MediaTransportMode.inline_bytes],
        },
        accepts_external_urls=True,
        supports_provider_file=True,
        can_convert_internal_to_remote=True,
    ),
    ModelMediaCapabilities(
        provider="openai",
        model_pattern="gpt-4o-mini*",
        transport_order={
            "image": [MediaTransportMode.remote_url, MediaTransportMode.inline_bytes],
        },
        accepts_external_urls=True,
        supports_provider_file=True,
        can_convert_internal_to_remote=True,
    ),
    # Google GenAI vision-capable models
    ModelMediaCapabilities(
        provider="google",
        model_pattern="gemini-*",
        transport_order={
            "image": [MediaTransportMode.provider_file, MediaTransportMode.inline_bytes],
        },
        accepts_external_urls=False,
        supports_provider_file=True,
        can_convert_internal_to_remote=False,
    ),
    # Text-only models (any provider)
    ModelMediaCapabilities(
        provider="openai",
        model_pattern="gpt-4",
        transport_order={
            "image": [MediaTransportMode.unsupported],
        },
        accepts_external_urls=False,
        supports_provider_file=False,
        can_convert_internal_to_remote=False,
    ),
    ModelMediaCapabilities(
        provider="openai",
        model_pattern="gpt-3.5-turbo*",
        transport_order={
            "image": [MediaTransportMode.unsupported],
        },
        accepts_external_urls=False,
        supports_provider_file=False,
        can_convert_internal_to_remote=False,
    ),
]


def _match_model(pattern: str, model: str) -> bool:
    """Match a model name against a glob-style pattern."""
    return fnmatch.fnmatch(model, pattern)


def get_capabilities(provider: str, model: str) -> ModelMediaCapabilities:
    """Look up media capabilities for a provider/model combination.

    Args:
        provider: Provider identifier (e.g. "openai", "google").
        model: Model name (e.g. "gpt-4o", "gemini-1.5-pro").

    Returns:
        The matching ``ModelMediaCapabilities`` entry.  If no explicit
        entry matches, returns a default entry that treats all media as
        ``unsupported`` and logs a warning.
    """
    for cap in _DEFAULT_CAPABILITIES:
        if cap.provider == provider and _match_model(cap.model_pattern, model):
            return cap

    logger.warning(
        "No media capabilities found for provider=%r, model=%r. "
        "Treating all media as unsupported. Add an entry to the "
        "capability matrix if this model supports media.",
        provider,
        model,
    )

    return ModelMediaCapabilities(
        provider=provider,
        model_pattern=model,
        transport_order={
            "image": [MediaTransportMode.unsupported],
        },
        accepts_external_urls=False,
        supports_provider_file=False,
        can_convert_internal_to_remote=False,
    )
