"""Exception raised when a model cannot accept a given media input.

This error is raised **before** the provider call when the capability
matrix determines that the target model does not support the media type
or no viable transport path exists.
"""

from __future__ import annotations

import logging
from typing import Any

from agentflow.storage.media.capabilities import MediaTransportMode


logger = logging.getLogger("agentflow.exceptions.media")


class UnsupportedMediaInputError(Exception):
    """Raised when a provider/model cannot accept the given media input.

    Attributes:
        provider: Provider identifier (e.g. "openai", "google").
        model: Model name (e.g. "gpt-4", "gemini-1.5-pro").
        media_type: Type of media (e.g. "image", "document", "audio").
        source_kind: How the media was provided
            (``"url"``, ``"file_id"``, ``"data"``, ``"internal_ref"``).
        transports_attempted: List of transport modes that were tried
            before giving up.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        media_type: str,
        source_kind: str,
        transports_attempted: list[MediaTransportMode] | None = None,
        message: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.media_type = media_type
        self.source_kind = source_kind
        self.transports_attempted = transports_attempted or []

        if message is None:
            message = self._build_message()

        self.message = message

        logger.error(
            "UnsupportedMediaInputError [%s/%s]: %s | source=%s, " "transports=%s",
            provider,
            model,
            message,
            source_kind,
            [t.value for t in self.transports_attempted],
        )

        super().__init__(message)

    def _build_message(self) -> str:
        """Build an actionable error message."""
        parts = [
            f"Model '{self.model}' from provider '{self.provider}' "
            f"does not support {self.media_type} inputs "
            f"(source: {self.source_kind}).",
        ]

        if self.transports_attempted:
            attempted = ", ".join(t.value for t in self.transports_attempted)
            parts.append(f"Transports attempted: {attempted}.")

        parts.append(self._suggestion())
        return " ".join(parts)

    def _suggestion(self) -> str:
        """Return a helpful suggestion based on the failure context."""
        if self.media_type == "image":
            return (
                "Use a vision-capable model (e.g. gpt-4o, gemini-1.5-pro) "
                "or remove media inputs."
            )
        if self.source_kind == "url":
            return (
                "Upload the file first and use a file_id reference, "
                "or switch to a model that supports external URLs."
            )
        return "Check the model documentation for supported media types " "and input formats."

    def to_dict(self) -> dict[str, Any]:
        """Convert the error to a structured dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "provider": self.provider,
            "model": self.model,
            "media_type": self.media_type,
            "source_kind": self.source_kind,
            "transports_attempted": [t.value for t in self.transports_attempted],
            "message": self.message,
        }

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return (
            f"UnsupportedMediaInputError("
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"media_type={self.media_type!r}, "
            f"source_kind={self.source_kind!r}, "
            f"transports_attempted={[t.value for t in self.transports_attempted]!r})"
        )
