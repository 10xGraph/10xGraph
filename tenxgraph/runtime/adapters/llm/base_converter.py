from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from tenxgraph.core.state import AgentState, Message
    from tenxgraph.runtime.publisher.events import EventModel


logger = logging.getLogger("tenxgraph.adapters")

#: Message metadata key set when a provider stopped for a reason the caller should
#: know about. Value is one of the kinds returned by :func:`classify_stop_reason`.
STOP_ISSUE_KEY = "stop_issue"

_TRUNCATED = frozenset({"length", "max_tokens", "max_output_tokens"})
_CONTEXT_EXCEEDED = frozenset({"model_context_window_exceeded", "context_length_exceeded"})
_CONTENT_FILTER = frozenset(
    {
        "content_filter",
        "safety",
        "blocklist",
        "prohibited_content",
        "spii",
        "recitation",
        "image_safety",
        "image_prohibited_content",
        "image_recitation",
    }
)
_MALFORMED_TOOL_CALL = frozenset({"malformed_function_call", "unexpected_tool_call"})


def classify_stop_reason(reason: Any) -> str | None:
    """Map a provider stop/finish reason to a stop-issue kind, or None when normal.

    Kinds: ``truncated``, ``context_window_exceeded``, ``content_filter``,
    ``malformed_tool_call``. Accepts raw strings and enum reprs such as
    ``FinishReason.SAFETY``.
    """
    if reason is None:
        return None
    name = str(getattr(reason, "value", reason)).rsplit(".", 1)[-1].strip().lower()
    if name in _TRUNCATED:
        return "truncated"
    if name in _CONTEXT_EXCEEDED:
        return "context_window_exceeded"
    if name in _CONTENT_FILTER:
        return "content_filter"
    if name in _MALFORMED_TOOL_CALL:
        return "malformed_tool_call"
    return None


def mark_stop_issue(metadata: dict[str, Any], reason: Any, provider: str) -> str | None:
    """Record an abnormal stop reason in ``metadata`` and log it; return the kind."""
    kind = classify_stop_reason(reason)
    if kind:
        metadata[STOP_ISSUE_KEY] = kind
        logger.warning("%s stopped abnormally: %s (finish reason %s)", provider, kind, reason)
    return kind


class ConverterType(str, Enum):
    """Enumeration of supported converter types for LLM responses."""

    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM = "custom"


class BaseConverter(ABC):
    """
    Abstract base class for all LLM response converters.

    Subclasses should implement methods to convert standard and streaming
    LLM responses into agentflow's internal message/event formats.

    Attributes:
        state (AgentState | None): Optional agent state for context during conversion.
    """

    def __init__(self, state: AgentState | None = None) -> None:
        """
        Initialize the converter.

        Args:
            state (AgentState | None): Optional agent state for context during conversion.
        """
        self.state = state

    @abstractmethod
    async def convert_response(self, response: Any) -> Message:
        """
        Convert a standard agent response to a Message.

        Args:
            response (Any): The raw response from the LLM or agent.

        Returns:
            Message: The converted message object.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Conversion not implemented for this converter")

    @abstractmethod
    async def convert_streaming_response(
        self,
        config: dict,
        node_name: str,
        response: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[EventModel | Message]:
        """
        Convert a streaming agent response to an async generator of EventModel or Message.

        Args:
            config (dict): Node configuration parameters.
            node_name (str): Name of the node processing the response.
            response (Any): The raw streaming response from the LLM or agent.
            meta (dict | None): Optional metadata for conversion.

        Yields:
            EventModel | Message: Chunks of the converted streaming response.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Streaming not implemented for this converter")
