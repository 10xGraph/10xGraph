from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from agentflow.core.state import AgentState, Message
    from agentflow.runtime.publisher.events import EventModel


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
