"""
Stream chunk primitives for unified streaming data handling.

This module provides a unified StreamChunk class that can encapsulate different types
of streaming data (Messages, EventModels, etc.) in a type-safe manner.
This enables clean separation between conversation content and execution state while
providing a consistent interface for streaming consumers.

Classes:
    StreamChunk: Unified wrapper for streaming data with type discrimination.
"""

import enum
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .agent_state import AgentState
from .message import Message


class StreamEvent(enum.StrEnum):
    STATE = "state"
    MESSAGE = "message"
    ERROR = "error"
    UPDATES = "updates"


class StreamChunk(BaseModel):
    """
    Unified wrapper for different types of streaming data.

    This class provides a single interface for handling various streaming chunk types
    (messages, events, state updates, errors) with type-safe discrimination.

    Attributes:
        type: The type of streaming chunk.
        data: The actual chunk data (Message, EventModel, dict, etc.).
        metadata: Optional additional metadata for the chunk.
    """

    event: StreamEvent = StreamEvent.MESSAGE
    # data holders for different chunk types
    message: Message | None = None
    state: AgentState | None = None
    # Placeholder for other chunk types
    data: dict | None = None

    # Optional identifiers
    thread_id: str | None = None
    run_id: str | None = None
    # Optional metadata
    metadata: dict | None = None
    timestamp: float = Field(
        default_factory=datetime.now().timestamp,
        description="UNIX timestamp of when chunk was created",
    )

    model_config = ConfigDict(use_enum_values=True)
