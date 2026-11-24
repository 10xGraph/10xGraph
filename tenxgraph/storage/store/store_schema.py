from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from tenxgraph.core.state import Message
from tenxgraph.core.state.message import generate_id


def _generate_memory_id() -> str:
    """A memory id from the bound id generator, else a uuid4 string."""
    return str(generate_id())


def _has_reasoning(message: Message) -> bool:
    """True when the message holds a reasoning block with text, details or redacted data."""
    return any(
        block.type == "reasoning" and bool(block.summary or block.details or block.redacted_data)
        for block in message.content
    )


class RetrievalStrategy(StrEnum):
    """Memory retrieval strategies."""

    SIMILARITY = "similarity"  # Vector similarity search
    TEMPORAL = "temporal"  # Time-based retrieval
    RELEVANCE = "relevance"  # Relevance scoring
    HYBRID = "hybrid"  # Combined approaches
    TEXT = "text"  # Keyword / full-text search
    GRAPH_TRAVERSAL = "graph_traversal"  # Knowledge graph navigation


class DistanceMetric(StrEnum):
    """Supported distance metrics for vector similarity."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class MemoryType(StrEnum):
    """Types of memories that can be stored."""

    EPISODIC = "episodic"  # Conversation memories
    SEMANTIC = "semantic"  # Facts and knowledge
    PROCEDURAL = "procedural"  # How-to knowledge
    ENTITY = "entity"  # Entity-based memories
    RELATIONSHIP = "relationship"  # Entity relationships
    CUSTOM = "custom"  # Custom memory types
    DECLARATIVE = "declarative"  # Explicit facts and events


# Keep in sync with MemoryType (a test checks it); a Literal gives the tool schema an enum.
MemoryTypeName = Literal[
    "episodic",
    "semantic",
    "procedural",
    "entity",
    "relationship",
    "custom",
    "declarative",
]


class MemoryOrigin(StrEnum):
    """Who wrote a memory. A stored ``None`` means unknown (old rows, or a write without it)."""

    USER = "user"  # The user, through the app (the API writes this)
    AGENT = "agent"  # The model, through the memory tool
    TOOL = "tool"  # A tool that saved external content (developer code)
    SYSTEM = "system"  # The developer's own code, such as seeded agent memory


#: Origins a developer may treat as trusted: ``asearch(..., origins=TRUSTED_ORIGINS)``.
TRUSTED_ORIGINS: tuple[MemoryOrigin, ...] = (MemoryOrigin.USER, MemoryOrigin.SYSTEM)


def check_origin(value: Any) -> MemoryOrigin | None:
    """``value`` as a ``MemoryOrigin`` (``None`` stays ``None``); anything else raises."""
    if value is None:
        return None
    try:
        return MemoryOrigin(value)
    except ValueError:
        names = ", ".join(o.value for o in MemoryOrigin)
        raise ValueError(f"unknown origin {value!r}; use one of: {names}") from None


def check_expires_at(value: Any) -> datetime | None:
    """A timezone-aware ``expires_at`` in UTC; a naive datetime or another type raises."""
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise ValueError("expires_at must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("expires_at must be timezone-aware (for example datetime.now(UTC))")
    return value.astimezone(UTC)


def check_origins(origins: Any) -> tuple[list[str], bool] | None:
    """Normalize an ``origins`` read filter to ``(origin values, include unknown)``.

    ``None`` means no origin filter. Otherwise a non-empty list of ``MemoryOrigin`` values,
    where ``None`` in the list also keeps rows of unknown origin.
    """
    if origins is None:
        return None
    if isinstance(origins, str | bytes) or not isinstance(origins, Iterable):
        raise ValueError("origins must be a list of MemoryOrigin values (None for unknown)")
    items = list(origins)
    if not items:
        raise ValueError("origins must not be empty; leave it out to read every origin")
    values = sorted({check_origin(o).value for o in items if o is not None})  # type: ignore[union-attr]
    return values, any(o is None for o in items)


def read_origin(value: Any) -> MemoryOrigin | None:
    """A stored origin as ``MemoryOrigin``; an unknown stored value reads as ``None``."""
    try:
        return check_origin(value)
    except ValueError:
        return None


def is_live(expires_at: datetime | None, now: datetime) -> bool:
    """True when a record with ``expires_at`` is not expired at ``now``."""
    return expires_at is None or expires_at > now


def _check_vector(v: Any) -> Any:
    if v is not None and (
        not isinstance(v, list) or any(not isinstance(x, int | float) for x in v)
    ):
        raise ValueError("vector must be list[float] or None")
    return v


class MemorySearchResult(BaseModel):
    """Result from a memory search operation (Pydantic model)."""

    id: str = Field(default_factory=_generate_memory_id)
    content: str = Field(default="", description="Primary textual content of the memory")
    score: float = Field(default=0.0, ge=0.0, description="Similarity / relevance score")
    memory_type: MemoryType = Field(default=MemoryType.EPISODIC)
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] | None = Field(default=None)
    user_id: str | None = None
    thread_id: str | None = None
    timestamp: datetime | None = Field(default_factory=lambda: datetime.now(UTC))
    origin: MemoryOrigin | None = None
    expires_at: datetime | None = None

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v):
        return _check_vector(v)

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, v):
        return check_expires_at(v)


class MemoryRecord(BaseModel):
    """Comprehensive memory record for storage (Pydantic model)."""

    id: str = Field(default_factory=_generate_memory_id)
    content: str
    user_id: str | None = None
    thread_id: str | None = None
    memory_type: MemoryType = Field(default=MemoryType.EPISODIC)
    metadata: dict[str, Any] = Field(default_factory=dict)
    category: str = Field(default="general")
    vector: list[float] | None = None
    timestamp: datetime | None = Field(default_factory=lambda: datetime.now(UTC))
    origin: MemoryOrigin | None = None
    expires_at: datetime | None = None

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v):
        return _check_vector(v)

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, v):
        return check_expires_at(v)

    @classmethod
    def from_message(
        cls,
        message: Message,
        user_id: str | None = None,
        thread_id: str | None = None,
        vector: list[float] | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> "MemoryRecord":
        content = message.text()

        # Convert timestamp (float) to ISO format string
        timestamp_str = None
        if message.timestamp:
            timestamp_dt = datetime.fromtimestamp(message.timestamp, tz=UTC)
            timestamp_str = timestamp_dt.isoformat()

        metadata = {
            "role": message.role,
            "message_id": str(message.message_id),
            "timestamp": timestamp_str,
            "has_tool_calls": bool(message.tools_calls),
            "has_reasoning": _has_reasoning(message),
            "token_usage": message.usages.model_dump() if message.usages else None,
            **(additional_metadata or {}),
        }
        return cls(
            content=content,
            user_id=user_id,
            thread_id=thread_id,
            memory_type=MemoryType.EPISODIC,
            metadata=metadata,
            vector=vector,
        )
