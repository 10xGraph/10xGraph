"""Configuration models for agent-level long-term memory."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentflow.storage.store.base_store import BaseStore

from .long_term_memory import ReadMode


class _MemoryScopeConfig(BaseModel):
    """Shared settings for a memory scope."""

    enabled: bool = True
    store: BaseStore | None = None
    memory_type: str = "episodic"
    category: str = "general"
    limit: int | None = None
    score_threshold: float | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("memory_type", "category")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("memory scope values must not be empty")
        return value

    @field_validator("limit")
    @classmethod
    def _validate_limit(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("limit must be greater than zero")
        return value

    @field_validator("score_threshold")
    @classmethod
    def _validate_score_threshold(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("score_threshold must be non-negative")
        return value


class UserMemoryConfig(_MemoryScopeConfig):
    """User-scoped memory that the model may search and write."""

    user_id: str | None = None

    @field_validator("user_id")
    @classmethod
    def _validate_user_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("user_id must not be empty")
        return value


class AgentMemoryConfig(_MemoryScopeConfig):
    """Agent/app-scoped memory that the model may only search."""

    enabled: bool = False
    agent_id: str | None = None
    app_id: str | None = None

    @field_validator("agent_id", "app_id")
    @classmethod
    def _validate_scope_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("agent/app ids must not be empty")
        return value


class MemoryConfig(BaseModel):
    """Primary public configuration object for ``Agent(..., memory=...)``."""

    store: BaseStore | None = None
    retrieval_mode: ReadMode | str = ReadMode.POSTLOAD
    limit: int = 5
    score_threshold: float = 0.0
    max_tokens: int | None = None
    inject_system_prompt: bool = True
    config: dict[str, Any] = Field(default_factory=dict)
    user_memory: UserMemoryConfig | None = Field(default_factory=UserMemoryConfig)
    agent_memory: AgentMemoryConfig | None = Field(default_factory=AgentMemoryConfig)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("retrieval_mode")
    @classmethod
    def _validate_retrieval_mode(cls, value: ReadMode | str) -> ReadMode:
        if isinstance(value, ReadMode):
            return value
        return ReadMode(value)

    @field_validator("limit")
    @classmethod
    def _validate_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("limit must be greater than zero")
        return value

    @field_validator("score_threshold")
    @classmethod
    def _validate_score_threshold(cls, value: float) -> float:
        if value < 0:
            raise ValueError("score_threshold must be non-negative")
        return value

    def model_facing_tools(self) -> list[Any]:
        """Return the tools this memory config should expose to an Agent."""
        if self.retrieval_mode != ReadMode.POSTLOAD:
            return []

        from agentflow.prebuilt.tools.memory import make_agent_memory_tool, make_user_memory_tool

        tools: list[Any] = []
        if self.user_memory and self.user_memory.enabled:
            tools.append(make_user_memory_tool(self))
        if self.agent_memory and self.agent_memory.enabled:
            tools.append(make_agent_memory_tool(self))
        return tools
