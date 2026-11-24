"""Configuration models for agent-level long-term memory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tenxgraph.core.authz import Reach, get_identity
from tenxgraph.storage.store.base_store import BaseStore, system_user_id

from .long_term_memory import ReadMode, _validate_memory_type
from .store_schema import RetrievalStrategy


def _non_blank(value: str, message: str) -> str:
    """Return ``value`` stripped, or raise ``ValueError(message)`` when nothing is left."""
    value = value.strip()
    if not value:
        raise ValueError(message)
    return value


def _as_strategy(value: RetrievalStrategy | str | None) -> RetrievalStrategy | None:
    return None if value is None else RetrievalStrategy(value)


class _MemoryScopeConfig(BaseModel):
    """Shared settings for a memory scope.

    ``collection`` names the vector-store collection for this scope: a fixed name, or a
    function ``(run_config) -> name``. Precedence for the collection a memory operation uses:

    1. This ``collection`` field (resolved with the run config when it is a function). It
       always wins, so a ``collection`` in a run config can never override it.
    2. A ``collection`` already in the merged config (``MemoryConfig.config``, the scope's
       ``config``, or a run config built by the developer's own code).
    3. The store default (``QdrantStore(collection=...)``).

    A function must read the authenticated identity from the run config (for example
    ``cfg["user"]``, set by the auth backend, or ``cfg["user_id"]``), never graph state or any
    client-chosen field. One collection per company::

        UserMemoryConfig(collection=lambda cfg: f"company_{cfg['user']['tenant_id']}")

    A function must return a non-empty ``str``; otherwise ``ValueError`` is raised when the
    scope's config is built.

    ``memory_type`` and ``category`` are the defaults for what the scope writes. They do not
    narrow a search made through the model-facing tools: a search filters by type or category
    only when the model names one.
    """

    enabled: bool = True
    store: BaseStore | None = None
    memory_type: str = "episodic"
    category: str = "general"
    limit: int | None = Field(default=None, ge=1)
    score_threshold: float | None = Field(default=None, ge=0)
    retrieval_strategy: RetrievalStrategy | str | None = None
    collection: str | Callable[[dict[str, Any]], str] | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("collection")
    @classmethod
    def _validate_collection(
        cls, value: str | Callable[[dict[str, Any]], str] | None
    ) -> str | Callable[[dict[str, Any]], str] | None:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("collection must not be empty")
        return value

    def resolve_collection(self, run_config: dict[str, Any]) -> str | None:
        """Return this scope's collection name, or ``None`` when the scope sets none."""
        if self.collection is None:
            return None
        if isinstance(self.collection, str):
            return self.collection
        result = self.collection(run_config)
        if not isinstance(result, str) or not result.strip():
            raise ValueError(
                f"memory scope collection function must return a non-empty string, got {result!r}"
            )
        return result.strip()

    @field_validator("memory_type", "category")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        return _non_blank(value, "memory scope values must not be empty")

    @field_validator("memory_type")
    @classmethod
    def _check_memory_type(cls, value: str) -> str:
        return _validate_memory_type(value).value

    @field_validator("retrieval_strategy")
    @classmethod
    def _validate_retrieval_strategy(
        cls, value: RetrievalStrategy | str | None
    ) -> RetrievalStrategy | None:
        return _as_strategy(value)


class UserMemoryConfig(_MemoryScopeConfig):
    """User-scoped memory that the model may search and write.

    ``user_id`` fixes the memory owner for runs without an identity (``config["user_id"]``
    is replaced). With an identity in ``config["user"]`` the identity's ``user_id`` wins, so
    user memory stays each authenticated caller's own.
    """

    user_id: str | None = None

    @field_validator("user_id")
    @classmethod
    def _validate_user_id(cls, value: str | None) -> str | None:
        return None if value is None else _non_blank(value, "user_id must not be empty")


class AgentMemoryConfig(_MemoryScopeConfig):
    """Agent/app-scoped memory that the model may only search."""

    enabled: bool = False
    agent_id: str | None = None
    app_id: str | None = None

    @field_validator("agent_id", "app_id")
    @classmethod
    def _validate_scope_id(cls, value: str | None) -> str | None:
        return None if value is None else _non_blank(value, "agent/app ids must not be empty")


class MemoryConfig(BaseModel):
    """Primary public configuration object for ``Agent(..., memory=...)``."""

    store: BaseStore | None = None
    retrieval_mode: ReadMode = ReadMode.POSTLOAD
    limit: int = Field(default=5, ge=1)
    score_threshold: float = Field(default=0.0, ge=0)
    retrieval_strategy: RetrievalStrategy | str | None = None
    max_tokens: int | None = None
    inject_system_prompt: bool = True
    config: dict[str, Any] = Field(default_factory=dict)
    user_memory: UserMemoryConfig | None = Field(default_factory=UserMemoryConfig)
    agent_memory: AgentMemoryConfig | None = Field(default_factory=AgentMemoryConfig)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("retrieval_strategy")
    @classmethod
    def _validate_retrieval_strategy(
        cls, value: RetrievalStrategy | str | None
    ) -> RetrievalStrategy | None:
        return _as_strategy(value)

    def model_facing_tools(self) -> list[Any]:
        """Return the tools this memory config should expose to an Agent.

        POSTLOAD: the user tool (search, remember, forget) and the read-only agent tool.
        PRELOAD: the user tool with remember and forget only; the preload step does the search
        and agent memory is read-only. NO_RETRIEVAL: none.
        """
        from tenxgraph.prebuilt.tools.memory import make_agent_memory_tool, make_user_memory_tool

        user_enabled = bool(self.user_memory and self.user_memory.enabled)
        if self.retrieval_mode == ReadMode.PRELOAD:
            return [make_user_memory_tool(self, ("remember", "forget"))] if user_enabled else []
        if self.retrieval_mode != ReadMode.POSTLOAD:
            return []

        tools: list[Any] = []
        if user_enabled:
            tools.append(make_user_memory_tool(self))
        if self.agent_memory and self.agent_memory.enabled:
            tools.append(make_agent_memory_tool(self))
        return tools


# ---------------------------------------------------------------------------
# Scope helpers: used by the memory tool factories and by the preload path
# ---------------------------------------------------------------------------


def _memory_scope_store(
    memory_config: Any,
    scope_config: Any,
    injected_store: BaseStore | None,
) -> BaseStore | None:
    return scope_config.store or memory_config.store or injected_store


def _memory_scope_limit(memory_config: Any, scope_config: Any, limit: int | None) -> int:
    return limit or scope_config.limit or memory_config.limit


def _memory_scope_score_threshold(memory_config: Any, scope_config: Any) -> float | None:
    if scope_config.score_threshold is not None:
        return scope_config.score_threshold
    return memory_config.score_threshold


def _memory_scope_retrieval_strategy(memory_config: Any, scope_config: Any) -> Any:
    """Scope value wins over the MemoryConfig value; ``None`` = the store's default."""
    if scope_config.retrieval_strategy is not None:
        return scope_config.retrieval_strategy
    return memory_config.retrieval_strategy


def _pin_identity(cfg: dict[str, Any], owner: str | None) -> None:
    """Set the identity in ``cfg`` to reach ``own``, with ``owner`` as its user when given.

    Tenant values are kept. Without an identity, ``cfg`` is left as it is.
    """
    identity = get_identity(cfg)
    if identity is None:
        return
    update: dict[str, Any] = {"reach": Reach.OWN}
    if owner is not None:
        update["user_id"] = owner
    cfg["user"] = identity.model_copy(update=update)


def _memory_scope_config(
    runtime_config: dict[str, Any] | None,
    memory_config: Any,
    scope_config: Any,
    *,
    scope: Literal["user", "agent"],
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        **(memory_config.config or {}),
        **(scope_config.config or {}),
        **(runtime_config or {}),
    }

    # The scope's own collection always wins over any collection in the merged config,
    # including one from the run config. A function gets the run config (identity comes
    # from the authenticated config, never state).
    scope_collection = scope_config.resolve_collection(cfg)
    if scope_collection is not None:
        cfg["collection"] = scope_collection

    if scope == "user":
        user_id = getattr(scope_config, "user_id", None)
        if user_id:
            # Used only without an identity: with one, the identity is the owner (owner_id)
            # so a fixed id never shares one user's memory with every authenticated caller.
            cfg["user_id"] = user_id
        # User memory is always the caller's own, whatever their reach.
        _pin_identity(cfg, None)
        return cfg

    agent_id = getattr(scope_config, "agent_id", None)
    app_id = getattr(scope_config, "app_id", None)
    # Agent/app memory is shared data: it is owned by a reserved system owner named after the
    # agent (or app), never by the user of the current run. The user filter then keeps one
    # agent's shared memories apart from another's.
    owner = agent_id or app_id
    if not owner:
        raise ValueError("Agent memory needs agent_id or app_id to name its shared owner")
    cfg["user_id"] = system_user_id(owner)
    # A wider reach must not turn the shared-data query into one over other users' rows.
    _pin_identity(cfg, cfg["user_id"])
    if agent_id:
        cfg["agent_id"] = agent_id
        # Existing Qdrant-backed stores use ``thread_id`` as the secondary
        # scope field, so agent memory maps agent identity there intentionally.
        cfg["thread_id"] = agent_id
    if app_id:
        cfg["app_id"] = app_id
    return cfg
