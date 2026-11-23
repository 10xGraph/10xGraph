"""Model-facing memory tools for AgentFlow agents.

These are the tools that are registered with the agent's ToolNode and exposed
to the LLM.  Lower-level helpers (read-mode constants, write dedup) live in
``tenxgraph.storage.store.long_term_memory``.

Public API
----------
``make_user_memory_tool(memory_config, actions=...)``
    Factory that returns the ``user_memory_tool`` used by
    ``Agent(..., memory=MemoryConfig(...))``.  Supports ``search``,
    ``remember`` and ``forget``, or the subset given in ``actions``.

``make_agent_memory_tool(memory_config)``
    Factory that returns the read-only ``agent_memory_tool`` used by
    ``Agent(..., memory=MemoryConfig(...))``.  Supports ``search`` only.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Literal

from injectq import Inject

from tenxgraph.storage.store.base_store import (
    BaseStore,
    warn_unsupported_strategy,
)
from tenxgraph.storage.store.long_term_memory import (
    _do_write,
    _flush_pending_writes,
    _format_search_results,
    _strip_thread_id,
    _validate_memory_type,
)
from tenxgraph.storage.store.memory_config import (
    _memory_scope_config,
    _memory_scope_limit,
    _memory_scope_retrieval_strategy,
    _memory_scope_score_threshold,
    _memory_scope_store,
    _pin_identity,  # noqa: F401
)
from tenxgraph.storage.store.store_schema import MemoryTypeName
from tenxgraph.utils.background_task_manager import BackgroundTaskManager
from tenxgraph.utils.decorators import tool
from tenxgraph.utils.injection import fresh


logger = logging.getLogger("tenxgraph.prebuilt.tools.memory")

# Longest ``memory_key`` the model may give; a longer one is an error, never truncated.
_MAX_MEMORY_KEY_LENGTH = 128

UserMemoryAction = Literal["search", "remember", "forget"]
_USER_MEMORY_ACTIONS: tuple[str, ...] = ("search", "remember", "forget")

# One description line per action; the tool description lists only the allowed ones.
_USER_ACTION_HELP = {
    "search": "Use action='search' with text to recall durable user facts. ",
    "remember": (
        "Use action='remember' with text to save useful user facts or preferences. "
        "For a fact that can change, also pass a short stable memory_key "
        "(for example 'package_manager' or 'editor'); remembering again with the same "
        "key replaces the old fact. "
    ),
    "forget": "Use action='forget' with the memory_key when the user asks to forget that fact. ",
}


# ---------------------------------------------------------------------------
# Scope helpers
# ---------------------------------------------------------------------------
# The scope helpers live in ``tenxgraph.storage.store.memory_config`` next to
# ``MemoryConfig``; they are re-imported here under their old names.
# ---------------------------------------------------------------------------


def _memory_tool_metadata(scope: Literal["user", "agent"]) -> dict[str, Any]:
    return {
        "source": f"{scope}_memory_tool",
        "scope": scope,
    }


async def _search_scope_memory(
    *,
    store: BaseStore,
    config: dict[str, Any],
    query: str,
    memory_type: str | None,
    category: str | None,
    limit: int,
    score_threshold: float | None,
    task_manager: BackgroundTaskManager | None,
    retrieval_strategy: Any = None,
) -> str:
    await _flush_pending_writes(task_manager)
    results = await store.asearch(
        config,
        query,
        memory_type=_validate_memory_type(memory_type) if memory_type else None,
        category=category,
        limit=limit,
        score_threshold=score_threshold,
        retrieval_strategy=retrieval_strategy,
    )
    return json.dumps(_format_search_results(results))


# ---------------------------------------------------------------------------
# Agent-level model-facing tools (Agent(memory=MemoryConfig(...)) path)
# ---------------------------------------------------------------------------


def make_user_memory_tool(
    memory_config: Any,
    actions: tuple[str, ...] = _USER_MEMORY_ACTIONS,
) -> Callable:
    """Create the user-scoped model-facing memory tool for an Agent.

    ``actions`` is the subset of ``search``, ``remember`` and ``forget`` the model may call.
    Only these appear in the ``action`` enum and the description; another action returns an
    error. Preload mode uses ``("remember", "forget")``, since the preload step does the search.

    ``remember`` with a ``memory_key`` replaces the memory saved earlier under that key;
    ``forget`` deletes the memories with that key. The key is unique per
    ``(collection, user_id, memory_type, memory_key)``, so the same key under two memory
    types is two memories, and ``forget`` by key removes it under every type.
    """
    user_config = memory_config.user_memory
    allowed = tuple(a for a in _USER_MEMORY_ACTIONS if a in actions)
    if not allowed or len(allowed) != len(set(actions)):
        raise ValueError(f"actions must be a non-empty subset of {_USER_MEMORY_ACTIONS}")
    default_action = allowed[0]
    verbs = ", ".join(allowed[:-1]) + (" or " if len(allowed) > 1 else "") + allowed[-1]

    @tool(
        description=(
            f"{verbs.capitalize()} user-scoped long-term memories. "
            + "".join(_USER_ACTION_HELP[a] for a in allowed)
            + "Never pass memory ids; the memory layer manages them."
        ),
        tags=["memory", "long_term_memory", "user_memory"],
    )
    async def user_memory_tool(  # noqa: PLR0911
        action: UserMemoryAction = default_action,
        text: str = "",
        memory_key: str | None = None,
        memory_type: MemoryTypeName | None = None,
        category: str | None = None,
        limit: int | None = None,
        config: dict[str, Any] | None = None,
        store: BaseStore | None = Inject[BaseStore],
        task_manager: BackgroundTaskManager = Inject[BackgroundTaskManager],
    ) -> str:
        store = fresh(store)
        task_manager = fresh(task_manager)
        if user_config is None or not user_config.enabled:
            return json.dumps({"error": "user memory is disabled"})
        if action not in allowed:
            return json.dumps(
                {"error": f"action '{action}' is not available; use one of: {', '.join(allowed)}"}
            )
        resolved_store = _memory_scope_store(memory_config, user_config, store)
        if resolved_store is None:
            return json.dumps({"error": "no user memory store configured"})
        strategy = _memory_scope_retrieval_strategy(memory_config, user_config)
        warn_unsupported_strategy(resolved_store, strategy)

        cfg = _memory_scope_config(config, memory_config, user_config, scope="user")
        try:
            resolved_memory_type = _validate_memory_type(memory_type or user_config.memory_type)
        except ValueError as e:
            return json.dumps({"error": str(e)})
        resolved_category = category or user_config.category
        resolved_limit = _memory_scope_limit(memory_config, user_config, limit)
        score_threshold = _memory_scope_score_threshold(memory_config, user_config)

        key = (memory_key or "").strip()
        if len(key) > _MAX_MEMORY_KEY_LENGTH:
            return json.dumps(
                {"error": f"memory_key is longer than {_MAX_MEMORY_KEY_LENGTH} characters"}
            )
        if action == "forget" and not key:
            return json.dumps({"error": "memory_key is required for forget"})
        if action != "forget" and not text:
            return json.dumps({"error": "text is required"})

        try:
            if action == "forget":
                # Let a remember scheduled earlier in this turn land first, so the delete
                # does not run before it and leave the fact behind.
                await _flush_pending_writes(task_manager)
                # Always narrowed by the key: without filters this deletes every memory
                # of the user.
                await resolved_store.aforget_memory(
                    _strip_thread_id(cfg), filters={"memory_key": key}
                )
                return json.dumps({"status": "forgotten", "memory_key": key})

            if action == "search":
                return await _search_scope_memory(
                    store=resolved_store,
                    config=_strip_thread_id(cfg),
                    query=text,
                    memory_type=memory_type,
                    category=category,
                    limit=resolved_limit,
                    score_threshold=score_threshold,
                    task_manager=task_manager,
                    retrieval_strategy=strategy,
                )

            metadata = _memory_tool_metadata("user")
            if key:
                metadata["memory_key"] = key
            write_coro = _do_write(
                resolved_store,
                cfg,
                "store",
                text,
                "",
                resolved_memory_type,
                resolved_category,
                metadata,
                "merge",
            )
            try:
                task_manager.create_task(
                    write_coro,
                    name="user_memory_remember_new",
                )
            except Exception:
                write_coro.close()
                raise
            return json.dumps({"status": "scheduled", "action": "remember"})
        except Exception as e:
            logger.exception("user_memory_tool error (action=%s): %s", action, e)
            return json.dumps({"error": str(e)})

    # The schema enum lists only the allowed actions.
    user_memory_tool.__annotations__["action"] = Literal.__getitem__(allowed)
    user_memory_tool.__name__ = "user_memory_tool"
    user_memory_tool.__qualname__ = "user_memory_tool"
    return user_memory_tool


def make_agent_memory_tool(memory_config: Any) -> Callable:
    """Create the read-only agent-scoped model-facing memory tool for an Agent."""
    agent_config = memory_config.agent_memory

    @tool(
        description=(
            "Search read-only agent/app-scoped long-term memories. "
            "This tool cannot write, update, or delete memory."
        ),
        tags=["memory", "long_term_memory", "agent_memory"],
    )
    async def agent_memory_tool(
        query: str,
        memory_type: MemoryTypeName | None = None,
        category: str | None = None,
        limit: int | None = None,
        config: dict[str, Any] | None = None,
        store: BaseStore | None = Inject[BaseStore],
        task_manager: BackgroundTaskManager = Inject[BackgroundTaskManager],
    ) -> str:
        store = fresh(store)
        task_manager = fresh(task_manager)
        if agent_config is None or not agent_config.enabled:
            return json.dumps({"error": "agent memory is disabled"})
        resolved_store = _memory_scope_store(memory_config, agent_config, store)
        if resolved_store is None:
            return json.dumps({"error": "no agent memory store configured"})
        strategy = _memory_scope_retrieval_strategy(memory_config, agent_config)
        warn_unsupported_strategy(resolved_store, strategy)
        if not query:
            return json.dumps({"error": "query is required"})

        cfg = _memory_scope_config(config, memory_config, agent_config, scope="agent")
        if memory_type:
            try:
                _validate_memory_type(memory_type)
            except ValueError as e:
                return json.dumps({"error": str(e)})
        try:
            return await _search_scope_memory(
                store=resolved_store,
                config=cfg,
                query=query,
                memory_type=memory_type,
                category=category,
                limit=_memory_scope_limit(memory_config, agent_config, limit),
                score_threshold=_memory_scope_score_threshold(memory_config, agent_config),
                task_manager=task_manager,
                retrieval_strategy=strategy,
            )
        except Exception as e:
            logger.exception("agent_memory_tool error: %s", e)
            return json.dumps({"error": str(e)})

    agent_memory_tool.__name__ = "agent_memory_tool"
    agent_memory_tool.__qualname__ = "agent_memory_tool"
    return agent_memory_tool
