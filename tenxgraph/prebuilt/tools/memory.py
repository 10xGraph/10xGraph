"""Model-facing memory tools for AgentFlow agents.

These are the tools that are registered with the agent's ToolNode and exposed
to the LLM.  Lower-level helpers (``MemoryIntegration``, preload-node factory,
read-mode constants) live in ``tenxgraph.storage.store.long_term_memory``.

Public API
----------
``memory_tool``
    Legacy LLM-callable tool used by the ``MemoryIntegration`` / manual graph
    wiring path.  Supports ``search``, ``store``, ``update``, ``delete``.

``make_user_memory_tool(memory_config)``
    Factory that returns the ``user_memory_tool`` used by
    ``Agent(..., memory=MemoryConfig(...))``.  Supports ``search`` and
    ``remember``.

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

from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.storage.store.long_term_memory import (
    _do_write,
    _flush_pending_writes,
    _format_search_results,
    _strip_thread_id,
    _validate_memory_type,
)
from tenxgraph.utils.background_task_manager import BackgroundTaskManager
from tenxgraph.utils.decorators import tool
from tenxgraph.utils.injection import fresh


logger = logging.getLogger("tenxgraph.prebuilt.tools.memory")


# ---------------------------------------------------------------------------
# Scope helpers
# ---------------------------------------------------------------------------
# Used by both the user/agent memory tool factories below AND by the preload
# path in AgentMemoryMixin._build_memory_prompts.
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

    if scope == "user":
        user_id = getattr(scope_config, "user_id", None)
        if user_id:
            cfg["user_id"] = user_id
        return cfg

    agent_id = getattr(scope_config, "agent_id", None)
    app_id = getattr(scope_config, "app_id", None)
    if agent_id:
        cfg["agent_id"] = agent_id
        # Existing Qdrant-backed stores use ``thread_id`` as the secondary
        # scope field, so agent memory maps agent identity there intentionally.
        cfg["thread_id"] = agent_id
    if app_id:
        cfg["app_id"] = app_id
    return cfg


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
) -> str:
    await _flush_pending_writes(task_manager)
    results = await store.asearch(
        config,
        query,
        memory_type=_validate_memory_type(memory_type or config.get("memory_type", "episodic")),
        category=category or config.get("category", "general"),
        limit=limit,
        score_threshold=score_threshold,
    )
    return json.dumps(_format_search_results(results))


# ---------------------------------------------------------------------------
# memory_tool - legacy LLM-callable tool (MemoryIntegration / manual wiring)
# ---------------------------------------------------------------------------


@tool(
    name="memory_tool",
    description=(
        "Search, store, update or delete long-term memories. "
        "Use action='search' with a query to recall relevant memories. "
        "Use action='store' with content and a short snake_case memory_key "
        "(e.g. 'user_name', 'favorite_language') to save new memories. "
        "The system uses memory_key to detect duplicates — if a memory with the "
        "same key already exists it will be updated automatically. "
        "Use action='delete' with memory_id to remove memories."
    ),
    tags=["memory", "long_term_memory"],
)
async def memory_tool(  # noqa: PLR0911, PLR0913
    action: Literal["search", "store", "update", "delete"] = "search",
    content: str = "",
    memory_key: str = "",
    memory_id: str = "",
    query: str = "",
    memory_type: str | None = None,
    category: str | None = None,
    metadata: dict[str, Any] | None = None,
    limit: int = 5,
    score_threshold: float | None = None,
    write_mode: Literal["merge", "replace"] = "merge",
    # Injectable params (excluded from LLM schema automatically)
    config: dict[str, Any] | None = None,
    store: BaseStore | None = Inject[BaseStore],
    task_manager: BackgroundTaskManager = Inject[BackgroundTaskManager],
) -> str:
    """Search, store, update, or delete long-term memories."""
    store = fresh(store)
    task_manager = fresh(task_manager)
    if store is None:
        return json.dumps({"error": "no memory store configured"})

    cfg = config or {}
    # Resolve memory_type and category from config if not explicitly provided.
    resolved_memory_type = memory_type or cfg.get("memory_type", "episodic")
    resolved_category = category or cfg.get("category", "general")
    mem_type = _validate_memory_type(resolved_memory_type)

    # Inject memory_key into metadata so _do_write can find it.
    if memory_key:
        metadata = {**(metadata or {}), "memory_key": memory_key}

    # --- Validation ---
    if action == "search" and not query:
        return json.dumps({"error": "query is required for search"})
    if action == "store" and not content:
        return json.dumps({"error": "content is required for store"})
    if action == "update" and not memory_id:
        return json.dumps({"error": "memory_id is required for update"})
    if action == "update" and not content:
        return json.dumps({"error": "content is required for update"})
    if action == "delete" and not memory_id:
        return json.dumps({"error": "memory_id is required for delete"})

    try:
        # --- Read ---
        if action == "search":
            # Flush any in-flight background writes so the search sees the
            # latest data (e.g. writes scheduled during a previous query).
            await _flush_pending_writes(task_manager)

            # Search across ALL threads for the user — long-term memory
            # is not scoped to a single conversation thread.
            results = await store.asearch(
                _strip_thread_id(cfg),
                query,
                memory_type=mem_type,
                limit=limit,
                score_threshold=score_threshold,
            )
            return json.dumps(_format_search_results(results))

        # --- Write (always async / background) ---
        write_coro = _do_write(
            store,
            cfg,
            action,
            content,
            memory_id,
            mem_type,
            resolved_category,
            metadata,
            write_mode,
        )
        try:
            task_manager.create_task(
                write_coro,
                name=f"memory_{action}_{memory_id or 'new'}",
            )
        except Exception:
            write_coro.close()
            raise
        return json.dumps({"status": "scheduled", "action": action})

    except Exception as e:
        logger.exception("memory_tool error (action=%s): %s", action, e)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Agent-level model-facing tools (Agent(memory=MemoryConfig(...)) path)
# ---------------------------------------------------------------------------


def make_user_memory_tool(memory_config: Any) -> Callable:
    """Create the user-scoped model-facing memory tool for an Agent."""
    user_config = memory_config.user_memory

    @tool(
        name="user_memory_tool",
        description=(
            "Search or remember user-scoped long-term memories. "
            "Use action='search' with text to recall durable user facts. "
            "Use action='remember' with text to save useful user facts or preferences. "
            "The model does not provide memory identifiers."
        ),
        tags=["memory", "long_term_memory", "user_memory"],
    )
    async def user_memory_tool(
        action: Literal["search", "remember"] = "search",
        text: str = "",
        memory_type: str | None = None,
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
        resolved_store = _memory_scope_store(memory_config, user_config, store)
        if resolved_store is None:
            return json.dumps({"error": "no user memory store configured"})

        cfg = _memory_scope_config(config, memory_config, user_config, scope="user")
        resolved_memory_type = memory_type or user_config.memory_type
        resolved_category = category or user_config.category
        resolved_limit = _memory_scope_limit(memory_config, user_config, limit)
        score_threshold = _memory_scope_score_threshold(memory_config, user_config)

        if not text:
            return json.dumps({"error": "text is required"})

        try:
            if action == "search":
                return await _search_scope_memory(
                    store=resolved_store,
                    config=_strip_thread_id(cfg),
                    query=text,
                    memory_type=resolved_memory_type,
                    category=resolved_category,
                    limit=resolved_limit,
                    score_threshold=score_threshold,
                    task_manager=task_manager,
                )

            metadata = _memory_tool_metadata("user")
            write_coro = _do_write(
                resolved_store,
                cfg,
                "store",
                text,
                "",
                _validate_memory_type(resolved_memory_type),
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

    user_memory_tool.__name__ = "user_memory_tool"
    user_memory_tool.__qualname__ = "user_memory_tool"
    return user_memory_tool


def make_agent_memory_tool(memory_config: Any) -> Callable:
    """Create the read-only agent-scoped model-facing memory tool for an Agent."""
    agent_config = memory_config.agent_memory

    @tool(
        name="agent_memory_tool",
        description=(
            "Search read-only agent/app-scoped long-term memories. "
            "This tool cannot write, update, or delete memory."
        ),
        tags=["memory", "long_term_memory", "agent_memory"],
    )
    async def agent_memory_tool(
        query: str,
        memory_type: str | None = None,
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
        if not query:
            return json.dumps({"error": "query is required"})

        cfg = _memory_scope_config(config, memory_config, agent_config, scope="agent")
        try:
            return await _search_scope_memory(
                store=resolved_store,
                config=cfg,
                query=query,
                memory_type=memory_type or agent_config.memory_type,
                category=category or agent_config.category,
                limit=_memory_scope_limit(memory_config, agent_config, limit),
                score_threshold=_memory_scope_score_threshold(memory_config, agent_config),
                task_manager=task_manager,
            )
        except Exception as e:
            logger.exception("agent_memory_tool error: %s", e)
            return json.dumps({"error": str(e)})

    agent_memory_tool.__name__ = "agent_memory_tool"
    agent_memory_tool.__qualname__ = "agent_memory_tool"
    return agent_memory_tool
