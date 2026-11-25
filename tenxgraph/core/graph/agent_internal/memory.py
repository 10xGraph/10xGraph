"""Agent-level memory support."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.storage.store.base_store import warn_unsupported_strategy
from tenxgraph.utils.converter import wrap_memory_block


if TYPE_CHECKING:
    from tenxgraph.storage.store.memory_config import MemoryConfig


logger = logging.getLogger("tenxgraph.agent")

# Preload blocks cached per run on each Agent. Entries only matter while a run is in
# flight, so a small LRU bounds memory in long-lived servers.
_MEMORY_BLOCK_CACHE_SIZE = 256


def _warn_unsupported_strategies(memory: MemoryConfig) -> None:
    """Warn for each enabled scope whose strategy its known store cannot run."""
    for scope in (memory.user_memory, memory.agent_memory):
        if scope is None or not scope.enabled:
            continue
        store = scope.store or memory.store
        if store is not None:
            warn_unsupported_strategy(store, scope.retrieval_strategy or memory.retrieval_strategy)


def _refuse_unsupported_provenance(memory: MemoryConfig) -> None:
    """Raise when an enabled scope sets ``origins`` or ``ttl`` on a known store that would
    ignore them. A store injected at run time is checked on each memory call instead."""
    from tenxgraph.storage.store.memory_config import _check_provenance_support

    for scope in (memory.user_memory, memory.agent_memory):
        if scope is None or not scope.enabled:
            continue
        store = scope.store or memory.store
        if store is not None:
            _check_provenance_support(store, memory, scope)


class AgentMemoryMixin:
    """Memory registration helpers for Agent."""

    _memory_config: MemoryConfig | None
    _memory_prompt: dict[str, Any] | None
    _memory_block_cache: OrderedDict[tuple[str, str], str | None]
    _tool_node: ToolNode | None

    def _setup_memory(self, memory: MemoryConfig | None) -> None:
        """Initialize agent-level memory from ``MemoryConfig``."""
        self._memory_config = None
        self._memory_prompt = None
        self._memory_block_cache = OrderedDict()

        if memory is None:
            return

        from tenxgraph.storage.store.long_term_memory import (
            get_agent_memory_system_prompt,
        )
        from tenxgraph.storage.store.memory_config import MemoryConfig

        if not isinstance(memory, MemoryConfig):
            raise TypeError(f"Expected MemoryConfig, got {type(memory)}")

        agent_memory = memory.agent_memory
        if (
            agent_memory
            and agent_memory.enabled
            and not (agent_memory.agent_id or agent_memory.app_id)
        ):
            raise ValueError(
                "MemoryConfig.agent_memory is enabled but has neither agent_id nor app_id; "
                "set one so shared agent memory has its own owner"
            )

        _refuse_unsupported_provenance(memory)
        self._memory_config = memory
        _warn_unsupported_strategies(memory)

        if memory.inject_system_prompt:
            self._memory_prompt = {
                "role": "system",
                "content": get_agent_memory_system_prompt(memory),
            }
            self.system_prompt.append(self._memory_prompt)

        memory_tools = memory.model_facing_tools()
        if memory_tools and self._tool_node is None:
            if getattr(self, "tool_node_name", None) is not None:
                # Named ToolNode resolved at execution time via InjectQ; queue
                # these tools so _resolve_tools() registers them on first call.
                extra = getattr(self, "_extra_tools", None)
                if extra is None:
                    self._extra_tools = list(memory_tools)
                else:
                    extra.extend(memory_tools)
            else:
                mode = getattr(memory.retrieval_mode, "value", memory.retrieval_mode)
                raise RuntimeError(
                    f"Memory requires an existing ToolNode: retrieval_mode '{mode}' exposes "
                    f"model-facing memory tools ({', '.join(t.__name__ for t in memory_tools)}). "
                    "Provide a ToolNode to the Agent or register the memory tools manually."
                )

        if self._tool_node is not None:
            for memory_tool in memory_tools:
                self._tool_node.add_tool(memory_tool)

        logger.info(
            "Memory enabled: user=%s agent=%s",
            bool(memory.user_memory and memory.user_memory.enabled),
            bool(memory.agent_memory and memory.agent_memory.enabled),
        )

    async def _build_memory_block(
        self,
        state: Any,
        config: dict[str, Any],
    ) -> str | None:
        """Return the preload memory block for this model call, searching once per run.

        A run is one ``invoke``/``stream`` call; ``CompiledGraph._prepare_config`` gives it a
        ``run_id`` that every node of that run sees. The finished block (or ``None`` for an
        empty search) is cached under ``(run_id, latest user message)``, so later model calls
        of a tool loop reuse the same text and the same random tag, which keeps the prompt
        cache warm. A new invocation has a new ``run_id`` and searches again; a different user
        message under a reused ``run_id`` also searches again. Without a ``run_id`` (the agent
        called outside a compiled graph) nothing is cached. The cache is per Agent and
        bounded; it holds request text only and is never saved to state or the checkpointer.
        """
        memory = getattr(self, "_memory_config", None)
        run_id = (config or {}).get("run_id")
        if memory is None or not run_id:
            return await self._search_memory_block(state, config)

        key = (str(run_id), self._latest_user_memory_query(state))
        cache = self._memory_block_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        block = await self._search_memory_block(state, config)
        cache[key] = block
        while len(cache) > _MEMORY_BLOCK_CACHE_SIZE:
            cache.popitem(last=False)
        return block

    async def _search_memory_block(
        self,
        state: Any,
        config: dict[str, Any],
    ) -> str | None:
        """Search the stores for preload mode and return the memories as a fenced text block.

        The block is user-level data, not a system prompt: retrieved memory can hold text
        written from untrusted sources, so it must not carry system rank. The caller adds it
        to the outgoing request only (see ``inject_memory_block``); it is never saved.
        Returns ``None`` when there is nothing to inject.
        """
        from tenxgraph.storage.store.long_term_memory import (
            ReadMode,
            _format_search_results,
            _strip_thread_id,
        )
        from tenxgraph.storage.store.memory_config import (
            _check_provenance_support,
            _memory_scope_config,
            _memory_scope_limit,
            _memory_scope_origins,
            _memory_scope_retrieval_strategy,
            _memory_scope_score_threshold,
            _memory_scope_store,
        )

        memory = getattr(self, "_memory_config", None)
        if memory is None or memory.retrieval_mode != ReadMode.PRELOAD:
            return None

        query = self._latest_user_memory_query(state)
        if not query:
            return None

        sections: list[str] = []
        scopes = [
            ("User memory", "user", memory.user_memory),
            ("Agent memory", "agent", memory.agent_memory),
        ]
        for label, scope_name, scope_config in scopes:
            if scope_config is None or not scope_config.enabled:
                continue
            store = _memory_scope_store(memory, scope_config, None)
            if store is None:
                continue
            strategy = _memory_scope_retrieval_strategy(memory, scope_config)
            warn_unsupported_strategy(store, strategy)

            search_config = _memory_scope_config(
                config,
                memory,
                scope_config,
                scope=scope_name,
            )
            if scope_name == "user":
                search_config = _strip_thread_id(search_config)

            origins = _memory_scope_origins(memory, scope_config)
            try:
                _check_provenance_support(store, memory, scope_config)
                results = await store.asearch(
                    search_config,
                    query,
                    # Every type and category: the scope values are write defaults only.
                    limit=_memory_scope_limit(memory, scope_config, None),
                    score_threshold=_memory_scope_score_threshold(memory, scope_config),
                    retrieval_strategy=strategy,
                    **({"max_tokens": memory.max_tokens} if memory.max_tokens else {}),
                    **({"origins": origins} if origins is not None else {}),
                )
            except Exception:
                logger.exception("Memory preload search failed for %s", scope_name)
                continue

            formatted = _format_search_results(results)
            if not formatted:
                continue
            lines = [f"- {item['content']} (relevance: {item['score']})" for item in formatted]
            sections.append(f"{label}:\n" + "\n".join(lines))

        if not sections:
            return None

        return wrap_memory_block("\n\n".join(sections))

    @staticmethod
    def _latest_user_memory_query(state: Any) -> str:
        for msg in reversed(getattr(state, "context", []) or []):
            if getattr(msg, "role", None) == "user":
                text = msg.text() if hasattr(msg, "text") else str(msg)
                return text.strip()
        return ""
