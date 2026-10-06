"""Agent-level memory support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agentflow.core.graph.tool_node import ToolNode


if TYPE_CHECKING:
    from agentflow.storage.store.memory_config import MemoryConfig


logger = logging.getLogger("agentflow.agent")


class AgentMemoryMixin:
    """Memory registration helpers for Agent."""

    _memory_config: MemoryConfig | None
    _memory_integration: Any | None
    _memory_prompt: dict[str, Any] | None
    _tool_node: ToolNode | None

    def _setup_memory(self, memory: MemoryConfig | None) -> None:
        """Initialize agent-level memory from ``MemoryConfig``."""
        self._memory_config = None
        self._memory_integration = None
        self._memory_prompt = None

        if memory is None:
            return

        from agentflow.storage.store.long_term_memory import (
            MemoryIntegration,
            get_agent_memory_system_prompt,
        )
        from agentflow.storage.store.memory_config import MemoryConfig

        if not isinstance(memory, MemoryConfig):
            raise TypeError(f"Expected MemoryConfig, got {type(memory)}")

        self._memory_config = memory

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
                raise RuntimeError(
                    "Memory requires an existing ToolNode when model-facing memory tools enabled"
                    "Provide a ToolNode to the Agent or register the memory tools manually."
                )

        if self._tool_node is not None:
            for memory_tool in memory_tools:
                self._tool_node.add_tool(memory_tool)

        default_store = (
            memory.store
            or (memory.user_memory.store if memory.user_memory else None)
            or (memory.agent_memory.store if memory.agent_memory else None)
        )
        if default_store is not None:
            self._memory_integration = MemoryIntegration(
                store=default_store,
                retrieval_mode=memory.retrieval_mode,
                limit=memory.limit,
                score_threshold=memory.score_threshold,
                max_tokens=memory.max_tokens,
            )

        logger.info(
            "Memory enabled: user=%s agent=%s",
            bool(memory.user_memory and memory.user_memory.enabled),
            bool(memory.agent_memory and memory.agent_memory.enabled),
        )

    async def _build_memory_prompts(
        self,
        state: Any,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Load and format memory context for preload mode."""
        from agentflow.prebuilt.tools.memory import (
            _memory_scope_config,
            _memory_scope_limit,
            _memory_scope_score_threshold,
            _memory_scope_store,
        )
        from agentflow.storage.store.long_term_memory import (
            ReadMode,
            _format_search_results,
            _strip_thread_id,
            _validate_memory_type,
        )

        memory = getattr(self, "_memory_config", None)
        if memory is None or memory.retrieval_mode != ReadMode.PRELOAD:
            return []

        query = self._latest_user_memory_query(state)
        if not query:
            return []

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

            search_config = _memory_scope_config(
                config,
                memory,
                scope_config,
                scope=scope_name,
            )
            if scope_name == "user":
                search_config = _strip_thread_id(search_config)

            try:
                results = await store.asearch(
                    search_config,
                    query,
                    memory_type=_validate_memory_type(scope_config.memory_type),
                    category=scope_config.category,
                    limit=_memory_scope_limit(memory, scope_config, None),
                    score_threshold=_memory_scope_score_threshold(memory, scope_config),
                    **({"max_tokens": memory.max_tokens} if memory.max_tokens else {}),
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
            return []

        return [
            {
                "role": "system",
                "content": "[Long-term Memory Context]\n" + "\n\n".join(sections),
            }
        ]

    @staticmethod
    def _latest_user_memory_query(state: Any) -> str:
        for msg in reversed(getattr(state, "context", []) or []):
            if getattr(msg, "role", None) == "user":
                text = msg.text() if hasattr(msg, "text") else str(msg)
                return text.strip()
        return ""
