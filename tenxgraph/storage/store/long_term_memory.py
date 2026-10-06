"""
Long-term memory integration for AgentFlow graphs.

Primary API
-----------
    **MemoryIntegration** - single configuration point for adding long-term
    memory to any agent graph. Users only need to specify a *store* and a
    *retrieval_mode* (``"no_retrieval"`` | ``"preload"`` | ``"postload"``).

Retrieval modes
~~~~~~~~~~~~~~~
    * **no_retrieval** (default): the LLM cannot read past memories but CAN
      write new ones via ``memory_tool`` (always available as a tool).
    * **preload**: relevant memories are retrieved from the vector store and
      injected as a system message *before* the LLM call.
    * **postload**: the LLM decides when to search memory by calling
      ``memory_tool(action="search", …)`` itself.

Write behaviour (all modes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * The LLM decides *what* to persist by calling ``memory_tool`` with
      ``action="store"`` / ``"update"`` / ``"delete"``.
    * Writes are executed **asynchronously in the background** so they never
      block the response.
    * Background writes are tracked by ``BackgroundTaskManager``, which
      flushes all pending writes on graph shutdown via ``aclose()``.
    * Write modes: ``merge`` (default) or ``replace``.

Quick start
~~~~~~~~~~~
    .. code-block:: python

        from agentflow.store import MemoryIntegration, QdrantStore, OpenAIEmbedding

        store = QdrantStore(embedding=OpenAIEmbedding(), path="./memory_data")
        memory = MemoryIntegration(store=store, retrieval_mode="preload")

        # Wire into your graph (one-liner):
        memory.wire(graph, entry_to="main")

        # Include the system prompt in your LLM node:
        system_prompt = memory.system_prompt

        # Register memory tools alongside your other tools:
        tool_node = ToolNode([my_tool, *memory.tools])

        app = graph.compile(store=store)

Lower-level helpers (advanced / backwards-compatible)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``memory_tool`` - the LLM-callable tool for CRUD on ``BaseStore``.
    * ``create_memory_preload_node`` - factory that returns a graph node.
    * ``get_memory_system_prompt`` - prompt fragment per retrieval mode.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from injectq import InjectQ

from agentflow.core.state import AgentState, Message
from agentflow.storage.store.base_store import BaseStore
from agentflow.storage.store.store_schema import MemorySearchResult, MemoryType
from agentflow.utils.background_task_manager import BackgroundTaskManager


if TYPE_CHECKING:
    from agentflow.core.graph.state_graph import StateGraph

logger = logging.getLogger("agentflow.store.long_term_memory")


_VALID_MEMORY_TYPES = {m.value for m in MemoryType}


class ReadMode(StrEnum):
    NO_RETRIEVAL = "no_retrieval"
    PRELOAD = "preload"
    POSTLOAD = "postload"


DEFAULT_READ_MODE = ReadMode.NO_RETRIEVAL


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_memory_type(value: str) -> MemoryType:
    if value in _VALID_MEMORY_TYPES:
        return MemoryType(value)
    return MemoryType.EPISODIC


def _strip_thread_id(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *cfg* with ``thread_id`` removed.

    Long-term memory is cross-thread: searches and deduplication checks must
    not be scoped to the current conversation thread.
    """
    return {k: v for k, v in cfg.items() if k != "thread_id"}


def _format_search_results(results: list[MemorySearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "id": r.id,
            "content": r.content,
            "score": round(r.score, 4),
            "memory_type": r.memory_type.value,
            "metadata": r.metadata,
        }
        for r in results
    ]


async def _flush_pending_writes(task_manager: BackgroundTaskManager | None = None) -> None:
    """Flush in-flight background memory writes before a read operation.

    When *task_manager* is provided (``memory_tool`` search path) it is used
    directly.  When called without an argument (``_preload_node`` context,
    where DI injection is not available) an ``InjectQ`` lookup is attempted
    as a best-effort fallback.
    """
    if task_manager is not None:
        if task_manager.get_task_count() > 0:
            logger.debug(
                "Waiting for %d background tasks before read…",
                task_manager.get_task_count(),
            )
            await task_manager.wait_for_all(timeout=15)
        return
    try:
        tm = InjectQ.get_instance().try_get(BackgroundTaskManager)
        if tm is not None and tm.get_task_count() > 0:
            logger.debug(
                "Waiting for %d background tasks before preload…",
                tm.get_task_count(),
            )
            await tm.wait_for_all(timeout=15)
    except Exception:
        logger.debug("Could not flush background tasks before preload", exc_info=True)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
# When the LLM provides a ``memory_key`` (a short snake_case topic label
# like ``"user_name"`` or ``"favorite_language"``), we use it for exact
# dedup: if a memory with the same key already exists for the user it is
# *updated* rather than duplicated.  This is independent of embedding
# similarity and thus works even when only the value changes (e.g. the
# user's name goes from "Atharv" to "Prashant").
#
# As a fallback (no key provided) we still try a high-similarity check.
# ---------------------------------------------------------------------------

_IDENTICAL_SCORE_THRESHOLD: float = 0.95


async def _find_duplicate_by_key(
    store: BaseStore,
    config: dict[str, Any],
    memory_key: str,
    mem_type: MemoryType,
) -> MemorySearchResult | None:
    """Find an existing memory with the same ``memory_key`` for this user.

    Uses a metadata filter — no embedding similarity needed.
    Config must already have thread_id stripped by the caller.
    """
    try:
        results = await store.asearch(
            config,
            memory_key,  # query text (needed for embedding, but filter does the work)
            memory_type=mem_type,
            limit=1,
            filters={"memory_key": memory_key},
        )
        return results[0] if results else None
    except Exception:
        logger.debug("Key-based duplicate check failed", exc_info=True)
        return None


async def _find_duplicate_by_similarity(
    store: BaseStore,
    config: dict[str, Any],
    content: str,
    mem_type: MemoryType,
    threshold: float = _IDENTICAL_SCORE_THRESHOLD,
) -> MemorySearchResult | None:
    """Fallback: find near-identical content via embedding similarity.

    Config must already have thread_id stripped by the caller.
    """
    try:
        results = await store.asearch(
            config,
            content,
            memory_type=mem_type,
            limit=1,
            score_threshold=threshold,
        )
        return results[0] if results else None
    except Exception:
        logger.debug("Similarity-based duplicate check failed", exc_info=True)
        return None


async def _do_write(
    store: BaseStore,
    config: dict[str, Any],
    action: str,
    content: str,
    memory_id: str,
    mem_type: MemoryType,
    category: str,
    metadata: dict[str, Any] | None,
    write_mode: str,
) -> dict[str, Any]:
    """Execute a write operation against the store.

    For ``action="store"`` an automatic **deduplication** check runs first:

    1. If ``memory_key`` is present in *metadata*, look for an existing
       memory with the same key (exact metadata filter).  If found →
       **update** the existing record.
    2. Otherwise, fall back to embedding similarity: if a near-identical
       memory exists (score ≥ 0.95) → **skip** the write.
    """
    # Strip thread_id once here — long-term memory is cross-thread;
    # neither dedup searches nor writes should be scoped to a single thread.
    cfg = _strip_thread_id(config)

    if action == "store":
        memory_key = (metadata or {}).get("memory_key", "")

        # --- Strategy 1: key-based dedup (reliable for topic changes) ---
        if memory_key:
            existing = await _find_duplicate_by_key(store, cfg, memory_key, mem_type)
            if existing:
                logger.info(
                    "Updating existing memory by key '%s' (id=%s)",
                    memory_key,
                    existing.id,
                )
                await store.aupdate(cfg, str(existing.id), content, metadata=metadata)
                return {
                    "status": "updated_existing",
                    "memory_id": str(existing.id),
                    "memory_key": memory_key,
                }

        # --- Strategy 2: similarity-based dedup (catch identical text) ---
        existing = await _find_duplicate_by_similarity(store, cfg, content, mem_type)
        if existing:
            logger.info(
                "Skipping duplicate memory (score=%.3f, id=%s)",
                existing.score,
                existing.id,
            )
            return {
                "status": "skipped_duplicate",
                "memory_id": str(existing.id),
                "score": round(existing.score, 4) if existing.score else None,
            }

        generated_id = await store.generate_framework_id()
        mid = await store.astore(
            cfg,
            content,
            memory_type=mem_type,
            category=category,
            metadata=metadata,
            memory_id=generated_id,
        )
        return {"status": "stored", "memory_id": str(mid)}

    if action == "update":
        if write_mode == "merge":
            existing = await store.aget(cfg, memory_id)
            merged = {
                **(existing.metadata if existing and existing.metadata else {}),
                **(metadata or {}),
            }
            await store.aupdate(cfg, memory_id, content, metadata=merged)
        else:
            await store.aupdate(cfg, memory_id, content, metadata=metadata, replace_metadata=True)
        return {"status": "updated", "memory_id": memory_id}

    if action == "delete":
        await store.adelete(cfg, memory_id)
        return {"status": "deleted", "memory_id": memory_id}

    return {"error": f"unknown write action: {action}"}


def get_agent_memory_system_prompt(memory_config: Any) -> str:
    """Build the system prompt fragment for ``Agent(memory=...)``."""
    lines: list[str] = ["[Long-term Memory]"]

    if memory_config.retrieval_mode == ReadMode.PRELOAD:
        lines.extend(
            [
                "Relevant long-term memories may be provided as system context before "
                "the user message.",
                "Use the provided memory context when it is relevant, but do not assume "
                "it is exhaustive.",
            ]
        )
        return "\n".join(lines)

    if memory_config.retrieval_mode != ReadMode.POSTLOAD:
        lines.append("Long-term memory retrieval tools are not available in this mode.")
        return "\n".join(lines)

    if memory_config.user_memory and memory_config.user_memory.enabled:
        lines.extend(
            [
                "You have access to user_memory_tool for user-specific long-term memory.",
                "- To recall durable user facts, call user_memory_tool with "
                "action='search' and text.",
                "- To save useful user facts or preferences, call user_memory_tool with "
                "action='remember' and text.",
                "- Do not invent memory identifiers; the memory layer manages identity "
                "and deduplication.",
            ]
        )

    if memory_config.agent_memory and memory_config.agent_memory.enabled:
        lines.extend(
            [
                "You have access to agent_memory_tool for read-only agent/app memory.",
                "- To recall agent-level context, call agent_memory_tool with query.",
                "- Agent memory is read-only in this agent flow; do not attempt to write, "
                "update, or delete it.",
            ]
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preload node factory
# ---------------------------------------------------------------------------

_DEFAULT_MEMORY_PROMPT = (
    "[Long-term Memory Context]\n"
    "The following memories were retrieved for the current conversation:\n"
    "{memories}\n"
    "Use this context to inform your response when relevant."
)


def _default_query_builder(state: AgentState) -> str:
    """Extract the latest user message text as the search query."""
    for msg in reversed(state.context):
        if msg.role == "user":
            return msg.text()
    return ""


def create_memory_preload_node(
    store: BaseStore,
    query_builder: Callable[[AgentState], str] | None = None,
    limit: int = 5,
    score_threshold: float = 0.0,
    memory_types: list[MemoryType] | None = None,
    system_prompt_template: str = _DEFAULT_MEMORY_PROMPT,
    max_tokens: int | None = None,
) -> Callable:
    """Factory returning a node function that preloads memory into state context.

    Wire into a graph before the LLM node:
        preload = create_memory_preload_node(store)
        graph.add_node("memory_preload", preload)
        graph.add_edge("memory_preload", "main")
        graph.set_entry_point("memory_preload")
    """
    _builder = query_builder or _default_query_builder

    async def _preload_node(state: AgentState, config: dict[str, Any]) -> list[Message]:
        # Flush any in-flight background writes so the preload search
        # always sees the latest persisted data.
        await _flush_pending_writes()

        query = _builder(state)
        if not query:
            return []

        search_kwargs: dict[str, Any] = {
            "limit": limit,
        }
        if score_threshold > 0:
            search_kwargs["score_threshold"] = score_threshold
        if max_tokens is not None:
            search_kwargs["max_tokens"] = max_tokens

        # Search across ALL threads for the user — long-term memory is not
        # scoped to a single conversation thread.
        search_config = _strip_thread_id(config)

        try:
            if memory_types:
                # Search each requested memory type and merge results.
                coros = [
                    store.asearch(search_config, query, memory_type=mt, **search_kwargs)
                    for mt in memory_types
                ]
                all_results = await asyncio.gather(*coros, return_exceptions=True)
                results = []
                for r in all_results:
                    if isinstance(r, BaseException):
                        logger.warning("Memory preload search for a type failed: %s", r)
                        continue
                    results.extend(r)
                # Re-sort by score (descending) and cap at the configured limit.
                results.sort(key=lambda r: r.score or 0.0, reverse=True)
                results = results[:limit]
            else:
                results = await store.asearch(search_config, query, **search_kwargs)
        except Exception:
            logger.exception("Memory preload search failed")
            return []

        if not results:
            return []

        lines = []
        for r in results:
            score_str = f" (relevance: {r.score:.2f})" if r.score else ""
            lines.append(f"- {r.content}{score_str}")
        memory_text = system_prompt_template.format(memories="\n".join(lines))

        return [Message.text_message(memory_text, role="system")]

    _preload_node.__name__ = "memory_preload"
    _preload_node.__qualname__ = "memory_preload"
    return _preload_node


# ---------------------------------------------------------------------------
# System prompt helpers
# ---------------------------------------------------------------------------


# Shared write instructions appended to every mode's prompt.
_WRITE_INSTRUCTIONS = (
    "\n\nYou have access to a memory_tool for writing to long-term memory.\n"
    "After processing each user message, decide whether any new information "
    "(facts, preferences, names, decisions) should be persisted.\n"
    "- To save or update facts, call memory_tool with action='store', "
    "content (the fact to remember), and a short snake_case memory_key "
    "that identifies the topic (e.g. 'user_name', 'favorite_language', "
    "'user_hobby'). The system uses memory_key to detect duplicates — "
    "if a memory with the same key already exists it is updated "
    "automatically, so you never create duplicates.\n"
    "- To remove outdated information, use action='delete' with the memory_id.\n"
    "Always use action='store' with a memory_key for new or changed "
    "information — do NOT use action='update' unless you have a specific "
    "memory_id from a previous search result.\n"
    "Writing is asynchronous — it will not slow down your response."
)


def get_memory_system_prompt(
    mode: Literal["no_retrieval", "preload", "postload"] = "no_retrieval",
) -> str:
    """Returns a system prompt fragment for the given read mode.

    All modes include write instructions because writing is always enabled
    and independent of the retrieval mode.
    Default mode is no_retrieval.
    """
    if mode == "no_retrieval":
        return (
            "You do NOT have access to read or search long-term memories. "
            "Do not attempt to recall information from previous sessions." + _WRITE_INSTRUCTIONS
        )

    if mode == "preload":
        return (
            "You have been provided with long-term memory context from previous "
            "interactions. Use it to personalize your responses when relevant. "
            "The memory context appears as system messages labeled "
            "'[Long-term Memory Context]'." + _WRITE_INSTRUCTIONS
        )

    if mode == "postload":
        return (
            "You have access to a memory_tool that can search, store, and "
            "delete long-term memories.\n"
            "- To recall relevant information, call memory_tool with action='search' "
            "and a descriptive query.\n"
            "- To save or update facts, call memory_tool with action='store', "
            "content (the fact), and a short snake_case memory_key that identifies "
            "the topic (e.g. 'user_name', 'favorite_language'). The system uses "
            "memory_key to detect duplicates — if a memory with the same key "
            "already exists it is updated automatically.\n"
            "- To remove outdated information, use action='delete' with the memory_id.\n"
            "Always use action='store' with a memory_key — do NOT use action='update' "
            "unless you have a specific memory_id from a previous search result.\n"
            "Only search memory when prior context would genuinely improve your response.\n"
            "Writing is asynchronous — it will not slow down your response."
        )

    return ""


# ---------------------------------------------------------------------------
# MemoryIntegration — unified public API
# ---------------------------------------------------------------------------


class MemoryIntegration:
    """Unified long-term memory integration for AgentFlow graphs.

    ``MemoryIntegration`` is the **single entry point** for adding long-term
    memory to any graph.  Callers only configure a *store* and a
    *retrieval_mode*; the class provides everything needed to wire memory
    into a ``StateGraph``.

    Parameters
    ----------
    store:
        A ``BaseStore`` instance (e.g. ``QdrantStore``, ``Mem0Store``).
    retrieval_mode:
        One of ``"no_retrieval"`` (default), ``"preload"``, ``"postload"``.
        Accepts both ``ReadMode`` enum values and plain strings.
    limit:
        Maximum number of memories to retrieve (preload / search).
    score_threshold:
        Minimum similarity score for retrieval (0.0 = no threshold).
    max_tokens:
        Optional token budget for retrieved memory context.
    query_builder:
        Custom function ``(AgentState) -> str`` to extract the search query
        from state.  Defaults to the latest user message text.
    preload_prompt_template:
        Custom Jinja-style template for the preload system message.
        Must contain ``{memories}`` placeholder.

    Examples
    --------
    Minimal preload setup::

        memory = MemoryIntegration(store=my_store, retrieval_mode="preload")
        memory.wire(graph, entry_to="main")
        tool_node = ToolNode([my_tool, *memory.tools])
        app = graph.compile(store=my_store)

    Postload (LLM-driven retrieval)::

        memory = MemoryIntegration(store=my_store, retrieval_mode="postload")
        memory.wire(graph, entry_to="main")
        tool_node = ToolNode([my_tool, *memory.tools])
        app = graph.compile(store=my_store)

    No retrieval (write-only)::

        memory = MemoryIntegration(store=my_store)
        # memory.retrieval_mode == ReadMode.NO_RETRIEVAL by default
        tool_node = ToolNode([my_tool, *memory.tools])
    """

    def __init__(
        self,
        store: BaseStore,
        retrieval_mode: ReadMode | str = ReadMode.NO_RETRIEVAL,
        limit: int = 5,
        score_threshold: float = 0.0,
        max_tokens: int | None = None,
        query_builder: Callable[[AgentState], str] | None = None,
        preload_prompt_template: str | None = None,
    ) -> None:
        if isinstance(retrieval_mode, str):
            retrieval_mode = ReadMode(retrieval_mode)
        self._store = store
        self._retrieval_mode = retrieval_mode
        self._limit = limit
        self._score_threshold = score_threshold
        self._max_tokens = max_tokens
        self._query_builder = query_builder
        self._preload_prompt_template = preload_prompt_template

        # Build preload node eagerly so .preload_node is stable.
        self._preload_node_fn: Callable | None = None
        if self._retrieval_mode == ReadMode.PRELOAD:
            self._preload_node_fn = create_memory_preload_node(
                store=self._store,
                query_builder=self._query_builder,
                limit=self._limit,
                score_threshold=self._score_threshold,
                system_prompt_template=(self._preload_prompt_template or _DEFAULT_MEMORY_PROMPT),
                max_tokens=self._max_tokens,
            )

    # -- Properties ---------------------------------------------------------

    @property
    def retrieval_mode(self) -> ReadMode:
        """The configured retrieval mode."""
        return self._retrieval_mode

    @property
    def store(self) -> BaseStore:
        """The underlying vector / memory store."""
        return self._store

    @property
    def system_prompt(self) -> str:
        """System prompt fragment for the configured retrieval mode.

        Include this in your LLM node's system prompts to enable
        memory awareness and write capability.
        """
        return get_memory_system_prompt(self._retrieval_mode.value)

    @property
    def tools(self) -> list[Callable]:
        """Memory tools to register with ``ToolNode``.

        All modes include ``memory_tool`` so the LLM can always decide to
        write.  In *postload* mode the LLM also uses it for reads.
        """
        # Lazy import avoids a circular dependency:
        # prebuilt.tools.memory → storage.store.long_term_memory
        from agentflow.prebuilt.tools.memory import memory_tool

        return [memory_tool]

    @property
    def preload_node(self) -> Callable | None:
        """Preload node function for *preload* mode, ``None`` otherwise.

        This is an async node function with signature
        ``(state, config) -> list[Message]``.
        """
        return self._preload_node_fn

    # -- Graph wiring -------------------------------------------------------

    def wire(
        self,
        graph: StateGraph,
        entry_to: str,
        preload_node_name: str = "memory_preload",
    ) -> None:
        """Wire memory into a ``StateGraph`` in one call.

        * **preload** mode → adds a preload node, sets it as entry point,
          and edges it to *entry_to*.
        * **no_retrieval / postload** → simply sets *entry_to* as the
          entry point.

        Parameters
        ----------
        graph:
            The ``StateGraph`` being built.
        entry_to:
            Name of the node that should receive control after any memory
            retrieval (typically your LLM node).
        preload_node_name:
            Name to use for the preload node (default ``"memory_preload"``).
        """
        if self._retrieval_mode == ReadMode.PRELOAD and self._preload_node_fn:
            graph.add_node(preload_node_name, self._preload_node_fn)
            graph.set_entry_point(preload_node_name)
            graph.add_edge(preload_node_name, entry_to)
        else:
            graph.set_entry_point(entry_to)
