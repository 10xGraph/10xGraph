"""Graph setup shared by the prebuilt agents: graph factory, tool node and ``compile``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tenxgraph.core.graph.compiled_graph import CompiledGraph
from tenxgraph.core.graph.state_graph import StateGraph
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.storage.checkpointer.base_checkpointer import BaseCheckpointer
from tenxgraph.storage.media.storage.base import BaseMediaStore
from tenxgraph.storage.media.url_policy import MediaFetchPolicy
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.utils.callbacks import CallbackManager


class _PrebuiltGraph[StateT: AgentState]:
    """Mixin for agents that hold ``_state``, ``_context_manager``, ``_publisher``,
    ``_id_generator``, ``_container`` and ``_graph`` and wire the graph in ``_configure_graph``.
    """

    _graph: Any

    def _configure_graph(self) -> None:
        raise NotImplementedError

    def _new_graph(self) -> StateGraph[StateT]:
        return StateGraph[StateT](
            state=self._state,  # type: ignore[attr-defined]
            context_manager=self._context_manager,  # type: ignore[attr-defined]
            publisher=self._publisher,  # type: ignore[attr-defined]
            id_generator=self._id_generator,  # type: ignore[attr-defined]
            container=self._container,  # type: ignore[attr-defined]
        )

    @staticmethod
    def _build_tool_node(
        *,
        tools: list[Callable],
        client: Any,
        pass_user_info_to_mcp: bool,
        mcp_auth: Any = None,
    ) -> ToolNode | None:
        if not tools and client is None:
            return None
        return ToolNode(
            tools,
            client=client,
            pass_user_info_to_mcp=pass_user_info_to_mcp,
            mcp_auth=mcp_auth,
        )

    def compile(
        self,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager | None = None,
        media_store: BaseMediaStore | None = None,
        shutdown_timeout: float = 30.0,
        media_policy: MediaFetchPolicy | None = None,
    ) -> CompiledGraph:
        """Wire the graph and return a :class:`~tenxgraph.core.graph.CompiledGraph`.

        Args:
            checkpointer: Persistence backend for state snapshots.
            store: Long-term key-value store.
            interrupt_before: Node names to pause execution before.
            interrupt_after: Node names to pause execution after.
            callback_manager: Callback hooks for observability.
            media_store: Media/file storage backend.
            shutdown_timeout: Graceful-shutdown timeout in seconds.
            media_policy: Which external media URLs the server may download.

        Returns:
            A compiled, ready-to-invoke graph.
        """
        self._configure_graph()
        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
            media_store=media_store,
            shutdown_timeout=shutdown_timeout,
            media_policy=media_policy,
        )
