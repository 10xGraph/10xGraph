"""AudioAgent -- prebuilt realtime (audio-to-audio) agent, React-style builder.

Mirrors :class:`~tenxgraph.prebuilt.agent.react.ReactAgent`'s construction surface but
wraps a :class:`~tenxgraph.core.realtime.live_agent.LiveAgent` as the graph root. The
compiled graph is driven by ``CompiledGraph.arealtime`` (a separate runtime), not
``invoke``/``stream``. No sub-agents / handoff are wired in v1 (a handoff tool is just a
tool, so the door stays open).
"""

from collections.abc import Callable, Iterable
from typing import Any

from tenxgraph.core.graph.compiled_graph import CompiledGraph
from tenxgraph.core.graph.state_graph import StateGraph
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.realtime.base import RealtimeClient, RealtimeConfig
from tenxgraph.core.realtime.live_agent import LiveAgent
from tenxgraph.core.skills.models import SkillConfig
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.base_context import BaseContextManager
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.storage.checkpointer.base_checkpointer import BaseCheckpointer
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.storage.store.memory_config import MemoryConfig
from tenxgraph.utils.callbacks import CallbackManager
from tenxgraph.utils.constants import END
from tenxgraph.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


class AudioAgent[StateT: AgentState]:
    """Build and compile a single realtime audio agent graph."""

    def __init__(  # noqa: PLR0913
        self,
        model: str,
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        publisher: BasePublisher | list[BasePublisher] | None = None,
        id_generator: BaseIDGenerator = DefaultIDGenerator(),
        container: Any | None = None,
        *,
        realtime_config: RealtimeConfig | None = None,
        system_prompt: list[dict[str, Any]] | None = None,
        tools: Iterable[Callable] | None = None,
        client: Any = None,
        pass_user_info_to_mcp: bool = False,
        skills: SkillConfig | None = None,
        memory: MemoryConfig | None = None,
        realtime_client_factory: Callable[[], RealtimeClient] | None = None,
        live_node_name: str = "LIVE",
        **agent_kwargs: Any,
    ) -> None:
        self._state = state
        self._context_manager = context_manager
        self._publisher = publisher
        self._id_generator = id_generator
        self._container = container
        self._live_node_name = live_node_name

        self._tool_node = self._build_tool_node(
            tools=list(tools or []),
            client=client,
            pass_user_info_to_mcp=pass_user_info_to_mcp,
        )

        self._agent = LiveAgent(
            model,
            realtime_config=realtime_config,
            system_prompt=system_prompt,
            tool_node=self._tool_node,
            skills=skills,
            memory=memory,
            realtime_client_factory=realtime_client_factory,
            **agent_kwargs,
        )
        self._graph: StateGraph[StateT] | None = None

    @staticmethod
    def _build_tool_node(
        *,
        tools: list[Callable],
        client: Any,
        pass_user_info_to_mcp: bool,
    ) -> ToolNode | None:
        if not tools and client is None:
            return None
        return ToolNode(tools, client=client, pass_user_info_to_mcp=pass_user_info_to_mcp)

    def _create_graph(self) -> StateGraph[StateT]:
        return StateGraph[StateT](
            state=self._state,
            context_manager=self._context_manager,
            publisher=self._publisher,
            id_generator=self._id_generator,
            container=self._container,
        )

    def _configure_graph(self) -> None:
        self._graph = self._create_graph()
        self._graph.add_node(self._live_node_name, self._agent)
        self._graph.set_entry_point(self._live_node_name)
        # The edge is never traversed in realtime (the live node owns the loop); it exists
        # only so the graph is well-formed for compile().
        self._graph.add_edge(self._live_node_name, END)

    def compile(
        self,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        callback_manager: CallbackManager | None = None,
        shutdown_timeout: float = 30.0,
    ) -> CompiledGraph:
        # No media_store: realtime media (images/video) is sent frame-by-frame straight to
        # the live model via the input queue (see LiveInputQueue.send_image); it is never
        # offloaded to or resolved from a media store, so the parameter would be dead here.
        self._configure_graph()

        if self._graph is None:  # pragma: no cover - _configure_graph always assigns
            raise RuntimeError("graph configuration failed")

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            callback_manager=callback_manager
            if callback_manager is not None
            else CallbackManager(),
            shutdown_timeout=shutdown_timeout,
        )
