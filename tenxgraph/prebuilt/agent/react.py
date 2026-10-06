from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from injectq import InjectQ

from agentflow.core.graph.agent import Agent
from agentflow.core.graph.compiled_graph import CompiledGraph
from agentflow.core.graph.state_graph import StateGraph
from agentflow.core.graph.tool_node import ToolNode
from agentflow.core.skills.models import SkillConfig
from agentflow.core.state.agent_state import AgentState
from agentflow.core.state.base_context import BaseContextManager
from agentflow.core.state.message import Message
from agentflow.runtime.publisher.base_publisher import BasePublisher
from agentflow.storage.checkpointer.base_checkpointer import BaseCheckpointer
from agentflow.storage.media.config import MultimodalConfig
from agentflow.storage.media.storage.base import BaseMediaStore
from agentflow.storage.store.base_store import BaseStore
from agentflow.storage.store.memory_config import MemoryConfig
from agentflow.utils.callbacks import CallbackManager
from agentflow.utils.constants import END
from agentflow.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


StateT = TypeVar("StateT", bound=AgentState)


def _should_use_tools(state: AgentState) -> str:
    """Return "TOOL" if the last assistant message has tool calls, else END."""
    if not state.context:
        return END

    last_message = state.context[-1]

    if (
        last_message.role == "assistant"
        and hasattr(last_message, "tools_calls")
        and last_message.tools_calls
    ):
        return "TOOL"

    return END


def _make_should_use_tools(tool_node_name: str) -> Callable[[AgentState], str]:
    """Create the routing function for the configured tool node name."""

    if tool_node_name == "TOOL":
        return _should_use_tools

    def _route(state: AgentState) -> str:
        result = _should_use_tools(state)
        if result == "TOOL":
            return tool_node_name
        return END

    return _route


class ReactAgent[StateT: AgentState]:
    def __init__(  # noqa: PLR0913
        self,
        model: str,
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        publisher: BasePublisher | list[BasePublisher] | None = None,
        id_generator: BaseIDGenerator = DefaultIDGenerator(),
        container: InjectQ | None = None,
        *,
        output_type: str = "text",
        system_prompt: list[dict[str, Any]] | None = None,
        tools: Iterable[Callable] | None = None,
        client: Any = None,  # FASTMCP client instance
        pass_user_info_to_mcp: bool = False,
        extra_messages: list[Message] | None = None,
        trim_context: bool = False,
        tools_tags: set[str] | None = None,
        reasoning_config: dict[str, Any] | bool | None = True,
        skills: SkillConfig | None = None,
        memory: MemoryConfig | None = None,
        retry_config: Any = True,
        fallback_models: list[str | tuple[str, str]] | None = None,
        multimodal_config: MultimodalConfig | None = None,
        output_schema: Any | None = None,
        main_node_name: str = "MAIN",
        tool_node_name: str = "TOOL",
        **agent_kwargs: Any,
    ):
        self._state = state
        self._context_manager = context_manager
        self._publisher = publisher
        self._id_generator = id_generator
        self._container = container

        self._graph = self._create_graph()
        self._main_node_name = main_node_name
        self._tool_node_name = tool_node_name
        self._tool_node = self._build_tool_node(
            tools=list(tools or []),
            client=client,
            pass_user_info_to_mcp=pass_user_info_to_mcp,
        )
        self._agent = Agent(
            model=model,
            output_type=output_type,
            system_prompt=system_prompt,
            tool_node=self._tool_node,
            extra_messages=extra_messages,
            trim_context=trim_context,
            tools_tags=tools_tags,
            reasoning_config=reasoning_config,
            skills=skills,
            memory=memory,
            retry_config=retry_config,
            fallback_models=fallback_models,
            multimodal_config=multimodal_config,
            output_schema=output_schema,
            **agent_kwargs,
        )

    def _create_graph(self) -> StateGraph[StateT]:
        return StateGraph[StateT](
            state=self._state,
            context_manager=self._context_manager,
            publisher=self._publisher,
            id_generator=self._id_generator,
            container=self._container,
        )

    def _build_tool_node(
        self,
        *,
        tools: list[Callable],
        client: Any,
        pass_user_info_to_mcp: bool,
    ) -> ToolNode | None:
        if not tools and client is None:
            return None

        return ToolNode(
            tools,
            client=client,
            pass_user_info_to_mcp=pass_user_info_to_mcp,
        )

    def _configure_graph(self) -> None:
        self._graph = self._create_graph()
        self._graph.add_node(self._main_node_name, self._agent)

        if self._tool_node is None:
            self._graph.set_entry_point(self._main_node_name)
            self._graph.add_edge(self._main_node_name, END)
            return

        self._graph.add_node(self._tool_node_name, self._tool_node)
        self._graph.add_conditional_edges(
            self._main_node_name,
            _make_should_use_tools(self._tool_node_name),
            {self._tool_node_name: self._tool_node_name, END: END},
        )
        self._graph.add_edge(self._tool_node_name, self._main_node_name)
        self._graph.set_entry_point(self._main_node_name)

    def compile(
        self,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
        media_store: "BaseMediaStore | None" = None,
        shutdown_timeout: float = 30.0,
    ) -> CompiledGraph:
        self._configure_graph()
        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
            media_store=media_store,
            shutdown_timeout=shutdown_timeout,
        )
