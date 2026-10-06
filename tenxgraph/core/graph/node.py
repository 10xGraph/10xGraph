"""Node execution and management for TAF graph workflows.

This module defines the Node class, which represents executable units within
a TAF graph workflow. Nodes encapsulate functions, ToolNode instances, or Agent instances
that perform specific tasks, handle dependency injection, manage execution context,
and support both synchronous and streaming execution modes.

Nodes are the fundamental building blocks of graph workflows, responsible for
processing state, executing business logic, and producing outputs that drive
the workflow forward. They integrate seamlessly with TAF's dependency
injection system and callback management framework.
"""

import logging
from collections.abc import AsyncIterable, Callable
from typing import TYPE_CHECKING, Any, Union

from injectq import Inject

from agentflow.core.graph.utils.invoke_node_handler import InvokeNodeHandler
from agentflow.core.graph.utils.stream_node_handler import StreamNodeHandler
from agentflow.core.state import AgentState, Message
from agentflow.core.state.stream_chunks import StreamChunk
from agentflow.utils import (
    CallbackManager,
)
from agentflow.utils.command import Command
from agentflow.utils.injection import fresh

from .tool_node import ToolNode


if TYPE_CHECKING:
    from .base_agent import BaseAgent


logger = logging.getLogger("agentflow.graph")


class Node:
    """Represents a node in the graph workflow.

    A Node encapsulates a function or ToolNode that can be executed as part of
    a graph workflow. It handles dependency injection, parameter mapping, and
    execution context management.

    The Node class supports both regular callable functions and ToolNode instances
    for handling tool-based operations. It automatically injects dependencies
    based on function signatures and provides legacy parameter support.

    Attributes:
        name (str): Unique identifier for the node within the graph.
        func (Union[Callable, ToolNode]): The function or ToolNode to execute.

    Example:
        >>> def my_function(state, config):
        ...     return {"result": "processed"}
        >>> node = Node("processor", my_function)
        >>> result = await node.execute(state, config)
    """

    def __init__(
        self,
        name: str,
        func: Union[Callable, "ToolNode", "BaseAgent"],
    ):
        """Initialize a new Node instance with function and dependencies.

        Args:
            name: Unique identifier for the node within the graph. This name
                is used for routing, logging, and referencing the node in
                graph configuration.
            func: The function, ToolNode, or Agent to execute when this node is called.
                Functions should accept at least 'state' and 'config' parameters.
                ToolNode instances handle tool-based operations and provide
                their own execution logic. Agent instances provide high-level
                LLM interaction capabilities.
            publisher: Optional event publisher for execution monitoring.
                Injected via dependency injection if not explicitly provided.
                Used for publishing node execution events and status updates.

        Note:
            The function signature is automatically analyzed to determine
            required parameters and dependency injection points. Parameters
            matching injectable service names will be automatically provided
            by the framework during execution.
        """
        logger.debug(
            "Initializing node '%s' with func=%s",
            name,
            getattr(func, "__name__", type(func).__name__),
        )
        self.name = name
        self.func = func
        self.invoke_handler = InvokeNodeHandler(
            name,
            func,
        )

        self.stream_handler = StreamNodeHandler(
            name,
            func,
        )

    async def execute(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ) -> dict[str, Any] | list[Message] | Command:
        """Execute the node function with comprehensive context and callback support.

        Executes the node's function or ToolNode with full dependency injection,
        callback hook execution, and error handling. This method provides the
        complete execution environment including state access, configuration,
        and injected services.

        Args:
            config: Configuration dictionary containing execution context,
                user settings, thread identification, and runtime parameters.
            state: Current AgentState providing workflow context, message history,
                and shared state information accessible to the node function.
            callback_mgr: Callback manager for executing pre/post execution hooks.
                Injected via dependency injection if not explicitly provided.

        Returns:
            Either a dictionary containing updated state and execution results,
            or a list of Message objects representing the node's output.
            The return type depends on the node function's implementation.

        Raises:
            Various exceptions depending on node function behavior. All exceptions
            are handled by the callback manager's error handling hooks before
            being propagated.

        Example:
            ```python
            # Node function that returns messages
            def process_data(state, config):
                result = process(state.data)
                return [Message.text_message(f"Processed: {result}")]


            node = Node("processor", process_data)
            messages = await node.execute(config, state)
            ```

        Note:
            The node function receives dependency-injected parameters based on
            its signature. Common injectable parameters include 'state', 'config',
            'context_manager', 'publisher', and other framework services.
        """
        callback_mgr = fresh(callback_mgr)
        return await self.invoke_handler.invoke(
            config,
            state,
            callback_mgr,
        )

    async def stream(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ) -> AsyncIterable[dict[str, Any] | Message | StreamChunk]:
        """Execute the node function with streaming output support.

        Similar to execute() but designed for streaming scenarios where the node
        function can produce incremental results. This method provides an async
        iterator interface over the node's outputs, allowing for real-time
        processing and response streaming.

        Args:
            config: Configuration dictionary with execution context and settings.
            state: Current AgentState providing workflow context and shared state.
            callback_mgr: Callback manager for pre/post execution hook handling.

        Yields:
            Dictionary objects or Message instances representing incremental
            outputs from the node function. The exact type and frequency of
            yields depends on the node function's streaming implementation.

        Example:
            ```python
            async def streaming_processor(state, config):
                for item in large_dataset:
                    result = process_item(item)
                    yield Message.text_message(f"Processed item: {result}")


            node = Node("stream_processor", streaming_processor)
            async for output in node.stream(config, state):
                print(f"Streamed: {output.content}")
            ```

        Note:
            Not all node functions support streaming. For non-streaming functions,
            this method will yield a single result equivalent to calling execute().
            The streaming capability is determined by the node function's implementation.
        """
        callback_mgr = fresh(callback_mgr)
        result = self.stream_handler.stream(
            config,
            state,
            callback_mgr,
        )

        async for item in result:
            yield item  # type: ignore
