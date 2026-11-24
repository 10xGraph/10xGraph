"""Streaming node handler for TAF graph workflows.

This module provides the StreamNodeHandler class, which manages the execution of graph nodes
that support streaming output. It handles regular function nodes, ToolNode instances,
and Agent instances,enabling incremental result processing, dependency injection,
callback management, and event publishing.

StreamNodeHandler is a key component for enabling real-time, chunked, or incremental responses
in agent workflows, supporting both synchronous and asynchronous execution patterns.
"""

import logging
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from typing import TYPE_CHECKING, Any, Union

from injectq import Inject

from tenxgraph.core.exceptions import NodeError
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.graph.utils.stream_utils import check_non_streaming
from tenxgraph.core.graph.utils.utils import process_node_result
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.stream_chunks import StreamChunk, StreamEvent
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.runtime.publisher.events import ContentType, EventType
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.utils import (
    CallbackManager,
    call_sync_or_async,
)
from tenxgraph.utils.command import Command
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.interrupt import activate as activate_interrupts

from .handler_mixins import BaseLoggingMixin, mirror_last_message
from .tool_runner import ToolRunner
from .tool_runtime import ToolRuntimeMixin


if TYPE_CHECKING:
    from tenxgraph.core.graph.base_agent import BaseAgent


logger = logging.getLogger("tenxgraph.graph")


class StreamNodeHandler(BaseLoggingMixin, ToolRuntimeMixin):
    """Handles streaming execution for graph nodes in TAF workflows.

    StreamNodeHandler manages the execution of nodes that can produce streaming output,
    including regular function nodes, ToolNode instances, and Agent instances. It supports
    dependency injection, callback management, event publishing, and incremental result processing.

    Attributes:
        name: Unique identifier for the node within the graph.
        func: The function, ToolNode, or Agent to execute. Determines streaming behavior.

    Example:
        ```python
        handler = StreamNodeHandler("process", process_function)
        async for chunk in handler.stream(config, state):
            print(chunk)
        ```
    """

    def __init__(
        self,
        name: str,
        func: Union[Callable, "ToolNode", "BaseAgent"],
    ):
        """Initialize a new StreamNodeHandler instance.

        Args:
            name: Unique identifier for the node within the graph.
            func: The function, ToolNode, or Agent to execute. Determines streaming behavior.
        """
        self.name = name
        self.func = func

    async def _stream_tool_node(
        self,
        state: AgentState,
        config: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any] | Message | StreamChunk | Command]:
        """Run a ToolNode step through the tool runner, in the stream item shapes."""
        async for item in ToolRunner(self.name, self.func).run(state, config, streaming=True):
            if isinstance(item, StreamChunk):
                yield item
                continue
            for message in item.messages:
                yield message
            if item.next_node is not None:
                yield Command(update=None, goto=item.next_node)  # type: ignore[arg-type]
            if item.next_node is not None or item.state_updated:
                yield {
                    "is_non_streaming": True,
                    "state": state,
                    "messages": item.messages,
                    "next_node": item.next_node,
                }

    async def _stream_by_node_type(
        self,
        state: AgentState,
        config: dict[str, Any],
        callback_mgr: CallbackManager,
    ) -> AsyncGenerator[dict[str, Any] | Message | StreamChunk | Command]:
        """Dispatch streaming execution to the appropriate node implementation."""
        from tenxgraph.core.graph.agent import Agent
        from tenxgraph.core.graph.base_agent import BaseAgent

        if isinstance(self.func, Agent | BaseAgent):
            logger.debug("Node '%s' is an Agent instance, executing agent streaming", self.name)
            async for item in self._call_agent_node(state, config, callback_mgr):
                yield item
            return

        if isinstance(self.func, ToolNode):
            async for item in self._stream_tool_node(state, config):
                yield item
            return

        async for item in self._call_normal_node(
            state,
            config,
            callback_mgr,
        ):
            yield item

    async def _call_normal_node(  # noqa: PLR0912, PLR0915
        self,
        state: "AgentState",
        config: dict[str, Any],
        callback_mgr: CallbackManager,
    ) -> AsyncIterable[dict[str, Any] | Message | StreamChunk]:
        logger.debug("Node '%s' calling normal function", self.name)
        result: dict[str, Any] | Message = {}

        logger.debug("Node '%s' is a regular function, executing with callbacks", self.name)
        # This is a regular function - likely AI function
        # Create callback context for AI invocation
        function_name = getattr(self.func, "__name__", str(self.func))
        context = self._node_callback_context(config, function_name)
        input_data = self._prepare_input_data(
            state,
            config,
        )
        event = self._publish_node_start(config, state, function_name)

        try:
            logger.debug("Node '%s' executing before_invoke callbacks", self.name)
            # Execute before_invoke callbacks
            input_data = await callback_mgr.execute_before_invoke(context, input_data)
            logger.debug("Node '%s' executing function", self.name)
            event.event_type = EventType.PROGRESS
            event.content = "Function execution started"
            publish_event(event)

            yield StreamChunk(
                event=StreamEvent.UPDATES,
                data={
                    "status": "invoking_node",
                    "tool_name": function_name,
                    "node": self.name,
                },
                thread_id=config.get("thread_id"),
                run_id=config.get("run_id"),
            )

            # Snapshot the context: a node may append to it and return the same state object.
            seen_message_ids = {msg.message_id for msg in state.context or []}
            # Execute the actual function
            # interrupt() in the node reads the scope the graph loop opened for this node.
            with activate_interrupts(config):
                result = await call_sync_or_async(
                    self.func,  # type: ignore
                    **input_data,
                )
            logger.debug("Node '%s' function execution completed", self.name)

            logger.debug("Node '%s' executing after_invoke callbacks", self.name)
            # Execute after_invoke callbacks
            result = await callback_mgr.execute_after_invoke(context, input_data, result)

            yield StreamChunk(
                event=StreamEvent.UPDATES,
                data={
                    "status": "node_invoked",
                    "tool_name": function_name,
                    "node": self.name,
                },
                thread_id=config.get("thread_id"),
                run_id=config.get("run_id"),
            )

            # Now lets convert the response here only, upstream will be easy to handle
            ##############################################################################
            ################### Logics for streaming ##########################
            ##############################################################################
            """
            Check user sending command or not
            if command then we will check its streaming or not
            if streaming then we will yield from converter stream
            if not streaming then we will convert it and yield end event
            if its not command then we will check its streaming or not
            if streaming then we will yield from converter stream
            if not streaming then we will convert it and yield end event
            """
            # first check its sync and not streaming
            next_node = None
            final_result = result

            def message_chunk(message: Any) -> StreamChunk:
                # One chunk per item: the stream is queued before it is read, so a shared
                # chunk would show every reader the last item.
                return StreamChunk(
                    event=StreamEvent.MESSAGE,
                    message=message,
                    thread_id=config.get("thread_id"),
                    run_id=config.get("run_id"),
                    metadata={
                        "node": self.name,
                        "function_name": function_name,
                    },
                )

            # if type of command then we will update it
            if isinstance(result, Command):
                # now check the updated
                if result.update:
                    final_result = result.update

                if result.state:
                    state = result.state
                    for msg in state.context:
                        if msg.message_id in seen_message_ids:
                            continue  # already streamed in an earlier step
                        yield msg
                        yield message_chunk(msg)

                next_node = result.goto

            messages = []
            if check_non_streaming(final_result):
                new_state, messages, next_node = await process_node_result(
                    final_result,
                    state,
                    messages,
                    seen_message_ids,
                )
                event.data["state"] = new_state
                event.event_type = EventType.END
                event.data["messages"] = [m for m in messages if not m.delta]
                event.data["next_node"] = next_node
                publish_event(event)
                for m in messages:
                    yield m
                    yield message_chunk(m)

                yield {
                    "is_non_streaming": True,
                    "state": new_state,
                    "messages": messages,
                    "next_node": next_node,
                }
                return  # done

            # If the result is a ConverterCall with stream=True, use the converter
            if isinstance(result, ModelResponseConverter) and result.response:
                stream_gen = result.stream(
                    config,
                    node_name=self.name,
                    meta={
                        "function_name": getattr(self.func, "__name__", str(self.func)),
                    },
                )
                # this will return event_model or message
                async for item in stream_gen:
                    if isinstance(item, Message) and not item.delta:
                        messages.append(item)
                    yield item
                    yield message_chunk(item)
            # Things are done, so publish event and yield final response
            event.event_type = EventType.END
            if messages:
                final_msg = messages[-1]
                event.data["message"] = final_msg
                # Populate simple content and structured blocks when available
                event.content = (
                    final_msg.text() if isinstance(final_msg.content, list) else final_msg.content
                )
                if isinstance(final_msg.content, list):
                    event.content_blocks = final_msg.content
            else:
                event.data["message"] = None
                event.content = ""
                event.content_blocks = None
            event.content_type = [ContentType.MESSAGE, ContentType.STATE]
            publish_event(event)
            # if user use command and its streaming in that case we need to handle next node also
            yield {
                "is_non_streaming": False,
                "messages": messages,
                "next_node": next_node,
            }

        except Exception as e:
            logger.warning(
                "Node '%s' execution failed, executing error callbacks: %s", self.name, e
            )
            # Execute error callbacks
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if isinstance(recovery_result, Message):
                # Use recovery result instead of raising the error
                event.content = "Function execution recovered from error"
                self._publish_node_recovered(event, recovery_result)
                yield recovery_result
            else:
                # Re-raise the original error
                event.content = f"Function execution failed: {e}"
                self._publish_node_failed(event, e, config)
                raise

    async def _call_agent_node(
        self,
        state: "AgentState",
        config: dict[str, Any],
        callback_mgr: CallbackManager,
    ) -> AsyncGenerator[StreamChunk | Message | dict[str, Any]]:
        """Execute an Agent instance node with streaming support.

        Runs the same AI callbacks (before_invoke, after_invoke, on_error) and publishes the
        same NODE_EXECUTION events as the invoke path. Chunks are already with the client
        when ``after_invoke`` runs, so what it returns is used for the saved result only.

        Args:
            state (AgentState): Current agent state.
            config (dict): Node configuration.
            callback_mgr (CallbackManager): Callback manager for hooks.

        Yields:
            StreamChunk | Message | dict: Streaming output from the agent.
        """
        logger.debug(
            "Node '%s' is an Agent instance, executing agent logic with streaming", self.name
        )

        agent = self.func  # type: ignore - func is Agent instance here

        context = self._node_callback_context(
            config, getattr(agent, "__class__", type(agent)).__name__
        )
        input_data = {"state": state, "config": config}
        event = self._publish_node_start(config, state, type(agent).__name__)

        try:
            input_data = await callback_mgr.execute_before_invoke(context, input_data)

            event.event_type = EventType.PROGRESS
            event.metadata["status"] = "Agent execution started"
            publish_event(event)

            converter = await agent.execute(state, config)  # type: ignore

            # Collect all messages during streaming
            messages = []

            def message_chunk(message: Any) -> StreamChunk:
                # One chunk per item (see the function-node path).
                return StreamChunk(
                    event=StreamEvent.MESSAGE,
                    message=message,
                    thread_id=config.get("thread_id"),
                    run_id=config.get("run_id"),
                    metadata={
                        "node": self.name,
                        "function_name": self.name,
                    },
                )

            stream_gen = converter.stream(
                config,
                node_name=self.name,
                meta={
                    "function_name": self.name,
                },
            )
            # this will return event_model or message

            async for item in stream_gen:
                if isinstance(item, Message) and not item.delta:
                    messages.append(item)
                yield item
                yield message_chunk(item)

            result = {"state": state, "messages": messages, "next_node": None}
            result = await callback_mgr.execute_after_invoke(context, input_data, result)

            is_dict = isinstance(result, dict)
            final_messages = result.get("messages", messages) if is_dict else messages
            event.data["state"] = state
            event.event_type = EventType.END
            event.metadata["status"] = "Agent execution completed"
            event.data["messages"] = final_messages
            event.data["next_node"] = None
            mirror_last_message(event, final_messages, stringify=False)
            publish_event(event)

            yield {
                "is_non_streaming": True,
                "state": result.get("state", state) if is_dict else state,
                "messages": final_messages,
                "next_node": result.get("next_node") if is_dict else None,
            }

        except Exception as e:
            logger.warning(
                "Node '%s' agent execution failed, executing error callbacks: %s",
                self.name,
                e,
            )
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if isinstance(recovery_result, Message):
                event.metadata["status"] = "Agent execution recovered from error"
                self._publish_node_recovered(event, recovery_result)
                yield recovery_result
                yield StreamChunk(
                    event=StreamEvent.MESSAGE,
                    message=recovery_result,
                    thread_id=config.get("thread_id"),
                    run_id=config.get("run_id"),
                    metadata={"node": self.name, "function_name": self.name},
                )
                yield {
                    "is_non_streaming": False,
                    "messages": [recovery_result],
                    "next_node": None,
                }
                return
            event.metadata["status"] = f"Agent execution failed: {e}"
            self._publish_node_failed(event, e, config)
            raise

    async def stream(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ) -> AsyncGenerator[dict[str, Any] | Message | StreamChunk | Command]:
        """Execute the node function with streaming output and callback support.

        Handles ToolNode, Agent, and regular function nodes, yielding incremental results
        as they become available. Supports dependency injection, callback management,
        and event publishing for monitoring and debugging.

        Args:
            config: Configuration dictionary containing execution context and settings.
            state: Current AgentState providing workflow context and shared state.
            callback_mgr: Callback manager for pre/post execution hook handling.

        Yields:
            Dictionary objects or Message instances representing incremental outputs
            from the node function. The exact type and frequency of yields depends on
            the node function's streaming implementation.

        Raises:
            NodeError: If node execution fails or encounters an error.

        Example:
            ```python
            async for chunk in handler.stream(config, state):
                print(chunk)
            ```
        """
        callback_mgr = fresh(callback_mgr)
        logger.info("Executing node '%s'", self.name)
        logger.debug(
            "Node '%s' execution with state context size=%d, config keys=%s",
            self.name,
            len(state.context) if state.context else 0,
            list(config.keys()) if config else [],
        )

        # In this function publishing events not required
        # If its tool node, its already handled there, from start to end
        # In this class we need to handle normal function calls only
        # We will yield events from here only for normal function calls
        # ToolNode will yield events from its own stream method

        try:
            async for item in self._stream_by_node_type(
                state,
                config,
                callback_mgr,
            ):
                yield item

            logger.info("Node '%s' execution completed successfully", self.name)
        except Exception as e:
            # This is the final catch-all for node execution errors
            logger.exception("Node '%s' execution failed: %s", self.name, e)
            raise NodeError(
                message=f"Error in node '{self.name}': {e!s}",
                error_code="NODE_001",
                context={
                    "node_name": self.name,
                    "error_type": type(e).__name__,
                },
            ) from e
