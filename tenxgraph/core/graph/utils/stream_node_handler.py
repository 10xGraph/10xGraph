"""Streaming node handler for TAF graph workflows.

This module provides the StreamNodeHandler class, which manages the execution of graph nodes
that support streaming output. It handles regular function nodes, ToolNode instances,
and Agent instances,enabling incremental result processing, dependency injection,
callback management, and event publishing.

StreamNodeHandler is a key component for enabling real-time, chunked, or incremental responses
in agent workflows, supporting both synchronous and asynchronous execution patterns.
"""

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from typing import TYPE_CHECKING, Any, Union

from injectq import Inject

from tenxgraph.core.exceptions import NodeError
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.graph.utils.stream_utils import check_non_streaming
from tenxgraph.core.graph.utils.utils import process_node_result
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ErrorBlock
from tenxgraph.core.state.stream_chunks import StreamChunk, StreamEvent
from tenxgraph.core.state.stream_emitter import StreamEmitter
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.utils import (
    CallbackContext,
    CallbackManager,
    InvocationType,
    call_sync_or_async,
)
from tenxgraph.utils.command import Command
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.interrupt import activate as activate_interrupts

from .handler_mixins import BaseLoggingMixin


if TYPE_CHECKING:
    from tenxgraph.core.graph.base_agent import BaseAgent


logger = logging.getLogger("tenxgraph.graph")


class StreamNodeHandler(BaseLoggingMixin):
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

    async def _handle_single_tool(
        self,
        tool_call: dict[str, Any],
        state: AgentState,
        config: dict[str, Any],
    ) -> AsyncIterable[Message | dict[str, Any] | StreamChunk]:
        function_name = tool_call.get("function", {}).get("name", "")
        function_args: dict = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
        tool_call_id = tool_call.get("id", "")

        logger.info(
            "Node '%s' executing tool '%s' with %d arguments",
            self.name,
            function_name,
            len(function_args),
        )
        logger.debug("Tool arguments: %s", function_args)

        yield StreamChunk(
            event=StreamEvent.UPDATES,
            data={
                "status": "invoking_tool",
                "tool_name": function_name,
                "args": function_args,
                "tool_call_id": tool_call_id,
                "node": self.name,
            },
            thread_id=config.get("thread_id"),
            run_id=config.get("run_id"),
        )

        # Create a per-tool queue.  The StreamEmitter pushes chunks into this
        # queue from the tool (including from sync-tool threads); the background
        # task pushes the final tool result(s); we drain the queue and yield
        # items live so the frontend sees progress before the tool finishes.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _DONE = object()  # sentinel

        emitter = StreamEmitter(
            tool_name=function_name,
            tool_call_id=tool_call_id,
            node_name=self.name,
            thread_id=config.get("thread_id"),
            run_id=config.get("run_id"),
            queue=queue,
            loop=loop,
        )

        async def _run_tool() -> None:
            try:
                async for result in self.func.stream(  # type: ignore[union-attr]
                    function_name,
                    function_args,
                    tool_call_id=tool_call_id,
                    state=state,
                    config=config,
                    emit=emitter,
                ):
                    await queue.put(result)
            finally:
                await queue.put(_DONE)

        task = asyncio.create_task(_run_tool())

        while True:
            item = await queue.get()
            if item is _DONE:
                break

            if isinstance(item, StreamChunk):
                # Already a StreamChunk (e.g. emitted by the tool via StreamEmitter)
                yield item
            elif isinstance(item, dict):
                # Dict result from ToolResult (state update) — yield the message
                # and pass the dict through so the consumer can handle state merging
                msg = item.get("messages")
                if isinstance(msg, Message):
                    yield msg
                    yield StreamChunk(
                        event=StreamEvent.MESSAGE,
                        message=msg,
                        data={
                            "status": "tool_result",
                            "tool_name": function_name,
                            "args": function_args,
                            "tool_call_id": tool_call_id,
                            "node": self.name,
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
                # Yield the raw dict so upstream can merge state
                yield item
            elif isinstance(item, Message):
                yield item
                is_error = (
                    any(isinstance(block, ErrorBlock) for block in item.content)
                    if isinstance(item.content, list)
                    else False
                )
                if is_error:
                    yield StreamChunk(
                        event=StreamEvent.ERROR,
                        data={
                            "status": "tool_failed",
                            "tool_name": function_name,
                            "args": function_args,
                            "tool_call_id": tool_call_id,
                            "node": self.name,
                            "reason": "Tool execution resulted in error",
                            "error": next(
                                block for block in item.content if isinstance(block, ErrorBlock)
                            ).message,
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
                else:
                    yield StreamChunk(
                        event=StreamEvent.MESSAGE,
                        message=item,
                        data={
                            "status": "tool_result",
                            "tool_name": function_name,
                            "args": function_args,
                            "tool_call_id": tool_call_id,
                            "node": self.name,
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
            else:
                yield item

        # Re-raise any exception that occurred inside the background task.
        await task

    async def _call_tools(
        self,
        last_message: Message,
        state: "AgentState",
        config: dict[str, Any],
    ) -> AsyncIterable[Message | dict[str, Any] | StreamChunk | Command]:
        logger.debug("Node '%s' calling tools from message", self.name)
        if (
            hasattr(last_message, "tools_calls")
            and last_message.tools_calls
            and len(last_message.tools_calls) > 0
        ):
            # Lazy import to avoid circular dependency
            from tenxgraph.prebuilt.tools.handoff import is_handoff_tool

            # Check for handoff BEFORE executing any tools
            for tool_call in last_message.tools_calls:
                tool_name = tool_call.get("function", {}).get("name", "")
                is_handoff, target_agent = is_handoff_tool(tool_name)

                if is_handoff:
                    logger.info(
                        "Handoff detected in node '%s': tool '%s' -> agent '%s'",
                        self.name,
                        tool_name,
                        target_agent,
                    )
                    yield Command(  # type: ignore
                        update=None,
                        goto=target_agent,
                    )
                    return

            logger.info(
                "Node '%s' executing %d tool calls in parallel",
                self.name,
                len(last_message.tools_calls),
            )

            # Shared output queue — all parallel tool workers push here so items
            # are yielded as soon as they are available, not after all tools finish.
            shared_queue: asyncio.Queue = asyncio.Queue()
            _WORKER_DONE = object()  # per-worker sentinel

            generators = [
                self._handle_single_tool(tool_call, state, config)
                for tool_call in last_message.tools_calls
            ]

            async def _worker(gen: AsyncIterable) -> None:
                try:
                    async for item in gen:
                        await shared_queue.put(item)
                finally:
                    await shared_queue.put(_WORKER_DONE)

            tasks = [asyncio.create_task(_worker(gen)) for gen in generators]
            remaining = len(tasks)

            while remaining > 0:
                item = await shared_queue.get()
                if item is _WORKER_DONE:
                    remaining -= 1
                else:
                    yield item

            # Re-raise the first exception from any worker task.
            for task in tasks:
                await task

        else:
            logger.exception("Node '%s': No tool calls to execute", self.name)
            raise NodeError(
                message="No tool calls to execute",
                error_code="NODE_002",
                context={"node_name": self.name},
            )

    def _prepare_input_data(
        self,
        state: "AgentState",
        config: dict[str, Any],
    ) -> dict:
        sig = inspect.signature(self.func)  # type: ignore Tool node won't come here
        input_data = {}
        default_data = {
            "state": state,
            "config": config,
        }

        # Collect all required parameters (excluding *args/**kwargs)
        required_params = []
        for param_name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if param.default is inspect.Parameter.empty:
                required_params.append((param_name, param))

        # If function has exactly one required parameter and it's not 'state' or 'config',
        # assume it should receive the state (common pattern for simple lambdas like lambda s: s)
        if len(required_params) == 1:
            param_name, param = required_params[0]
            if param_name not in ["state", "config"]:
                input_data[param_name] = state
                return input_data

        # Otherwise, handle standard named parameters
        for param_name, param in sig.parameters.items():
            # Skip *args/**kwargs
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            # check its state, config
            if param_name in ["state", "config"]:
                input_data[param_name] = default_data[param_name]
            # Include regular function arguments
            elif param.default is inspect.Parameter.empty:
                raise TypeError(
                    f"Missing required parameter '{param_name}' for function '{self.func}'"
                )

        return input_data

    def _get_last_tool_message(self, state: AgentState) -> Message:
        """Return the latest context message required for tool execution."""
        if state.context and len(state.context) > 0:
            return state.context[-1]

        error_msg = "No context available for tool execution"
        logger.error("Node '%s': %s", self.name, error_msg)
        raise NodeError(
            message=error_msg,
            error_code="NODE_003",
            context={"node_name": self.name},
        )

    def _merge_tool_state(self, target_state: AgentState, tool_state: AgentState) -> None:
        """Merge tool-produced state fields into the target state."""
        for field_name in tool_state.model_fields:
            if field_name in ("context", "context_summary", "execution_meta"):
                continue
            setattr(target_state, field_name, getattr(tool_state, field_name))

    def _extract_tool_messages(self, payload: dict[str, Any]) -> list[Message]:
        """Extract tool messages from either legacy or normalized payload keys."""
        message_payload = payload.get("messages", payload.get("message"))

        if isinstance(message_payload, Message):
            return [message_payload]
        if isinstance(message_payload, list):
            return [item for item in message_payload if isinstance(item, Message)]
        return []

    def _collect_tool_stream_dict(
        self,
        item: dict[str, Any],
        merged_state: AgentState,
        collected_messages: list[Message],
    ) -> tuple[AgentState, bool]:
        """Merge streamed tool state updates and accumulate associated messages."""
        has_state_update = False
        tool_state = item.get("state")

        if isinstance(tool_state, AgentState):
            self._merge_tool_state(merged_state, tool_state)
            has_state_update = True

        for message in self._extract_tool_messages(item):
            if message not in collected_messages:
                collected_messages.append(message)

        return merged_state, has_state_update

    async def _stream_tool_node(
        self,
        state: AgentState,
        config: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any] | Message | StreamChunk | Command]:
        """Execute a ToolNode and emit normalized streaming payloads."""
        logger.debug("Node '%s' is a ToolNode, executing tool calls", self.name)
        last_message = self._get_last_tool_message(state)
        logger.debug("Node '%s' processing tool calls from last message", self.name)

        result = self._call_tools(
            last_message,
            state,
            config,
        )

        collected_messages: list[Message] = []
        merged_state = state
        has_state_update = False
        command_goto: str | None = None

        async for item in result:
            if isinstance(item, Command):
                command_goto = item.goto
                yield item
                continue

            if isinstance(item, dict):
                merged_state, did_update = self._collect_tool_stream_dict(
                    item,
                    merged_state,
                    collected_messages,
                )
                has_state_update = has_state_update or did_update
                continue

            if isinstance(item, StreamChunk):
                yield item
                continue

            if isinstance(item, Message):
                if item not in collected_messages:
                    collected_messages.append(item)
                yield item

        if command_goto is not None or has_state_update:
            yield {
                "is_non_streaming": True,
                "state": merged_state,
                "messages": collected_messages,
                "next_node": command_goto,
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
            async for item in self._call_agent_node(state, config):
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

        context = CallbackContext(
            invocation_type=InvocationType.AI,
            node_name=self.name,
            function_name=function_name,
            metadata={"config": config},
        )

        # Execute before_invoke callbacks
        input_data = self._prepare_input_data(
            state,
            config,
        )

        last_message = state.context[-1] if state.context and len(state.context) > 0 else None

        event = EventModel.default(
            config,
            data={"state": state.model_dump()},
            event=Event.NODE_EXECUTION,
            content_type=[ContentType.STATE],
            node_name=self.name,
            extra={
                "node": self.name,
                "function_name": function_name,
                "last_message": last_message.model_dump() if last_message else None,
            },
        )
        publish_event(event)

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
            stream_event = StreamChunk(
                event=StreamEvent.MESSAGE,
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
                        stream_event.message = msg
                        yield stream_event

                next_node = result.goto

            messages = []
            if check_non_streaming(final_result):
                new_state, messages, next_node = await process_node_result(
                    final_result,
                    state,
                    messages,
                    seen_message_ids,
                )
                event.data["state"] = new_state.model_dump()
                event.event_type = EventType.END
                event.data["messages"] = (
                    [m.model_dump() for m in messages if not m.delta] if messages else []
                )
                event.data["next_node"] = next_node
                publish_event(event)
                for m in messages:
                    yield m
                    stream_event.message = m
                    yield stream_event

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
                    stream_event.message = item
                    yield stream_event
            # Things are done, so publish event and yield final response
            event.event_type = EventType.END
            if messages:
                final_msg = messages[-1]
                event.data["message"] = final_msg.model_dump()
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
                logger.info(
                    "Node '%s' recovered from error using callback result",
                    self.name,
                )
                # Use recovery result instead of raising the error
                event.event_type = EventType.END
                event.content = "Function execution recovered from error"
                event.data["message"] = recovery_result.model_dump()
                event.content_type = [ContentType.MESSAGE, ContentType.STATE]
                publish_event(event)

                yield recovery_result
            else:
                # Re-raise the original error
                logger.error("Node '%s' could not recover from error", self.name)
                event.event_type = EventType.ERROR
                event.content = f"Function execution failed: {e}"
                event.data["error"] = str(e)
                event.content_type = [ContentType.ERROR, ContentType.STATE]
                publish_event(event)
                raise

    async def _call_agent_node(
        self,
        state: "AgentState",
        config: dict[str, Any],
    ) -> AsyncGenerator[StreamChunk | Message | dict[str, Any]]:
        """Execute an Agent instance node with streaming support.

        Args:
            state (AgentState): Current agent state.
            config (dict): Node configuration.

        Yields:
            StreamChunk | Message | dict: Streaming output from the agent.
        """
        logger.debug(
            "Node '%s' is an Agent instance, executing agent logic with streaming", self.name
        )

        agent = self.func  # type: ignore - func is Agent instance here

        converter = await agent.execute(state, config)  # type: ignore

        # Collect all messages during streaming
        messages = []

        stream_event = StreamChunk(
            event=StreamEvent.MESSAGE,
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
            stream_event.message = item
            yield stream_event

        yield {
            "is_non_streaming": True,
            "state": state,
            "messages": messages,
            "next_node": None,
        }

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
