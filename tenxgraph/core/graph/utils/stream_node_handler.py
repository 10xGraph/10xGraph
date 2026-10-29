"""Streaming node handler for TAF graph workflows.

This module provides the StreamNodeHandler class, which manages the execution of graph nodes
that support streaming output. It handles regular function nodes, ToolNode instances,
and Agent instances,enabling incremental result processing, dependency injection,
callback management, and event publishing.

StreamNodeHandler is a key component for enabling real-time, chunked, or incremental responses
in agent workflows, supporting both synchronous and asynchronous execution patterns.
"""

import asyncio
import copy
import inspect
import logging
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

from injectq import Inject

from tenxgraph.core.exceptions import NodeError, NodeTimeoutError
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.graph.utils.guards import execute_with_guards, resolve_timeout
from tenxgraph.core.graph.utils.stream_utils import check_non_streaming
from tenxgraph.core.graph.utils.tool_dispatch import (
    allowed_tool_names,
    hidden_tool_message,
    invalid_arguments_message,
    parse_call_arguments,
    resolve_max_parallel_tool_calls,
)
from tenxgraph.core.graph.utils.utils import process_node_result
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ErrorBlock
from tenxgraph.core.state.stream_chunks import StreamChunk, StreamEvent
from tenxgraph.core.state.stream_emitter import StreamEmitter
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.storage.checkpointer import BaseCheckpointer
from tenxgraph.utils import (
    CallbackContext,
    CallbackManager,
    InvocationType,
    call_sync_or_async,
    metrics,
)
from tenxgraph.utils.command import Command
from tenxgraph.utils.constants import DEFAULT_TOOL_TIMEOUT_SECONDS
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.interrupt import GraphInterrupt
from tenxgraph.utils.interrupt import activate as activate_interrupts

from .handler_mixins import BaseLoggingMixin
from .tool_runtime import ToolRuntimeMixin


if TYPE_CHECKING:
    from tenxgraph.core.graph.base_agent import BaseAgent


logger = logging.getLogger("tenxgraph.graph")


@dataclass
class _ToolDone:
    """The last item a tool's stream produces: its final result (None if it gave none)."""

    result: Any


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

    async def _handle_single_tool(  # noqa: PLR0912, PLR0915
        self,
        tool_call: dict[str, Any],
        state: AgentState,
        config: dict[str, Any],
        origin_message_id: str | None = None,
        checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],
    ) -> AsyncIterable[StreamChunk | _ToolDone]:
        """Run one tool call, yielding live progress chunks and, last, a ``_ToolDone``.

        Same rules as ``InvokeNodeHandler._handle_single_tool``: a call already in the
        checkpointer's tool ledger is replayed instead of run, a finished call is recorded
        at once, and a ``tool_timeout`` the developer set cuts the call off.
        """
        checkpointer = fresh(checkpointer)
        function_name = tool_call.get("function", {}).get("name", "")
        tool_call_id = tool_call.get("id", "")

        ledger_key = self._ledger_key(origin_message_id, tool_call_id)
        if checkpointer and ledger_key:
            recorded = await self._get_recorded_tool_result(checkpointer, config, ledger_key)
            if recorded is not None:
                logger.info(
                    "Node '%s': tool '%s' (call %s) already executed; "
                    "replaying recorded result instead of re-running it",
                    self.name,
                    function_name,
                    tool_call_id,
                )
                for chunk in self._result_chunks(recorded, function_name, {}, tool_call_id, config):
                    yield chunk
                yield _ToolDone(recorded)
                return

        function_args, args_error = parse_call_arguments(tool_call)
        if args_error:
            # The model's arguments were unusable: answer with a tool error it can
            # retry from, and never run the tool with guessed arguments.
            logger.warning(
                "Node '%s': tool '%s' called with invalid arguments: %s",
                self.name,
                function_name,
                args_error,
            )
            yield StreamChunk(
                event=StreamEvent.ERROR,
                data={
                    "status": "tool_failed",
                    "tool_name": function_name,
                    "tool_call_id": tool_call_id,
                    "node": self.name,
                    "reason": "Tool call arguments were not a valid JSON object",
                    "error": args_error,
                },
                thread_id=config.get("thread_id"),
                run_id=config.get("run_id"),
            )
            yield _ToolDone(invalid_arguments_message(tool_call))
            return

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

        timeout = resolve_timeout(config, "tool_timeout", DEFAULT_TOOL_TIMEOUT_SECONDS)
        # The name comes from the model. Only a tool this node can run is a safe metric
        # label; anything else would create a new series per made-up name.
        known = self.func.knows_tool(function_name, config)  # type: ignore
        tool_attrs = {"node": self.name, "tool": function_name if known else "unknown"}
        metrics.counter("tenxgraph.tool.calls").inc(attributes=tool_attrs)

        # Create a per-tool queue.  The StreamEmitter pushes chunks into this
        # queue from the tool (including from sync-tool threads); the background
        # task pushes the final tool result; we drain the queue and yield
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

        async def _drive() -> None:
            async for result in self.func.stream(  # type: ignore[union-attr]
                function_name,
                function_args,
                tool_call_id=tool_call_id,
                state=state,
                config=config,
                emit=emitter,
            ):
                await queue.put(result)

        async def _run_tool() -> None:
            try:
                await execute_with_guards(
                    _drive(),
                    timeout=timeout,
                    on_timeout=lambda: NodeTimeoutError(
                        message=(
                            f"Tool '{function_name}' exceeded its timeout of {timeout}s "
                            f"and was cancelled"
                        ),
                        error_code="NODE_TIMEOUT_002",
                        context={
                            "node_name": self.name,
                            "tool_name": function_name,
                            "timeout": timeout,
                        },
                    ),
                )
            finally:
                await queue.put(_DONE)

        task = asyncio.create_task(_run_tool())
        final: Any = None
        try:
            with metrics.timer("tenxgraph.tool.duration", attributes=tool_attrs):
                while True:
                    item = await queue.get()
                    if item is _DONE:
                        break
                    if isinstance(item, StreamChunk):
                        # Emitted by the tool through its StreamEmitter.
                        yield item
                        continue
                    final = item
                    for chunk in self._result_chunks(
                        item, function_name, function_args, tool_call_id, config
                    ):
                        yield chunk
                # Re-raise any exception that occurred inside the background task.
                await task
        except NodeTimeoutError:
            metrics.counter("tenxgraph.tool.timeouts").inc(attributes=tool_attrs)
            raise
        except Exception:
            metrics.counter("tenxgraph.tool.errors").inc(attributes=tool_attrs)
            raise
        finally:
            # Also runs when the run is cancelled or the consumer stops reading: the tool
            # must not keep running on its own.
            if not task.done():
                task.cancel()
                await asyncio.wait({task})

        if checkpointer and ledger_key and final is not None:
            await self._record_tool_result(checkpointer, config, ledger_key, final)
        yield _ToolDone(final)

    def _result_chunks(
        self,
        item: Any,
        function_name: str,
        function_args: dict[str, Any],
        tool_call_id: str,
        config: dict[str, Any],
    ) -> list[StreamChunk]:
        """The chunks a client sees for one tool result (a Message or a state-update dict)."""
        meta = {
            "tool_name": function_name,
            "args": function_args,
            "tool_call_id": tool_call_id,
            "node": self.name,
        }
        ids = {"thread_id": config.get("thread_id"), "run_id": config.get("run_id")}
        if isinstance(item, dict):
            msg = item.get("messages")
            if isinstance(msg, Message):
                return [
                    StreamChunk(
                        event=StreamEvent.MESSAGE,
                        message=msg,
                        data={"status": "tool_result", **meta},
                        **ids,
                    )
                ]
            return []
        if isinstance(item, Message):
            error = next(
                (b for b in item.content or [] if isinstance(b, ErrorBlock)),
                None,
            )
            if error is not None:
                return [
                    StreamChunk(
                        event=StreamEvent.ERROR,
                        data={
                            "status": "tool_failed",
                            **meta,
                            "reason": "Tool execution resulted in error",
                            "error": error.message,
                        },
                        **ids,
                    )
                ]
            return [
                StreamChunk(
                    event=StreamEvent.MESSAGE,
                    message=item,
                    data={"status": "tool_result", **meta},
                    **ids,
                )
            ]
        return []

    def _hidden_tool_chunks(
        self,
        tool_call: dict[str, Any],
        config: dict[str, Any],
    ) -> list[StreamChunk]:
        """The error chunk for a call to a tool the model was not offered."""
        return [
            StreamChunk(
                event=StreamEvent.ERROR,
                data={
                    "status": "tool_failed",
                    "tool_name": tool_call.get("function", {}).get("name", ""),
                    "tool_call_id": tool_call.get("id", ""),
                    "node": self.name,
                    "reason": "Tool was not offered to the model",
                    "error": f"Tool '{tool_call.get('function', {}).get('name', '')}' not found.",
                },
                thread_id=config.get("thread_id"),
                run_id=config.get("run_id"),
            )
        ]

    async def _call_tools(  # noqa: PLR0912, PLR0915
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
            # Only tools the model was actually offered may run (Agent.tools_tags).
            allowed = allowed_tool_names(last_message)

            # A handoff is honoured only for a registered handoff tool. A name that
            # merely starts with "transfer_to_" is an ordinary (unknown) tool call.
            # Every tool call of the message is answered before control moves on.
            handoff = self._find_handoff(last_message)
            if handoff:
                target_agent, answers = handoff
                for answer in answers:
                    yield answer
                yield Command(  # type: ignore
                    update=None,
                    goto=target_agent,
                )
                return

            calls = last_message.tools_calls
            logger.info("Node '%s' executing %d tool calls in parallel", self.name, len(calls))

            # The assistant message that issued these calls makes each invocation uniquely
            # identifiable for the idempotency ledger.
            origin_message_id = getattr(last_message, "message_id", None)

            # Branch isolation, as on the invoke path: parallel tools each get a copy of the
            # state and are merged back afterwards against a baseline. A single call keeps
            # operating on the shared state.
            is_parallel = len(calls) > 1
            baseline = copy.deepcopy(state) if is_parallel else None
            branch_states: list[AgentState | None] = [None] * len(calls)
            results: list[Any] = [None] * len(calls)
            interrupts: list[GraphInterrupt] = []

            # Shared output queue: progress chunks are yielded as soon as they are available.
            # Results are NOT yielded live: they are collected and handed on in call order.
            shared_queue: asyncio.Queue = asyncio.Queue()
            _WORKER_DONE = object()  # per-worker sentinel

            # At most this many tools run at once; the rest wait for a slot.
            semaphore = asyncio.Semaphore(resolve_max_parallel_tool_calls(config))

            async def _worker(index: int, tool_call: dict[str, Any]) -> None:
                name = tool_call.get("function", {}).get("name", "")
                try:
                    if allowed is not None and name not in allowed:
                        logger.warning(
                            "Node '%s': tool '%s' was not offered to the model; not executing",
                            self.name,
                            name,
                        )
                        for chunk in self._hidden_tool_chunks(tool_call, config):
                            await shared_queue.put(chunk)
                        results[index] = hidden_tool_message(tool_call)
                        return

                    async with semaphore:
                        branch = copy.deepcopy(state) if is_parallel else state
                        branch_states[index] = branch
                        async for item in self._handle_single_tool(
                            tool_call, branch, config, origin_message_id=origin_message_id
                        ):
                            if isinstance(item, _ToolDone):
                                results[index] = item.result
                            else:
                                await shared_queue.put(item)
                        if is_parallel and self._branch_untouched(branch, baseline, results[index]):
                            branch_states[index] = None
                except GraphInterrupt as paused:
                    # A tool that called interrupt() paused the run; not a tool failure.
                    interrupts.append(paused)
                except Exception as e:
                    results[index] = self._failure_message(tool_call, e)
                    await shared_queue.put(
                        StreamChunk(
                            event=StreamEvent.ERROR,
                            data={
                                "status": "tool_failed",
                                "tool_name": name,
                                "tool_call_id": tool_call.get("id", ""),
                                "node": self.name,
                                "reason": "Tool execution failed",
                            },
                            thread_id=config.get("thread_id"),
                            run_id=config.get("run_id"),
                        )
                    )
                finally:
                    await shared_queue.put(_WORKER_DONE)

            tasks = [asyncio.create_task(_worker(i, tc)) for i, tc in enumerate(calls)]
            try:
                remaining = len(tasks)
                while remaining > 0:
                    item = await shared_queue.get()
                    if item is _WORKER_DONE:
                        remaining -= 1
                    else:
                        yield item
            finally:
                # Cancel and the consumer going away both end here: no tool outlives the run.
                pending = [t for t in tasks if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    await asyncio.wait(pending)

            if interrupts:
                raise interrupts[0]

            outcomes = [r for r in results if r is not None]
            if is_parallel:
                with_state = [
                    r
                    for r in results
                    if isinstance(r, dict) and isinstance(r.get("state"), AgentState)
                ]
                self._fold_parallel_branches(state, baseline, branch_states, results)
                for r in with_state:
                    # Merged already: point it at the shared state so it is not merged twice,
                    # and the node still reports a state update.
                    r["state"] = state

            # Results go on in call order, whichever tool finished first.
            for outcome in outcomes:
                if isinstance(outcome, dict) and isinstance(outcome.get("messages"), Message):
                    # The tool's message also goes on by itself, as before.
                    yield outcome["messages"]
                yield outcome

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

    async def _call_agent_node(  # noqa: PLR0915
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

        context = CallbackContext(
            invocation_type=InvocationType.AI,
            node_name=self.name,
            function_name=getattr(agent, "__class__", type(agent)).__name__,
            metadata={"config": config},
        )
        input_data = {"state": state, "config": config}
        last_message = state.context[-1] if state.context and len(state.context) > 0 else None

        event = EventModel.default(
            config,
            data={"state": state.model_dump()},
            event=Event.NODE_EXECUTION,
            content_type=[ContentType.STATE],
            node_name=self.name,
            extra={
                "node": self.name,
                "function_name": type(agent).__name__,
                "last_message": last_message.model_dump() if last_message else None,
            },
        )
        publish_event(event)

        try:
            input_data = await callback_mgr.execute_before_invoke(context, input_data)

            event.event_type = EventType.PROGRESS
            event.metadata["status"] = "Agent execution started"
            publish_event(event)

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

            result = {"state": state, "messages": messages, "next_node": None}
            result = await callback_mgr.execute_after_invoke(context, input_data, result)

            is_dict = isinstance(result, dict)
            final_messages = result.get("messages", messages) if is_dict else messages
            event.data["state"] = state.model_dump()
            event.event_type = EventType.END
            event.metadata["status"] = "Agent execution completed"
            event.data["messages"] = [m.model_dump() for m in final_messages]
            event.data["next_node"] = None
            if final_messages:
                last = final_messages[-1]
                event.content = last.text() if isinstance(last.content, list) else last.content
                if isinstance(last.content, list):
                    event.content_blocks = last.content
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
                logger.info("Node '%s' recovered from error using callback result", self.name)
                event.event_type = EventType.END
                event.metadata["status"] = "Agent execution recovered from error"
                event.data["message"] = recovery_result.model_dump()
                event.content_type = [ContentType.MESSAGE, ContentType.STATE]
                publish_event(event)
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
            logger.error("Node '%s' could not recover from error", self.name)
            event.event_type = EventType.ERROR
            event.metadata["status"] = f"Agent execution failed: {e}"
            event.data["error"] = str(e)
            event.content_type = [ContentType.ERROR, ContentType.STATE]
            publish_event(event)
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
