"""Streaming graph execution handler for TAF workflows.

This module provides the StreamHandler class, which manages the execution of graph workflows
with support for streaming output, interrupts, state persistence, and event publishing.
It enables incremental result processing, pause/resume capabilities, and robust error handling
for agent workflows that require real-time or chunked responses.
"""

from __future__ import annotations  # isort: skip_file

import asyncio
import logging
import time
from functools import partial
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any, TypeVar

from collections.abc import Callable

from injectq import inject, Inject

from tenxgraph.core.authz import public_config
from tenxgraph.core.exceptions import GraphStopRequested
from tenxgraph.core.graph.edge import Edge
from tenxgraph.core.graph.node import Node
from tenxgraph.core.graph.utils.guards import node_guards, node_metrics, stream_with_guards
from tenxgraph.core.graph.utils.error_text import client_error_text
from tenxgraph.storage.checkpointer import BaseCheckpointer
from tenxgraph.utils.logging import log_context_scope
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.stream_chunks import StreamChunk, StreamEvent
from tenxgraph.utils import END, ResponseGranularity, add_messages
from tenxgraph.utils.callbacks import CallbackManager, GraphLifecycleContext
from .handler_utils import (
    check_stop_requested,
    finalize_cancelled_run,
    interrupt_graph,
    is_remote_tool_marker,
    node_after,
)

from .handler_mixins import (
    BaseLoggingMixin,
    GraphRunMixin,
    InterruptConfigMixin,
)
from .utils import (
    LastSave,
    calculate_token_usage,
    process_node_result,
)
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.interrupt import (
    GraphInterrupt,
    node_scope,
    resume_values,
)


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger("tenxgraph.graph")


PAUSE_KINDS = ("interrupt", "before", "after", "remote_tool")


def is_pause_chunk(chunk: StreamChunk) -> bool:
    """True for any pause update: ``interrupt()``, interrupt_before/after, remote-tool wait.

    It is a control signal (the run stopped and waits for a resume), so it reaches clients at
    every response granularity, unlike the other ``UPDATES`` progress chunks.
    """
    data = chunk.data
    return (
        chunk.event == StreamEvent.UPDATES
        and isinstance(data, dict)
        and (
            data.get("status") == "interrupted"
            or "interrupt" in data
            or data.get("pause") in PAUSE_KINDS
        )
    )


class StreamHandler[StateT: AgentState](
    BaseLoggingMixin,
    InterruptConfigMixin,
    GraphRunMixin,
):
    """Handles streaming execution for graph workflows in TAF.

    StreamHandler manages the execution of agent workflows as directed graphs,
    supporting streaming output, pause/resume via interrupts, state persistence,
    and event publishing for monitoring and debugging. It enables incremental
    result processing and robust error handling for complex agent workflows.

    Attributes:
        nodes: Dictionary mapping node names to Node instances.
        edges: List of Edge instances defining graph connections and routing.
        interrupt_before: List of node names where execution should pause before execution.
        interrupt_after: List of node names where execution should pause after execution.

    Example:
        ```python
        handler = StreamHandler(nodes, edges)
        async for chunk in handler.stream(input_data, config, state):
            print(chunk)
        ```
    """

    @inject
    def __init__(
        self,
        nodes: dict[str, Node],
        edges: list[Edge],
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        get_node_factory: Callable[[str], Node] | None = None,
        node_aliases: dict[str, str] | None = None,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ):
        callback_mgr = fresh(callback_mgr)
        self.nodes: dict[str, Node] = nodes
        self.edges: list[Edge] = edges
        # Store factory for node lookup - enables override_node after compile
        # Uses the nodes dict reference directly - modifications to the dict are reflected
        self._get_node = get_node_factory if get_node_factory else lambda x: nodes[x]
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
        self._set_interrupts(interrupt_before, interrupt_after)
        self.callback_mgr = callback_mgr
        self.node_aliases: dict[str, str] = dict(node_aliases or {})

    async def _execute_node_stream(
        self,
        node: Node,
        node_name: str,
        config: dict[str, Any],
        state: StateT,
        checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],
    ) -> AsyncGenerator[Any]:
        """Stream a node under a deadline, cancellable by an out-of-band stop.

        The node's generator is driven by a task core owns (see ``stream_with_guards``), so
        a node that hangs can be stopped or timed out mid-flight, as on the invoke path. The
        deadline is the developer's ``node_timeout``; with none set the node has no limit.
        """
        guards = node_guards(node_name, config, fresh(checkpointer))
        with node_metrics(node_name):
            async for item in stream_with_guards(
                node.stream(config, state),  # type: ignore[arg-type]
                **guards,
            ):
                yield item

    async def _execute_graph(  # noqa: PLR0911, PLR0912, PLR0915
        self,
        state: StateT,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> AsyncIterable[StreamChunk]:
        """
        Execute the entire graph with support for interrupts and resuming.

        Why so many chunks are yielded?
        We allow user to set response type, if they want low granularity
        Only few chunks like Message will be sent to user
        """
        logger.info(
            "Starting graph execution from node '%s' at step %d",
            state.execution_meta.current_node,
            state.execution_meta.step,
        )
        messages: list[Message] = []
        messages_ids = set()
        # How many of `messages` a per-step checkpoint has already saved.
        persisted_upto = 0
        # The run's latest durable save, for the cancel cleanup (never two saves at once).
        last_save = LastSave()
        max_steps = config.get("recursion_limit", 25)
        logger.debug("Max steps limit set to %d", max_steps)

        last_human_messages = input_data.get("messages", []) or []
        # Stream initial input messages (e.g., human messages) so callers see full conversation
        # Only emit when present and avoid duplicates by tracking message_ids and existing context
        for m in last_human_messages:
            if m.message_id not in messages_ids:
                messages.append(m)
                messages_ids.add(m.message_id)
                yield StreamChunk(
                    event=StreamEvent.MESSAGE,
                    message=m,
                    metadata={
                        "status": "invoking_graph",
                        "reason": "initial human message",
                    },
                    thread_id=config.get("thread_id"),
                    run_id=config.get("run_id"),
                )

        current_node, step = self._start_node(state, config)

        # Build lifecycle context (shared across all hooks in this execution)
        lifecycle_context = GraphLifecycleContext(config=public_config(config))

        # Create event for graph execution
        event = EventModel.default(
            config,
            data={"state": state},
            content_type=[ContentType.STATE],
            extra={"step": step, "current_node": current_node},
            event=Event.GRAPH_EXECUTION,
            node_name=current_node,
        )

        yield StreamChunk(
            event=StreamEvent.UPDATES,
            data={
                "status": "invoking_graph",
                "node": current_node,
                "step": step,
                "max_steps": max_steps,
            },
            thread_id=config.get("thread_id"),
            run_id=config.get("run_id"),
        )

        try:
            while current_node != END and step < max_steps:
                logger.debug("Executing step %d at node '%s'", step, current_node)

                # TODO: check if ai called for a tool in that case we should remove last message
                res = await check_stop_requested(
                    state,
                    current_node,
                    event,
                    messages,
                    config,
                    last_save=last_save,
                )
                if res:
                    event.event_type = EventType.INTERRUPTED
                    event.metadata["status"] = "Graph execution stopped by request"
                    event.data["state"] = state
                    publish_event(event)
                    # stream updated state and updates
                    yield StreamChunk(
                        event=StreamEvent.UPDATES,
                        data={
                            "status": "invoking_node",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "reason": "Graph execution stopped by request",
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
                    return

                await self._begin_step(state, config, event, step, current_node)
                event.data["step"] = step
                event.data["current_node"] = current_node
                event.metadata["status"] = f"Executing step {step} at node '{current_node}'"
                publish_event(event)

                if await self._interrupt_before(
                    current_node, state, config, event, messages[persisted_upto:], last_save
                ):
                    yield StreamChunk(
                        event=StreamEvent.UPDATES,
                        data={
                            "status": "invoking_node",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "reason": "Graph execution interrupted before node execution",
                            "pause": "before",
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
                    return

                # Execute current node - use factory for override support
                logger.debug("Executing node '%s'", current_node)
                node = self._get_node(current_node)

                has_hooks, old_state_snapshot, node_event = self._start_node_event(
                    state, config, step, current_node
                )
                logger.debug("Node '%s' execution started", current_node)

                # Process result and get next node
                yield StreamChunk(
                    event=StreamEvent.UPDATES,
                    data={
                        "status": "node_invoked",
                        "node": current_node,
                        "step": step,
                        "max_steps": max_steps,
                    },
                    thread_id=config.get("thread_id"),
                    run_id=config.get("run_id"),
                )

                # From Here message no need to stream its already streamed
                # from execute node function, only stream updates and state
                next_node = None
                remote_pending = False
                try:
                    with node_scope(config, current_node, resume_values(state)):
                        async for rs in self._execute_node_stream(
                            node, current_node, config, state
                        ):
                            # Allow stop to break inner result loop as well
                            if isinstance(rs, StreamChunk):
                                yield rs

                            # A tool call the client runs. Keep draining, so parallel server
                            # tools finish and are saved, then pause after this node.
                            elif is_remote_tool_marker(rs):
                                remote_pending = True

                            elif isinstance(rs, Message) and not rs.delta:
                                if rs.message_id not in messages_ids:
                                    messages.append(rs)
                                    messages_ids.add(rs.message_id)

                            elif isinstance(rs, dict) and "is_non_streaming" in rs:
                                if rs["is_non_streaming"]:
                                    new_state = rs.get("state", None)
                                    if new_state:
                                        state = new_state
                                        yield StreamChunk(
                                            event=StreamEvent.STATE,
                                            state=state,
                                            metadata={
                                                "node": current_node,
                                                "step": step,
                                            },
                                            thread_id=config.get("thread_id"),
                                            run_id=config.get("run_id"),
                                        )

                                    new_messages = rs.get("messages", [])
                                    for m in new_messages:
                                        if is_remote_tool_marker(m):
                                            remote_pending = True
                                            continue
                                        if m.message_id not in messages_ids and not m.delta:
                                            messages.append(m)
                                            messages_ids.add(m.message_id)
                                    next_node = rs.get("next_node", next_node)
                                else:
                                    # Streaming path completed: ensure any collected
                                    # messages are persisted
                                    new_messages = rs.get("messages", [])
                                    for m in new_messages:
                                        if is_remote_tool_marker(m):
                                            remote_pending = True
                                            continue
                                        if m.message_id not in messages_ids and not m.delta:
                                            messages.append(m)
                                            messages_ids.add(m.message_id)
                                    next_node = rs.get("next_node", next_node)
                            else:
                                # Process as node result (non-streaming path)
                                try:
                                    state, new_messages, next_node = await process_node_result(
                                        rs,
                                        state,
                                        [],
                                    )
                                    for m in new_messages:
                                        if m.message_id not in messages_ids and not m.delta:
                                            messages.append(m)
                                            messages_ids.add(m.message_id)
                                except Exception as e:
                                    logger.error("Failed to process node result: %s", e)
                except GraphInterrupt as paused:
                    await self._pause_at_interrupt(
                        paused,
                        current_node,
                        state,
                        config,
                        event,
                        messages[persisted_upto:],
                        last_save,
                    )
                    yield StreamChunk(
                        event=StreamEvent.UPDATES,
                        data={
                            "status": "interrupted",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "pause": "interrupt",
                            "reason": "Graph execution paused by interrupt()",
                            "interrupt": paused.interrupt.model_dump(mode="json"),
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
                    return
                except GraphStopRequested:
                    await self._stop_mid_node(
                        current_node, state, config, event, messages, last_save
                    )
                    yield StreamChunk(
                        event=StreamEvent.UPDATES,
                        data={
                            "status": "invoking_node",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "reason": "Graph execution stopped by request",
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
                    return

                self._end_node_event(node_event, next_node, current_node, messages)

                # Add collected messages to state context
                if messages:
                    state.context = add_messages(state.context, messages)
                    logger.debug("Added %d messages to state context", len(messages))
                    yield StreamChunk(
                        event=StreamEvent.STATE,
                        state=state,
                        metadata={
                            "node": current_node,
                            "step": step,
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )

                state = await self._publish_node_update(
                    has_hooks,
                    lifecycle_context,
                    current_node,
                    old_state_snapshot,
                    state,
                    config,
                    event,
                    step,
                    messages,
                    stringify=True,
                )

                if remote_pending:
                    # The client runs this node's remaining tool calls. Pause; the run
                    # resumes after this node once their results arrive.
                    await interrupt_graph(
                        current_node,
                        state,
                        config,
                        messages=messages[persisted_upto:],
                        last_save=last_save,
                    )
                    yield StreamChunk(
                        event=StreamEvent.UPDATES,
                        data={
                            "status": "node_invoked",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "reason": "Remote tool call - graph interrupted",
                            "pause": "remote_tool",
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
                    return

                # Check stop again after node execution. The node finished, so the stop is
                # saved at the node that follows it.
                if await check_stop_requested(
                    state,
                    current_node,
                    event,
                    messages,
                    config,
                    resume_after=partial(node_after, current_node, next_node, state, self.edges),
                    last_save=last_save,
                ):
                    yield StreamChunk(
                        event=StreamEvent.UPDATES,
                        data={
                            "status": "node_invoked",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "reason": "Graph execution stopped by request",
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
                    return

                is_interrupted_requested, next_node = await self._interrupt_after(
                    current_node,
                    next_node,
                    state,
                    config,
                    event,
                    messages[persisted_upto:],
                    last_save,
                )
                if is_interrupted_requested:
                    yield StreamChunk(
                        event=StreamEvent.UPDATES,
                        data={
                            "status": "node_invoked",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "reason": "Graph execution interrupted after node execution",
                            "pause": "after",
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )

                current_node, step, persisted_upto = await self._advance(
                    current_node,
                    next_node,
                    state,
                    config,
                    event,
                    step,
                    messages,
                    persisted_upto,
                    last_save,
                )

                yield StreamChunk(
                    event=StreamEvent.UPDATES,
                    state=state,
                    data={
                        "status": "node_invoked",
                        "node": current_node,
                        "step": step,
                        "max_steps": max_steps,
                    },
                    thread_id=config.get("thread_id"),
                    run_id=config.get("run_id"),
                )

                if is_interrupted_requested:
                    return

                if step >= max_steps and current_node != END:
                    error = await self._recursion_limit_hit(
                        state, config, event, step, max_steps, current_node
                    )
                    yield StreamChunk(
                        event=StreamEvent.ERROR,
                        state=state,
                        data={
                            "status": "graph_invoked",
                            "node": current_node,
                            "step": step,
                            "max_steps": max_steps,
                            "reason": self.RECURSION_MESSAGE,
                        },
                        thread_id=config.get("thread_id"),
                        run_id=config.get("run_id"),
                    )
                    raise error

            state, is_context_trimmed = await self._complete_run(
                state,
                config,
                event,
                lifecycle_context,
                messages,
                step,
                current_node,
                last_save,
                stringify=True,
            )

            # Include messages list for token calculation in stream method
            yield StreamChunk(
                event=StreamEvent.UPDATES,
                state=state,
                data={
                    "status": "graph_invoked",
                    "node": current_node,
                    "step": step,
                    "max_steps": max_steps,
                    "is_context_trimmed": is_context_trimmed,
                    "reason": "Graph execution completed successfully",
                    # Internal: messages from current run for token calculation
                    "_messages": messages,
                },
                thread_id=config.get("thread_id"),
                run_id=config.get("run_id"),
            )

        except asyncio.CancelledError:
            # Cancelled mid-run (client gone, shutdown, timeout around the run): save the
            # thread as INTERRUPTED so the next run can use it, then let the cancel go on.
            await finalize_cancelled_run(
                state, config, messages, event, current_node, last_save=last_save
            )
            raise
        except Exception as e:
            state = await self._fail_run(
                e,
                state,
                config,
                event,
                lifecycle_context,
                messages,
                step,
                current_node,
                last_save,
            )

            yield StreamChunk(
                event=StreamEvent.ERROR,
                state=state,
                data={
                    "status": "invoked_graph",
                    "node": current_node,
                    "step": step,
                    "max_steps": max_steps,
                    "reason": client_error_text(e),
                },
                thread_id=config.get("thread_id"),
                run_id=config.get("run_id"),
            )

            raise e

    async def stream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        default_state: StateT,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> AsyncGenerator[StreamChunk]:
        """Execute the graph asynchronously with streaming output.

        Runs the graph workflow from start to finish, yielding incremental results
        as they become available. Automatically detects whether to start a fresh
        execution or resume from an interrupted state, supporting pause/resume
        and checkpointing.

        Args:
            input_data: Input dictionary for graph execution. For new executions,
                should contain 'messages' key with initial messages. For resumed
                executions, can contain additional data to merge.
            config: Configuration dictionary containing execution settings and context.
            default_state: Initial or template AgentState for workflow execution.
            response_granularity: Level of detail in the response (LOW, PARTIAL, FULL).

        Yields:
            Message objects representing incremental results from graph execution.
            The exact type and frequency of yields depends on node implementations
            and workflow configuration.

        Raises:
            GraphRecursionError: If execution exceeds recursion limit.
            ValueError: If input_data is invalid for new execution.
            Various exceptions: Depending on node execution failures.

        Example:
            ```python
            async for chunk in handler.stream(input_data, config, state):
                print(chunk)
            ```
        """
        config = config or {}
        # Bind run correlation for every log line emitted underneath this run, and put the
        # earlier ids back when the stream ends.
        with log_context_scope(config):
            async for chunk in self._stream(
                input_data, config, default_state, response_granularity
            ):
                yield chunk

    async def _stream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        default_state: StateT,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> AsyncGenerator[StreamChunk]:
        logger.info(
            "Starting asynchronous graph execution with %d input keys, granularity=%s",
            len(input_data) if input_data else 0,
            response_granularity,
        )
        config = config or {}
        input_data = input_data or {}

        start_time = time.time()

        state: StateT = await self._load_state(input_data, config, default_state)

        cfg = public_config(config)
        if "user" in cfg:
            # This will be available when you are calling
            # vi agentflow api
            del cfg["user"]

        event = EventModel.default(
            config,
            data={"state": state},
            content_type=[ContentType.STATE],
            extra={
                "is_interrupted": state.is_interrupted(),
                "current_node": state.execution_meta.current_node,
                "step": state.execution_meta.step,
                "config": cfg,
                "response_granularity": response_granularity.value,
            },
        )

        # Publish graph initialization event
        publish_event(event)

        # Apply a resume, then fire on_graph_start before execution begins
        state, config = await self._start_run(state, input_data, config)

        # Now start Execution
        # Execute graph
        logger.debug("Beginning graph execution")
        result = self._execute_graph(state, input_data, config)

        # Track messages from current run for token calculation
        current_run_messages = []

        # Stream results based on response granularity
        async for chunk in result:
            # Extract messages from final completion chunk (internal use only)
            if (
                chunk.event == StreamEvent.UPDATES
                and chunk.data
                and chunk.data.get("status") == "graph_invoked"
            ):
                current_run_messages = chunk.data.pop("_messages", [])

            match response_granularity:
                case ResponseGranularity.FULL:
                    yield chunk
                case ResponseGranularity.PARTIAL:
                    if chunk.event != StreamEvent.UPDATES or is_pause_chunk(chunk):
                        yield chunk
                case ResponseGranularity.LOW:
                    if chunk.event in [StreamEvent.MESSAGE, StreamEvent.ERROR] or is_pause_chunk(
                        chunk
                    ):
                        yield chunk

        # Publish graph completion event
        time_taken = time.time() - start_time
        logger.info("Graph execution finished in %.2f seconds", time_taken)

        # Calculate token usage from current run messages only
        token_usage = calculate_token_usage(current_run_messages)

        event.event_type = EventType.END
        event.metadata.update(
            {
                "time_taken": time_taken,
                "state": state,
                "step": state.execution_meta.step,
                "current_node": state.execution_meta.current_node,
                "is_interrupted": state.is_interrupted(),
                "total_messages": len(state.context) if state.context else 0,
                **token_usage,
            }
        )
        publish_event(event)
        if response_granularity == ResponseGranularity.FULL:
            yield StreamChunk(
                event=StreamEvent.UPDATES,
                state=state,
                data={
                    "status": "graph_invoked",
                    "reason": "Graph execution finished",
                    "time_taken": time_taken,
                    "is_interrupted": state.is_interrupted(),
                    "total_messages": len(state.context) if state.context else 0,
                    **token_usage,
                },
                thread_id=config.get("thread_id"),
                run_id=config.get("run_id"),
            )
