"""Core utility functions for graph execution and state management.

This module provides essential utilities for TAF graph execution, including
state management, message processing, response formatting, and execution flow control.
These functions handle the low-level operations that support graph workflow execution.

The utilities in this module are designed to work with TAF's dependency injection
system and provide consistent interfaces for common operations across different
execution contexts.

Key functionality areas:
- State loading, creation, and synchronization
- Message processing and deduplication
- Response formatting based on granularity levels
- Node execution result processing
- Interrupt handling and execution flow control
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from injectq import Inject

from tenxgraph.core.state import AgentState, ExecutionStatus
from tenxgraph.core.state.execution_state import StopRequestStatus
from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import RemoteToolCallBlock
from tenxgraph.runtime.publisher.events import EventModel, EventType
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.storage.checkpointer import BaseCheckpointer
from tenxgraph.utils import (
    START,
)
from tenxgraph.utils.callbacks import CallbackManager, GraphLifecycleContext
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.interrupt import (
    RESUME_KEY,
    Interrupt,
    pause_at,
    pending_interrupt,
    record_resume,
)

from .utils import reload_state, sync_data


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger("tenxgraph.graph")


async def check_interrupted[StateT: AgentState](
    state: StateT,
    input_data: dict[str, Any],
    config: dict[str, Any],
    callback_mgr: CallbackManager = Inject[CallbackManager],
) -> tuple[StateT, dict[str, Any]]:
    callback_mgr = fresh(callback_mgr)
    request = pending_interrupt(state)
    if request is not None:
        # Paused by interrupt(): the retried node needs the answer, and a plain new message
        # cannot stand in for it.
        if RESUME_KEY not in input_data:
            raise ValueError(
                f"Thread is paused at interrupt '{request.id}' in node '{request.node}'. "
                f"Resume it with {{'{RESUME_KEY}': value}}."
            )
        record_resume(state, input_data[RESUME_KEY])
    elif RESUME_KEY in input_data:
        raise ValueError("Nothing to resume: this thread is not paused at an interrupt()")

    if state.is_interrupted():
        logger.info(
            "Resuming from interrupted state at node '%s'", state.execution_meta.current_node
        )

        # Fire on_resume hook before clearing the interrupt
        if callback_mgr and callback_mgr._lifecycle_hooks:
            context = GraphLifecycleContext(config=config)
            resumed_node = state.execution_meta.interrupted_node or ""
            modified = await callback_mgr.fire_on_resume(
                context,
                resumed_node=resumed_node,
                state=state,
                resume_data=input_data,
            )
            if modified is not None and modified is not state:
                state = modified  # type: ignore[assignment]

        # Paused for a client-side tool: the resumed run continues after that node.
        if state.execution_meta.interrupt_reason == REMOTE_TOOL_REASON:
            config[RESUME_AFTER_NODE_KEY] = state.execution_meta.interrupted_node

        # Save the interrupted node info before clearing so we don't re-interrupt
        config["_skip_interrupt_at"] = {
            "node": state.execution_meta.interrupted_node,
            "status": state.execution_meta.status,
        }
        # This is a resume case - clear interrupt and merge input data
        if input_data:
            config["resume_data"] = input_data
            logger.debug("Added resume data with %d keys", len(input_data))
        state.clear_interrupt()
    elif not input_data.get("messages") and not state.context:
        # This is a fresh execution - validate input data
        error_msg = "Input data must contain 'messages' for new execution."
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif state.execution_meta.status == ExecutionStatus.COMPLETED:
        # Previous execution completed - reset to entry point for new execution
        logger.info(
            "Previous execution completed. Resetting to entry point for new execution "
            "with %d messages",
            len(input_data.get("messages", [])),
        )
        # Reset execution metadata for fresh start
        state.execution_meta.current_node = START
        state.execution_meta.step = 0
        state.execution_meta.status = ExecutionStatus.RUNNING
        state.execution_meta.interrupted_node = None
        state.execution_meta.interrupt_reason = None
        state.execution_meta.interrupt_data = None
    else:
        # Fresh execution, state is already at START
        logger.info(
            "Starting fresh execution with %d messages", len(input_data.get("messages", []))
        )

    return state, config


async def check_and_handle_interrupt[StateT: AgentState](
    current_node: str,
    interrupt_type: str,
    state: StateT,
    config: dict[str, Any],
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    callback_mgr: CallbackManager = Inject[CallbackManager],
) -> bool:
    """Check for interrupts and save state if needed. Returns True if interrupted."""
    callback_mgr = fresh(callback_mgr)
    interrupt_nodes: list[str] = (
        interrupt_before if interrupt_type == "before" else interrupt_after
    ) or []

    # Check if we just resumed from an interrupt at this node with this type
    skip_info = config.get("_skip_interrupt_at", {})
    if skip_info.get("node") == current_node:
        expected_status = (
            ExecutionStatus.INTERRUPTED_BEFORE
            if interrupt_type == "before"
            else ExecutionStatus.INTERRUPTED_AFTER
        )
        if skip_info.get("status") == expected_status:
            logger.debug(
                "Skipping %s interrupt check for node '%s' - just resumed from it",
                interrupt_type,
                current_node,
            )
            # Clear the flag after using it once
            config.pop("_skip_interrupt_at", None)
            return False

    if current_node in interrupt_nodes:
        status = (
            ExecutionStatus.INTERRUPTED_BEFORE
            if interrupt_type == "before"
            else ExecutionStatus.INTERRUPTED_AFTER
        )
        state.set_interrupt(
            current_node,
            f"interrupt_{interrupt_type}: {current_node}",
            status,
        )

        # Fire on_interrupt hook
        if callback_mgr and callback_mgr._lifecycle_hooks:
            context = GraphLifecycleContext(config=config)
            await callback_mgr.fire_on_interrupt(
                context,
                interrupted_node=current_node,
                interrupt_type=interrupt_type,
                state=state,
            )

        # Save state and interrupt
        await sync_data(
            state=state,
            config=config,
            messages=[],
            trim=True,
        )
        logger.debug("Node '%s' interrupted", current_node)
        return True

    logger.debug(
        "No interrupts found for node '%s', continuing execution",
        current_node,
    )
    return False


# ``execution_meta.interrupt_reason`` while a tool call waits for the client to run it.
REMOTE_TOOL_REASON = "remote_tool_call"

# Run-config key telling the loop to continue after the node that paused for a client tool.
RESUME_AFTER_NODE_KEY = "_resume_after_node"


def is_remote_tool_marker(item: Any) -> bool:
    """Whether ``item`` is the placeholder a ToolNode returns for a client-executed call."""
    return isinstance(item, Message) and any(
        isinstance(block, RemoteToolCallBlock) for block in item.content or []
    )


def split_remote_calls(result: Any) -> tuple[list[Message], Any]:
    """Separate client-tool placeholders from a node result.

    Returns the placeholders and the result without them, so the rest (for example parallel
    server tools that finished) is merged into the state as usual.
    """
    if is_remote_tool_marker(result):
        return [result], []
    if isinstance(result, list):
        markers = [item for item in result if is_remote_tool_marker(item)]
        return markers, [item for item in result if not is_remote_tool_marker(item)]
    if isinstance(result, dict) and isinstance(result.get("messages"), list):
        markers = [m for m in result["messages"] if is_remote_tool_marker(m)]
        if markers:
            kept = [m for m in result["messages"] if not is_remote_tool_marker(m)]
            return markers, {**result, "messages": kept}
    return [], result


async def interrupt_graph[StateT: AgentState](
    current_node: str,
    state: StateT,
    config: dict[str, Any],
    callback_mgr: CallbackManager = Inject[CallbackManager],
) -> bool:
    """Pause after ``current_node`` until the client sends the results of its tool calls.

    ``current_node`` stays put: which node comes next can depend on those results, so the
    resumed run works it out once they are in the context (see ``RESUME_AFTER_NODE_KEY``).
    """
    callback_mgr = fresh(callback_mgr)
    status = ExecutionStatus.INTERRUPTED_AFTER
    state.set_interrupt(
        current_node,
        REMOTE_TOOL_REASON,
        status,
    )

    # Fire on_interrupt hook
    if callback_mgr and callback_mgr._lifecycle_hooks:
        context = GraphLifecycleContext(config=config)
        await callback_mgr.fire_on_interrupt(
            context,
            interrupted_node=current_node,
            interrupt_type="remote_tool",
            state=state,
        )

    # Save state and interrupt
    await sync_data(
        state=state,
        config=config,
        messages=[],
        trim=False,
    )
    logger.debug("Node '%s' interrupted", current_node)
    return True


async def pause_for_interrupt[StateT: AgentState](
    current_node: str,
    state: StateT,
    config: dict[str, Any],
    request: Interrupt,
    callback_mgr: CallbackManager = Inject[CallbackManager],
) -> None:
    """Save ``state`` paused before ``current_node`` because it called ``interrupt()``.

    Resuming re-runs the node, and ``interrupt()`` then returns the resume value.
    """
    callback_mgr = fresh(callback_mgr)
    pause_at(state, request)

    if callback_mgr and callback_mgr._lifecycle_hooks:
        context = GraphLifecycleContext(config=config)
        await callback_mgr.fire_on_interrupt(
            context,
            interrupted_node=current_node,
            interrupt_type="interrupt",
            state=state,
        )

    await sync_data(
        state=state,
        config=config,
        messages=[],
        trim=False,
    )
    logger.info("Node '%s' paused at interrupt '%s'", current_node, request.id)


async def check_stop_requested[StateT: AgentState](
    state: StateT,
    current_node: str,
    event: EventModel,
    messages: list[Message],
    config: dict[str, Any],
    callback_mgr: CallbackManager = Inject[CallbackManager],
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],
) -> bool:
    """Check if a stop has been requested externally."""
    callback_mgr = fresh(callback_mgr)
    checkpointer = fresh(checkpointer)
    state = await reload_state(config, state)  # type: ignore

    # A stop request lives in its own checkpointer key, NOT in the cached state.
    # This loop rewrites the state cache after every node, so a flag carried in
    # the state blob can be overwritten by our own flag-less copy before we get
    # here. The dedicated key is never written by the loop, so it survives.
    # The state blob is still consulted, so a custom checkpointer that does not
    # back the stop key keeps working the way it did before.
    stop_requested = state.is_stopped_requested()
    if not stop_requested and checkpointer:
        stop_requested = await checkpointer.ais_stop_requested(config)
        if stop_requested:
            state.execution_meta.stop_current_execution = StopRequestStatus.STOP_REQUESTED

    # Check if a stop was requested externally (e.g., frontend)
    if stop_requested:
        logger.info(
            "Stop requested for thread '%s' at node '%s'",
            config.get("thread_id"),
            current_node,
        )
        # Consume the request so it cannot leak into the next run on this thread.
        if checkpointer:
            await checkpointer.aclear_stop_request(config)

        state.set_interrupt(
            current_node,
            "stop_requested",
            ExecutionStatus.INTERRUPTED_AFTER,
            data={"source": "stop", "info": "requested via is_stopped_requested"},
        )

        # Fire on_interrupt hook
        if callback_mgr and callback_mgr._lifecycle_hooks:
            context = GraphLifecycleContext(config=config)
            await callback_mgr.fire_on_interrupt(
                context,
                interrupted_node=current_node,
                interrupt_type="stop",
                state=state,
            )

        await sync_data(state=state, config=config, messages=messages, trim=True)
        event.event_type = EventType.INTERRUPTED
        event.metadata["interrupted"] = "Stop"
        event.metadata["status"] = "Graph execution stopped by request"
        event.data["state"] = state.model_dump()
        publish_event(event)
        return True
    return False
