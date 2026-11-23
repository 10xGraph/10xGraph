"""Shared mixins for graph and node handler classes.

This module provides lightweight mixins that add common functionality to handler
classes without changing their core runtime behavior. The mixins follow the
composition pattern to keep responsibilities explicit and allow handlers to
inherit only the capabilities they need.

The mixins provide structured logging, configuration management, and other
cross-cutting concerns that are commonly needed across different handler types.
By using mixins, the core handler logic remains focused while gaining these
shared capabilities.

Typical usage:
    ```python
    class MyHandler(BaseLoggingMixin, InterruptConfigMixin):
        def __init__(self):
            self._set_interrupts(["node1"], ["node2"])
            self._log_start("Handler initialized")
    ```
"""

from __future__ import annotations

import logging
from typing import Any

from tenxgraph.core.authz import public_config
from tenxgraph.core.exceptions import GraphRecursionError
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.publish import publish_event, publish_lifecycle
from tenxgraph.utils import END, START
from tenxgraph.utils.callbacks import GraphLifecycleContext
from tenxgraph.utils.interrupt import GraphInterrupt, clear_resume_values
from tenxgraph.utils.logging import set_log_context

from .error_text import client_error_text
from .handler_utils import (
    RESUME_AFTER_NODE_KEY,
    check_and_handle_interrupt,
    check_interrupted,
    check_stop_requested,
    close_unanswered_tool_calls,
    pause_for_interrupt,
    publish_graph_end,
    publish_graph_error,
    publish_state_update,
    resolve_stored_node,
    skip_answered_tool_node,
)
from .tool_dispatch import validate_goto
from .utils import LastSave, call_realtime_sync, get_next_node, load_or_create_state, sync_data


logger = logging.getLogger("tenxgraph.graph")


class BaseLoggingMixin:
    """Provides structured logging helpers for handler classes.

    This mixin adds consistent logging capabilities to handler classes without
    requiring them to manage logger instances directly. It automatically creates
    loggers based on the module name and provides convenience methods for
    common logging operations.

    The mixin is designed to be lightweight and non-intrusive, adding only
    logging functionality without affecting the core behavior of the handler.

    Attributes:
        _logger: Cached logger instance for the handler class.

    Example:
        ```python
        class MyHandler(BaseLoggingMixin):
            def process(self):
                self._log_start("Processing started")
                try:
                    # Do work
                    self._log_debug("Work completed successfully")
                except Exception as e:
                    self._log_error("Processing failed: %s", e)
        ```
    """

    @property
    def _logger(self) -> logging.Logger:
        """Get or create a logger instance for this handler.

        Creates a logger using the handler's module name, providing consistent
        logging across different handler instances while maintaining proper
        logger hierarchy and configuration.

        Returns:
            Logger instance configured for this handler's module.
        """
        return logging.getLogger("tenxgraph.graph")

    def _log_start(self, msg: str, *args: Any) -> None:
        """Log an informational message for process start/initialization.

        Args:
            msg: Log message format string.
            *args: Arguments for message formatting.
        """
        self._logger.info(msg, *args)

    def _log_debug(self, msg: str, *args: Any) -> None:
        """Log a debug message for detailed execution information.

        Args:
            msg: Log message format string.
            *args: Arguments for message formatting.
        """
        self._logger.debug(msg, *args)

    def _log_error(self, msg: str, *args: Any) -> None:
        """Log an error message for exceptional conditions.

        Args:
            msg: Log message format string.
            *args: Arguments for message formatting.
        """
        self._logger.error(msg, *args)


class InterruptConfigMixin:
    """Manages interrupt configuration for graph-level execution handlers.

    This mixin provides functionality to store and manage interrupt points
    configuration for graph execution. Interrupts allow graph execution to be
    paused before or after specific nodes for debugging, human intervention,
    or checkpoint creation.

    The mixin maintains separate lists for "before" and "after" interrupts,
    allowing fine-grained control over when graph execution should pause.

    Attributes:
        interrupt_before: List of node names where execution should pause
            before node execution begins.
        interrupt_after: List of node names where execution should pause
            after node execution completes.

    Example:
        ```python
        class GraphHandler(InterruptConfigMixin):
            def __init__(self):
                self._set_interrupts(
                    interrupt_before=["approval_node"], interrupt_after=["data_processing"]
                )
        ```
    """

    interrupt_before: list[str] | None
    interrupt_after: list[str] | None

    def _set_interrupts(
        self,
        interrupt_before: list[str] | None,
        interrupt_after: list[str] | None,
    ) -> None:
        """Configure interrupt points for graph execution control.

        Sets up the interrupt configuration for this handler, defining which
        nodes should trigger execution pauses. This method normalizes None
        values to empty lists for consistent handling.

        Args:
            interrupt_before: List of node names where execution should be
                interrupted before the node begins execution. Pass None to
                disable before-interrupts.
            interrupt_after: List of node names where execution should be
                interrupted after the node completes execution. Pass None to
                disable after-interrupts.

        Note:
            This method should be called during handler initialization to
            establish the interrupt configuration before graph execution begins.
        """
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []


def mirror_last_message(event: EventModel, messages: list[Message], stringify: bool) -> None:
    """Copy the last message's text and blocks onto ``event``.

    ``stringify`` wraps the text in ``str()``, as the stream path has always done.
    """
    if messages:
        last = messages[-1]
        content = last.text() if isinstance(last.content, list) else last.content
        event.content = str(content) if stringify else content  # type: ignore[assignment]
        if isinstance(last.content, list):
            event.content_blocks = last.content


class GraphRunMixin:
    """Steps of the graph loop shared by ``InvokeHandler`` and ``StreamHandler``.

    Each method is one step both loops take the same way; the loops keep their own
    control flow and what they hand to the caller (a result or stream chunks). The host
    class provides the attributes below.
    """

    nodes: dict[str, Any]
    edges: list[Any]
    node_aliases: dict[str, str]
    interrupt_before: list[str]
    interrupt_after: list[str]
    callback_mgr: Any
    _get_node: Any

    RECURSION_MESSAGE = "Graph execution exceeded maximum steps"

    def _has_hooks(self) -> bool:
        return bool(self.callback_mgr and self.callback_mgr._lifecycle_hooks)

    async def _load_state(
        self, input_data: dict[str, Any], config: dict[str, Any], default_state: AgentState
    ) -> Any:
        """Load the thread's state, or start a new one from ``default_state``."""
        logger.debug("Loading or creating state from input data")
        state = await load_or_create_state(input_data, config, default_state)
        logger.debug(
            "State loaded: interrupted=%s, current_node=%s, step=%d",
            state.is_interrupted(),
            state.execution_meta.current_node,
            state.execution_meta.step,
        )
        return state

    async def _start_run(
        self, state: Any, input_data: dict[str, Any], config: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """Apply a resume, then run ``on_graph_start`` and publish ``graph_start``."""
        state, config = await check_interrupted(state, input_data, config)
        if self._has_hooks():
            lc_context = GraphLifecycleContext(config=public_config(config))
            state = await self.callback_mgr.fire_on_graph_start(lc_context, state)
        publish_lifecycle(config, "graph_start", {})
        return state, config

    def _start_node(self, state: AgentState, config: dict[str, Any]) -> tuple[str, int]:
        """The node and step this run starts at."""
        stored_node = state.execution_meta.current_node
        step = state.execution_meta.step
        # A deploy may have renamed or removed the node this thread was stored at.
        current_node = resolve_stored_node(
            stored_node, self.nodes, self.node_aliases, config.get("thread_id")
        )
        if current_node != stored_node:
            state.execution_meta.current_node = current_node
            if state.execution_meta.interrupted_node == stored_node:
                state.execution_meta.interrupted_node = current_node
        # Paused for client-side tools whose results are now in the context: move past
        # that node (the next node can depend on those results).
        if config.pop(RESUME_AFTER_NODE_KEY, None) in (stored_node, current_node):
            current_node = get_next_node(current_node, state, self.edges)
        if current_node != END:
            current_node = skip_answered_tool_node(
                self._get_node(current_node),
                current_node,
                state,
                lambda n: get_next_node(n, state, self.edges),
            )
        return current_node, step

    async def _begin_step(
        self,
        state: AgentState,
        config: dict[str, Any],
        event: EventModel,
        step: int,
        current_node: str,
    ) -> None:
        """Record the node and step about to run; the caller publishes ``event``."""
        state.set_current_node(current_node)
        state.execution_meta.step = step
        await call_realtime_sync(state, config)
        event.data["state"] = state
        event.metadata["step"] = step
        event.metadata["current_node"] = current_node
        event.event_type = EventType.PROGRESS

    async def _interrupt_before(
        self,
        current_node: str,
        state: AgentState,
        config: dict[str, Any],
        event: EventModel,
        messages: list[Message],
        last_save: LastSave,
    ) -> bool:
        """Pause before ``current_node`` when it is in ``interrupt_before``."""
        if not await check_and_handle_interrupt(
            current_node,
            "before",
            state,
            config,
            interrupt_before=self.interrupt_before,
            interrupt_after=self.interrupt_after,
            messages=messages,
            last_save=last_save,
        ):
            return False
        logger.info("Graph execution interrupted before node '%s'", current_node)
        event.event_type = EventType.INTERRUPTED
        event.metadata["interrupted"] = "Before"
        event.metadata["status"] = "Graph execution interrupted before node execution"
        event.data["interrupted"] = "Before"
        publish_event(event)
        return True

    def _start_node_event(
        self, state: AgentState, config: dict[str, Any], step: int, current_node: str
    ) -> tuple[bool, AgentState, EventModel]:
        """Publish the node START event and scope logs to the node.

        Returns whether lifecycle hooks exist, the state before the node (a copy only
        when ``on_state_update`` hooks need it) and the node event.
        """
        has_hooks = self._has_hooks()
        old_state_snapshot = state.model_copy(deep=True) if has_hooks else state
        node_event = EventModel.default(
            config,
            data={"step": step, "state": state},
            event=Event.NODE_EXECUTION,
            event_type=EventType.START,
            content_type=[ContentType.STATE],
            node_name=current_node,
        )
        publish_event(node_event)
        config["_node_name"] = current_node
        # Narrow the correlation context to the node now executing.
        set_log_context(node=current_node)
        return has_hooks, old_state_snapshot, node_event

    async def _pause_at_interrupt(
        self,
        paused: GraphInterrupt,
        current_node: str,
        state: AgentState,
        config: dict[str, Any],
        event: EventModel,
        messages: list[Message],
        last_save: LastSave,
    ) -> None:
        """interrupt() inside the node: save the pause. Resuming re-runs this node."""
        await pause_for_interrupt(
            current_node,
            state,
            config,
            paused.interrupt,
            messages=messages,
            last_save=last_save,
        )
        event.event_type = EventType.INTERRUPTED
        event.metadata["interrupted"] = "Interrupt"
        event.metadata["status"] = "Graph execution paused by interrupt()"
        event.data["interrupted"] = "Interrupt"
        event.data["interrupt"] = paused.interrupt.model_dump(mode="json")
        publish_event(event)

    async def _stop_mid_node(
        self,
        current_node: str,
        state: AgentState,
        config: dict[str, Any],
        event: EventModel,
        messages: list[Message],
        last_save: LastSave,
    ) -> None:
        """The node was cancelled because a stop arrived: finish as a between-nodes stop.

        This is control flow, not a failure: the run is marked stopped, saved and
        reported exactly as a stop seen between nodes.
        """
        logger.info("Node '%s' cancelled by stop request; finalizing run", current_node)
        await check_stop_requested(
            state,
            current_node,
            event,
            messages,
            config,
            last_save=last_save,
        )

    def _end_node_event(
        self,
        node_event: EventModel,
        next_node: str | None,
        current_node: str,
        messages: list[Message],
    ) -> None:
        """Check the node's goto target and publish the node END event."""
        # Fail with a clear error instead of a KeyError on an unknown target.
        validate_goto(next_node, self.nodes, current_node)
        logger.debug(
            "Node result processed, next_node=%s, total_messages=%d",
            next_node,
            len(messages),
        )
        node_event.event_type = EventType.END
        node_event.data["messages"] = messages
        node_event.content_type = [ContentType.MESSAGE]
        publish_event(node_event)

    async def _publish_node_update(
        self,
        has_hooks: bool,
        lifecycle_context: GraphLifecycleContext,
        current_node: str,
        old_state_snapshot: AgentState,
        state: Any,
        config: dict[str, Any],
        event: EventModel,
        step: int,
        messages: list[Message],
        stringify: bool,
    ) -> Any:
        """Run ``on_state_update`` after the node's result is merged, then publish UPDATE."""
        if has_hooks:
            state = await self.callback_mgr.fire_on_state_update(
                lifecycle_context,
                node_name=current_node,
                old_state=old_state_snapshot,
                new_state=state,
                step=step,
            )
        publish_state_update(config, current_node, step)

        await call_realtime_sync(state, config)
        event.event_type = EventType.UPDATE
        event.data["state"] = state
        event.data["messages"] = messages
        mirror_last_message(event, messages, stringify)
        event.content_type = [ContentType.STATE, ContentType.MESSAGE]
        publish_event(event)
        return state

    async def _interrupt_after(
        self,
        current_node: str,
        next_node: str | None,
        state: AgentState,
        config: dict[str, Any],
        event: EventModel,
        messages: list[Message],
        last_save: LastSave,
    ) -> tuple[bool, str | None]:
        """Pause after ``current_node`` when it is in ``interrupt_after``.

        The state is parked at the node that follows, which is also returned.
        """
        if not await check_and_handle_interrupt(
            current_node,
            "after",
            state,
            config,
            interrupt_before=self.interrupt_before,
            interrupt_after=self.interrupt_after,
            messages=messages,
            last_save=last_save,
        ):
            return False, next_node
        logger.info("Graph execution interrupted after node '%s'", current_node)
        if next_node is None:
            next_node = get_next_node(current_node, state, self.edges)
        state.set_current_node(next_node)

        event.event_type = EventType.INTERRUPTED
        event.data["interrupted"] = "After"
        event.metadata["interrupted"] = "After"
        event.data["state"] = state
        publish_event(event)
        return True, next_node

    async def _advance(
        self,
        current_node: str,
        next_node: str | None,
        state: AgentState,
        config: dict[str, Any],
        event: EventModel,
        step: int,
        messages: list[Message],
        persisted_upto: int,
        last_save: LastSave,
    ) -> tuple[str, int, int]:
        """Move to the next node, count the step and save it.

        Returns the next node, the new step and how many of ``messages`` are saved.
        """
        if next_node is None:
            current_node = get_next_node(current_node, state, self.edges)
            logger.debug("Next node determined by graph logic: '%s'", current_node)
        else:
            current_node = next_node
            logger.debug("Next node determined by command: '%s'", current_node)

        # The node finished, so answers to its interrupt() calls are spent.
        clear_resume_values(state)
        step += 1
        state.advance_step()

        # Durable per-step checkpoint, so a pod killed mid-run replays at most one node
        # instead of resuming from the last terminal save (or, once the cache is gone,
        # from the beginning). Only messages not saved yet are written, so a long run
        # does not re-upsert its whole history on every step.
        if config.get("durable_checkpoint_every_step", True):
            await sync_data(
                state=state,
                config=config,
                messages=messages[persisted_upto:],
                trim=False,
                last_save=last_save,
            )
            persisted_upto = len(messages)
        else:
            await call_realtime_sync(state, config)

        event.event_type = EventType.UPDATE
        event.metadata["State_Updated"] = "State Updated"
        event.data["state"] = state
        publish_event(event)
        return current_node, step, persisted_upto

    async def _recursion_limit_hit(
        self,
        state: AgentState,
        config: dict[str, Any],
        event: EventModel,
        step: int,
        max_steps: int,
        current_node: str,
    ) -> GraphRecursionError:
        """Park the state for the next turn, publish the error and return it to raise."""
        logger.error(self.RECURSION_MESSAGE)
        state.error(self.RECURSION_MESSAGE)
        # The limit is spent for this turn. Park the state at the entry so the
        # next message starts a new turn instead of stopping at once.
        state.set_current_node(START)
        state.execution_meta.step = 0
        await call_realtime_sync(state, config)
        event.event_type = EventType.ERROR
        event.data["state"] = state
        event.metadata["error"] = self.RECURSION_MESSAGE
        event.metadata["step"] = step
        event.metadata["current_node"] = current_node
        publish_event(event)
        return GraphRecursionError(
            message=f"Graph execution exceeded recursion limit: {max_steps}",
            error_code="RECURSION_001",
            context={
                "max_steps": max_steps,
                "current_step": step,
                "current_node": current_node,
            },
        )

    async def _complete_run(
        self,
        state: Any,
        config: dict[str, Any],
        event: EventModel,
        lifecycle_context: GraphLifecycleContext,
        messages: list[Message],
        step: int,
        current_node: str,
        last_save: LastSave,
        stringify: bool,
    ) -> tuple[Any, Any]:
        """Mark the run complete, run ``on_graph_end``, save and publish END.

        Returns the final state and whether the context was trimmed.
        """
        logger.info(
            "Graph execution completed successfully at node '%s' after %d steps",
            current_node,
            step,
        )
        state.complete()

        # Fire on_graph_end hook before final state sync
        if self._has_hooks():
            state = await self.callback_mgr.fire_on_graph_end(
                lifecycle_context,
                final_state=state,
                messages=messages,
                total_steps=step,
            )
        publish_graph_end(config, step)

        is_context_trimmed = await sync_data(
            state=state,
            config=config,
            messages=messages,
            trim=True,
            last_save=last_save,
        )
        event.event_type = EventType.END
        event.data["state"] = state
        event.data["messages"] = messages
        mirror_last_message(event, messages, stringify)
        event.content_type = [ContentType.STATE, ContentType.MESSAGE]
        event.metadata["status"] = "Graph execution completed"
        event.metadata["step"] = step
        event.metadata["current_node"] = current_node
        event.metadata["is_context_trimmed"] = is_context_trimmed
        publish_event(event)
        return state, is_context_trimmed

    async def _fail_run(
        self,
        error: Exception,
        state: Any,
        config: dict[str, Any],
        event: EventModel,
        lifecycle_context: GraphLifecycleContext,
        messages: list[Message],
        step: int,
        current_node: str,
        last_save: LastSave,
    ) -> Any:
        """Record a failed run: error state, ERROR event, ``on_graph_error``, final save.

        Called from the loop's ``except`` block; the caller re-raises. Returns the state
        the hooks left.
        """
        logger.exception("Graph execution failed: %s", error)
        state.error(client_error_text(error))
        # Close tool calls the failed node never answered, so the saved history is valid.
        await close_unanswered_tool_calls(state, config, messages)

        event.event_type = EventType.ERROR
        event.metadata["error"] = str(error)
        event.data["state"] = state
        publish_event(event)

        # Fire on_graph_error hook before persisting the error state
        error_message = str(error)
        if self._has_hooks():
            state, error_message = await self.callback_mgr.fire_on_graph_error(
                lifecycle_context,
                error=error,
                partial_state=state,
                messages=messages,
                step=step,
                node_name=current_node,
            )
        publish_graph_error(config, error_message, step, current_node)

        await sync_data(
            state=state,
            config=config,
            messages=messages,
            trim=True,
            last_save=last_save,
        )
        return state
