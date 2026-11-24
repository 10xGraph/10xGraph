"""The steps of the graph loop (``GraphRunMixin``) and ``mirror_last_message``."""

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


def mirror_last_message(event: EventModel, messages: list[Message]) -> None:
    """Copy the last message's text (as ``str``) and blocks onto ``event``."""
    if messages:
        last = messages[-1]
        content = last.text() if isinstance(last.content, list) else last.content
        event.content = str(content)
        if isinstance(last.content, list):
            event.content_blocks = last.content


class GraphRunMixin:
    """Steps of the graph loop (``GraphEngine``), one method per step.

    The host class provides the attributes below.
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
        """Record the node and step about to run and publish PROGRESS."""
        state.set_current_node(current_node)
        state.execution_meta.step = step
        await call_realtime_sync(state, config)
        event.data["state"] = state
        event.data["step"] = step
        event.data["current_node"] = current_node
        event.metadata["step"] = step
        event.metadata["current_node"] = current_node
        event.metadata["status"] = f"Executing step {step} at node '{current_node}'"
        event.event_type = EventType.PROGRESS
        publish_event(event)

    async def _interrupt_before(
        self,
        current_node: str,
        state: AgentState,
        config: dict[str, Any],
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

    async def _stop_mid_node(
        self,
        current_node: str,
        state: AgentState,
        config: dict[str, Any],
        messages: list[Message],
        last_save: LastSave,
    ) -> None:
        """The node was cancelled because a stop arrived: finish as a between-nodes stop.

        This is control flow, not a failure: the run is marked stopped, saved and
        reported exactly as a stop seen between nodes.
        """
        logger.info("Node '%s' cancelled by stop request; finalizing run", current_node)
        await check_stop_requested(state, current_node, messages, config, last_save=last_save)

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
        mirror_last_message(event, messages)
        event.content_type = [ContentType.STATE, ContentType.MESSAGE]
        publish_event(event)
        return state

    async def _interrupt_after(
        self,
        current_node: str,
        next_node: str | None,
        state: AgentState,
        config: dict[str, Any],
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
        mirror_last_message(event, messages)
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
