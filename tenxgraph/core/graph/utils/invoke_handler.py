from __future__ import annotations  # isort: skip_file

import asyncio
import logging
from functools import partial
from typing import Any, TypeVar

from collections.abc import Callable

from injectq import inject, Inject

from tenxgraph.core.authz import public_config
from tenxgraph.core.exceptions import GraphStopRequested
from tenxgraph.core.graph.edge import Edge
from tenxgraph.core.graph.node import Node
from tenxgraph.core.graph.utils.guards import execute_with_guards, node_guards, node_metrics
from tenxgraph.core.graph.utils.utils import (
    LastSave,
    calculate_token_usage,
    parse_response,
    process_node_result,
)
from tenxgraph.storage.checkpointer import BaseCheckpointer
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import END, ResponseGranularity
from tenxgraph.utils.command import Command
from tenxgraph.utils.logging import log_context_scope
from tenxgraph.core.state.reducers import add_messages
from tenxgraph.utils.callbacks import CallbackManager, GraphLifecycleContext

from .handler_utils import (
    check_stop_requested,
    finalize_cancelled_run,
    interrupt_graph,
    node_after,
    split_remote_calls,
)

from .handler_mixins import (
    BaseLoggingMixin,
    GraphRunMixin,
    InterruptConfigMixin,
)
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.interrupt import GraphInterrupt, node_scope, resume_values


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger("tenxgraph.graph")


class InvokeHandler[StateT: AgentState](
    BaseLoggingMixin,
    InterruptConfigMixin,
    GraphRunMixin,
):
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
        # Keep existing attributes for backward-compatibility
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
        # And set via mixin for a single source of truth
        self._set_interrupts(interrupt_before, interrupt_after)
        self.callback_mgr = callback_mgr
        self.node_aliases: dict[str, str] = dict(node_aliases or {})

    async def _execute_node_guarded(
        self,
        node: Node,
        node_name: str,
        config: dict[str, Any],
        state: StateT,
        checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],
    ):
        """Execute a node under a deadline, cancellable by an out-of-band stop.

        Bare `await node.execute(...)` had no bound and no way in: a node hung on
        a socket blocked the run forever, and because control never returned to
        the loop, the between-nodes stop check never ran. Here the node runs as a
        task we can actually cancel -- on the deadline, or as soon as a stop
        request shows up.
        """
        guards = node_guards(node_name, config, fresh(checkpointer))
        with node_metrics(node_name):
            return await execute_with_guards(
                node.execute(config, state),  # type: ignore[arg-type]
                **guards,
            )

    async def _execute_graph(  # noqa: PLR0911, PLR0912, PLR0915
        self,
        state: StateT,
        config: dict[str, Any],
        input_data: dict[str, Any] | None = None,
    ) -> tuple[StateT, list[Message]]:
        """Execute the entire graph with support for interrupts and resuming."""
        logger.info(
            "Starting graph execution from node '%s' at step %d",
            state.execution_meta.current_node,
            state.execution_meta.step,
        )
        logger.debug("DEBUG: Current node value: %r", state.execution_meta.current_node)
        logger.debug("DEBUG: END constant value: %r", END)
        logger.debug("DEBUG: Are they equal? %s", state.execution_meta.current_node == END)
        messages: list[Message] = []
        # How many of `messages` have already been durably persisted by a per-step
        # checkpoint, so terminal and subsequent writes only send the new ones.
        persisted_upto = 0
        # The run's latest durable save, for the cancel cleanup (never two saves at once).
        last_save = LastSave()
        max_steps = config.get("recursion_limit", 25)
        logger.debug("Max steps limit set to %d", max_steps)

        # The user's message belongs to the run's messages only when this call brought it.
        # A resume with no new input must not send the last one again.
        last_human_message = None
        if input_data is None or input_data.get("messages"):
            last_human_message = state.context[-1] if state.context else None
            if last_human_message and last_human_message.role != "user":
                msg = [msg for msg in reversed(state.context) if msg.role == "user"]
                last_human_message = msg[0] if msg else None

        if last_human_message:
            logger.debug("Last human message: %s", last_human_message.content)
            messages.append(last_human_message)

        current_node, step = self._start_node(state, config)

        # Build lifecycle context (shared across all hooks in this execution)
        lifecycle_context = GraphLifecycleContext(config=public_config(config))

        # Create event for graph execution
        event = EventModel.default(
            config,
            data={"state": state},
            event=Event.GRAPH_EXECUTION,
            content_type=[ContentType.STATE],
            node_name=current_node,
            extra={
                "current_node": current_node,
                "step": step,
                "max_steps": max_steps,
            },
        )

        try:
            while current_node != END and step < max_steps:
                logger.debug("Executing step %d at node '%s'", step, current_node)
                # Reload state in each iteration to get latest (in case of external updates)
                res = await check_stop_requested(
                    state,
                    current_node,
                    event,
                    messages,
                    config,
                    last_save=last_save,
                )
                if res:
                    return state, messages

                await self._begin_step(state, config, event, step, current_node)
                publish_event(event)

                if await self._interrupt_before(
                    current_node, state, config, event, messages[persisted_upto:], last_save
                ):
                    return state, messages

                # Execute current node - use factory for override support
                logger.debug("Executing node '%s'", current_node)
                node = self._get_node(current_node)
                has_hooks, old_state_snapshot, node_event = self._start_node_event(
                    state, config, step, current_node
                )

                try:
                    with node_scope(config, current_node, resume_values(state)):
                        result = await self._execute_node_guarded(node, current_node, config, state)
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
                    return state, messages
                except GraphStopRequested:
                    await self._stop_mid_node(
                        current_node, state, config, event, messages, last_save
                    )
                    return state, messages

                logger.debug("Node '%s' execution completed", current_node)

                next_node = None

                # Tool calls the client runs: merge the rest of the result as usual, then pause.
                remote_calls, result = split_remote_calls(result)

                # Process result and get next node
                if isinstance(result, list):
                    # If result is a list of Message, append to messages
                    messages.extend(result)
                    logger.debug(
                        "Node '%s' returned %d messages, total messages now %d",
                        current_node,
                        len(result),
                        len(messages),
                    )
                    # Add messages to state context so they're visible to subsequent nodes
                    state.context = add_messages(state.context, result)

                # No state change beyond adding messages, just advance to next node
                if isinstance(result, dict):
                    state = result.get("state", state)
                    next_node = result.get("next_node")
                    new_messages = result.get("messages", [])
                    if new_messages:
                        # Make them visible to the next node, as the list branch does.
                        state.context = add_messages(state.context, new_messages)
                        messages.extend(new_messages)
                        logger.debug(
                            "Node '%s' returned %d messages, total messages now %d",
                            current_node,
                            len(new_messages),
                            len(messages),
                        )

                # A Command (for example a handoff from a ToolNode) carries both a state
                # update and the next node. Follow it the same way the stream path does.
                if isinstance(result, Command):
                    command_messages: list[Message] = []
                    state, command_messages, next_node = await process_node_result(
                        result, state, command_messages
                    )
                    messages.extend(command_messages)

                self._end_node_event(node_event, next_node, current_node, messages)

                if remote_calls:
                    # The run resumes after this node once the client sends the results. The
                    # placeholders are returned (clients read them) but kept out of the context.
                    await interrupt_graph(
                        current_node,
                        state,
                        config,
                        messages=messages[persisted_upto:],
                        last_save=last_save,
                    )
                    messages.extend(remote_calls)
                    return state, messages

                # Check stop again after node execution. The node finished, so the stop is
                # saved at the node that follows it.
                res = await check_stop_requested(
                    state,
                    current_node,
                    event,
                    messages,
                    config,
                    resume_after=partial(node_after, current_node, next_node, state, self.edges),
                    last_save=last_save,
                )
                if res:
                    return state, messages

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
                    stringify=False,
                )

                is_interrupted_requested, next_node = await self._interrupt_after(
                    current_node,
                    next_node,
                    state,
                    config,
                    event,
                    messages[persisted_upto:],
                    last_save,
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

                # If we interrupted after, exit now
                if is_interrupted_requested:
                    return state, messages

                if step >= max_steps and current_node != END:
                    raise await self._recursion_limit_hit(
                        state, config, event, step, max_steps, current_node
                    )

            state, _ = await self._complete_run(
                state,
                config,
                event,
                lifecycle_context,
                messages,
                step,
                current_node,
                last_save,
                stringify=False,
            )
            return state, messages

        except asyncio.CancelledError:
            # Cancelled mid-run (client gone, shutdown, timeout around the run): save the
            # thread as INTERRUPTED so the next run can use it, then let the cancel go on.
            await finalize_cancelled_run(
                state, config, messages, event, current_node, last_save=last_save
            )
            raise
        except Exception as e:
            await self._fail_run(
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
            raise

    async def invoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        default_state: StateT,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ):
        """Execute the graph asynchronously with event publishing."""
        # Bind run correlation for every log line emitted underneath this run, so a
        # single run can actually be grepped out of a busy multi-tenant server. The ids
        # are put back when the run ends.
        with log_context_scope(config):
            return await self._invoke(input_data, config, default_state, response_granularity)

    async def _invoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        default_state: StateT,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ):
        logger.info(
            "Starting asynchronous graph execution with %d input keys, granularity=%s",
            len(input_data) if input_data else 0,
            response_granularity,
        )
        input_data = input_data or {}
        state: StateT = await self._load_state(input_data, config, default_state)

        # Event publishing logic
        event = EventModel.default(
            config,
            data={"state": state},
            event=Event.GRAPH_EXECUTION,
            content_type=[ContentType.STATE],
            node_name=state.execution_meta.current_node,
            extra={
                "current_node": state.execution_meta.current_node,
                "step": state.execution_meta.step,
            },
        )
        event.event_type = EventType.START
        publish_event(event)

        # Apply a resume, then fire on_graph_start before execution begins
        state, config = await self._start_run(state, input_data, config)

        event.event_type = EventType.UPDATE
        event.metadata["status"] = "Graph invoked"
        publish_event(event)

        try:
            logger.debug("Beginning graph execution")
            event.event_type = EventType.PROGRESS
            event.metadata["status"] = "Graph execution started"
            publish_event(event)

            final_state, messages = await self._execute_graph(state, config, input_data)
            logger.info("Graph execution completed with %d final messages", len(messages))

            # Calculate token usage
            token_usage = calculate_token_usage(messages)

            event.event_type = EventType.END
            event.metadata["status"] = "Graph execution completed"
            event.metadata.update(token_usage)
            event.data["state"] = final_state
            event.data["messages"] = messages
            publish_event(event)

            return await parse_response(
                final_state,
                messages,
                response_granularity,
                token_usage=token_usage,
            )
        except Exception as e:
            logger.exception("Graph execution failed: %s", e)
            event.event_type = EventType.ERROR
            event.metadata["status"] = f"Graph execution failed: {e}"
            event.data["error"] = str(e)
            publish_event(event)
            raise
