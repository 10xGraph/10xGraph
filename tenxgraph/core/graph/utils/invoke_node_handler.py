"""
InvokeNodeHandler utilities for TAF agent graph execution.

This module provides the InvokeNodeHandler class, which manages the invocation of node functions,
tool nodes, and agent instances within the agent graph. It supports dependency injection,
callback hooks, event publishing, and error recovery for regular, tool-based, and agent nodes.

Classes:
    InvokeNodeHandler: Handles execution of node functions, tool nodes, and agents with DI
        and callbacks.

Usage:
    handler = InvokeNodeHandler(name, func, publisher)
    result = await handler.invoke(config, state)
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

from injectq import Inject

from tenxgraph.core.exceptions import NodeError
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.graph.utils.utils import process_node_result
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.publisher.events import EventType
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.utils import (
    CallbackManager,
    call_sync_or_async,
)
from tenxgraph.utils.command import Command
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.interrupt import activate as activate_interrupts

from .handler_mixins import BaseLoggingMixin, mirror_last_message
from .tool_runner import ToolRunner, ToolRunResult
from .tool_runtime import ToolRuntimeMixin


if TYPE_CHECKING:
    from tenxgraph.core.graph.base_agent import BaseAgent


logger = logging.getLogger("tenxgraph.graph")


class InvokeNodeHandler(BaseLoggingMixin, ToolRuntimeMixin):
    """
    Handles invocation of node functions, tool nodes, and agent instances in the agent graph.

    Supports dependency injection, callback hooks, event publishing, and error recovery.

    Args:
        name (str): Name of the node.
        func (Callable | ToolNode | Agent): The function, ToolNode, or Agent to execute.
    """

    def __init__(
        self,
        name: str,
        func: Union[Callable, "ToolNode", "BaseAgent"],
    ):
        self.name = name
        self.func = func

    async def _invoke_tool_node(
        self,
        state: AgentState,
        config: dict[str, Any],
    ) -> list[Message] | dict[str, Any] | Command:
        """Run a ToolNode step through the tool runner, in the invoke result shapes."""
        result = ToolRunResult()
        async for item in ToolRunner(self.name, self.func).run(state, config, streaming=False):
            result = item  # type: ignore[assignment]  # only the result when not streaming
        if result.next_node is not None:
            return Command(update=result.messages, goto=result.next_node)  # type: ignore[arg-type]
        if result.state_updated:
            return {"state": state, "messages": result.messages, "next_node": None}
        return result.messages

    async def _invoke_by_node_type(
        self,
        state: AgentState,
        config: dict[str, Any],
        callback_mgr: CallbackManager,
    ) -> dict[str, Any] | list[Message] | Command:
        """Dispatch node execution to the appropriate implementation."""
        from tenxgraph.core.graph.agent import Agent
        from tenxgraph.core.graph.base_agent import BaseAgent

        if isinstance(self.func, Agent | BaseAgent):
            logger.debug("Node '%s' is an Agent instance, executing agent", self.name)
            return await self._call_agent_node(
                state,
                config,
                callback_mgr,
            )

        if isinstance(self.func, ToolNode):
            return await self._invoke_tool_node(state, config)  # type: ignore

        return await self._call_normal_node(
            state,
            config,
            callback_mgr,
        )

    async def _call_normal_node(
        self,
        state: "AgentState",
        config: dict[str, Any],
        callback_mgr: CallbackManager,
    ) -> dict[str, Any]:
        """
        Execute a regular node function with callback hooks and event publishing.

        Args:
            state (AgentState): Current agent state.
            config (dict): Node configuration.
            callback_mgr (CallbackManager): Callback manager for hooks.

        Returns:
            dict: Result containing new state, messages, and next node.

        Raises:
            Exception: If function execution fails and cannot be recovered.
        """
        logger.debug("Node '%s' calling normal function", self.name)
        result: dict[str, Any] = {}

        logger.debug("Node '%s' is a regular function, executing with callbacks", self.name)
        # This is a regular function - likely AI function
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
            event.metadata["status"] = "Function execution started"
            publish_event(event)

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

            # Process result and publish END event
            messages = []
            new_state, messages, next_node = await process_node_result(
                result, state, messages, seen_message_ids
            )
            self._stamp_node_name(messages, roles=("assistant",))
            event.data["state"] = new_state
            event.event_type = EventType.END
            event.metadata["status"] = "Function execution completed"
            event.data["messages"] = messages
            event.data["next_node"] = next_node
            # mirror simple content + structured blocks for the last message
            mirror_last_message(event, messages, stringify=True)
            publish_event(event)

            return {
                "state": new_state,
                "messages": messages,
                "next_node": next_node,
            }

        except Exception as e:
            logger.warning(
                "Node '%s' execution failed, executing error callbacks: %s", self.name, e
            )
            # Execute error callbacks
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if recovery_result is not None:
                # Use recovery result instead of raising the error
                event.metadata["status"] = "Function execution recovered from error"
                self._publish_node_recovered(event, recovery_result)
                return {
                    "state": state,
                    "messages": [recovery_result],
                    "next_node": None,
                }
            # Re-raise the original error
            event.metadata["status"] = f"Function execution failed: {e}"
            self._publish_node_failed(event, e, config)
            raise

    async def _call_agent_node(
        self,
        state: "AgentState",
        config: dict[str, Any],
        callback_mgr: CallbackManager,
    ) -> dict[str, Any]:
        """Execute an Agent instance node with callback hooks and event publishing.

        Args:
            state (AgentState): Current agent state.
            config (dict): Node configuration.
            callback_mgr (CallbackManager): Callback manager for hooks.

        Returns:
            dict: Result containing new state, messages, and next node.
        """
        logger.debug("Node '%s' is an Agent instance, executing agent logic", self.name)

        agent = self.func  # type: ignore - func is Agent instance here

        context = self._node_callback_context(
            config, getattr(agent, "__class__", type(agent)).__name__
        )
        input_data = {"state": state, "config": config}
        event = self._publish_node_start(config, state, type(agent).__name__)

        try:
            # Execute before_invoke callbacks
            input_data = await callback_mgr.execute_before_invoke(context, input_data)

            event.event_type = EventType.PROGRESS
            event.metadata["status"] = "Agent execution started"
            publish_event(event)

            # Execute the agent's logic
            converter = await agent.execute(state, config)  # type: ignore

            # Process the converter result (invoke for non-streaming)
            message = await converter.invoke()
            self._stamp_node_name(message, roles=("assistant",))

            # Update state with new message
            new_state = state.model_copy(deep=True)
            new_state.context.append(message)

            result = {
                "state": new_state,
                "messages": [message],
                "next_node": None,
            }

            # Execute after_invoke callbacks
            result = await callback_mgr.execute_after_invoke(context, input_data, result)

            # Publish END event
            messages = result.get("messages", [])
            event.data["state"] = result["state"]
            event.event_type = EventType.END
            event.metadata["status"] = "Agent execution completed"
            event.data["messages"] = messages
            event.data["next_node"] = result.get("next_node")
            mirror_last_message(event, messages, stringify=False)
            publish_event(event)

            return result

        except Exception as e:
            logger.warning(
                "Node '%s' agent execution failed, executing error callbacks: %s",
                self.name,
                e,
            )
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if recovery_result is not None:
                event.metadata["status"] = "Agent execution recovered from error"
                self._publish_node_recovered(event, recovery_result)
                return {
                    "state": state,
                    "messages": [recovery_result],
                    "next_node": None,
                }
            event.metadata["status"] = f"Agent execution failed: {e}"
            self._publish_node_failed(event, e, config)
            raise

    async def invoke(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ) -> dict[str, Any] | list[Message] | Command:
        """
        Execute the node function or ToolNode with dependency injection and callback hooks.

        Args:
            config (dict): Node configuration.
            state (AgentState): Current agent state.
            callback_mgr (CallbackManager, optional): Callback manager for hooks.

        Returns:
            dict | list[Message] | Command: Result of node execution (regular node,
            tool node, or a Command for handoff).

        Raises:
            NodeError: If execution fails or context is missing for tool nodes.
        """
        callback_mgr = fresh(callback_mgr)
        logger.info("Executing node '%s'", self.name)
        logger.debug(
            "Node '%s' execution with state context size=%d, config keys=%s",
            self.name,
            len(state.context) if state.context else 0,
            list(config.keys()) if config else [],
        )

        try:
            result = await self._invoke_by_node_type(
                state,
                config,
                callback_mgr,
            )

            logger.info("Node '%s' execution completed successfully", self.name)
            # we are flattening the result here because we want to return
            # a consistent format for both tool and normal nodes
            return result
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
