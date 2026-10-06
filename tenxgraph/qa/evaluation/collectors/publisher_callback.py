"""
PublisherCallback — wraps a BasePublisher as an AfterInvokeCallback.

Converts graph callback invocations into EventModel events and publishes
them to a BasePublisher (typically TrajectoryCollector).  Registered for
TOOL, MCP, and AI invocation types so all tool calls and node visits are
captured during graph execution.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agentflow.runtime.publisher.base_publisher import BasePublisher
from agentflow.runtime.publisher.events import ContentType, Event, EventModel, EventType
from agentflow.utils.callbacks import AfterInvokeCallback, CallbackContext, InvocationType


logger = logging.getLogger("agentflow.evaluation.collectors")


class PublisherCallback(AfterInvokeCallback):
    """Wraps a BasePublisher as an AfterInvokeCallback for graph execution.

    Builds an EventModel from each callback invocation and calls
    publisher.publish() directly (awaited), avoiding the race condition
    of the internal publish_event() which fires as a background task.

    Example:
        ```python
        collector = TrajectoryCollector()
        cb = PublisherCallback(collector, config={"thread_id": "run-1"})

        mgr = CallbackManager()
        mgr.register_after_invoke(InvocationType.AI, cb)
        mgr.register_after_invoke(InvocationType.TOOL, cb)
        ```
    """

    def __init__(self, publisher: BasePublisher, config: dict | None = None):
        self._publisher = publisher
        self._config = config or {}

    async def __call__(self, context: CallbackContext, input_data: Any, output_data: Any) -> Any:
        node_message = None
        if context.invocation_type == InvocationType.AI:
            node_message = await self._extract_node_message(output_data)
        event = self._build_event(context, input_data, output_data, node_message=node_message)
        if event:
            await self._publisher.publish(event)
        return output_data

    async def _extract_node_message(self, output_data: Any) -> Any:
        """Safely extract a Message from a ModelResponseConverter or result dict.

        Handles two pathways:
        1. ModelResponseConverter — returned by normal node functions.
        2. dict with a "messages" key — returned by _call_agent_node().

        Returns None on any failure so the callback never raises.
        """
        try:
            from agentflow.runtime.adapters.llm.model_response_converter import (
                ModelResponseConverter,
            )

            if isinstance(output_data, ModelResponseConverter):
                return await output_data.invoke()
        except Exception:
            logger.debug("ModelResponseConverter extraction failed", exc_info=True)

        try:
            if isinstance(output_data, dict):
                messages = output_data.get("messages")
                if messages and len(messages) > 0:
                    return messages[-1]
        except Exception:
            logger.debug("Agent-node message extraction failed", exc_info=True)

        return None

    def _build_event(
        self,
        context: CallbackContext,
        input_data: Any,
        output_data: Any,
        node_message: Any = None,
    ) -> EventModel | None:
        """Build an EventModel from callback arguments, or None for unhandled types."""
        if context.invocation_type in (InvocationType.TOOL, InvocationType.MCP):
            return EventModel(
                event=Event.TOOL_EXECUTION,
                event_type=EventType.END,
                node_name=context.function_name or context.node_name or "",
                data={
                    "function_name": context.function_name,
                    "args": input_data if isinstance(input_data, dict) else {},
                    "result": str(output_data) if output_data is not None else "",
                    "tool_call_id": (context.metadata or {}).get("tool_call_id", ""),
                },
                content_type=[ContentType.TOOL_RESULT],
                thread_id=self._config.get("thread_id", ""),
                run_id=self._config.get("run_id", ""),
                timestamp=time.time(),
            )

        if context.invocation_type == InvocationType.AI:
            state = (
                input_data.get("state", input_data) if isinstance(input_data, dict) else input_data
            )
            input_messages: list[dict] = []
            if hasattr(state, "context") and state.context:
                input_messages = [
                    {"role": m.role, "content": m.text() or ""}
                    for m in state.context
                    if hasattr(m, "role")
                ]

            response_text = ""
            has_tool_calls = False
            tool_call_names: list[str] = []
            if node_message is not None:
                response_text = node_message.text() or ""
                tc = getattr(node_message, "tools_calls", None) or []
                has_tool_calls = bool(tc)
                for t in tc:
                    if isinstance(t, dict):
                        tool_call_names.append(t.get("name", ""))
                    elif hasattr(t, "name"):
                        tool_call_names.append(t.name)
                    elif hasattr(t, "model_dump"):
                        tool_call_names.append(t.model_dump().get("name", ""))

            token_data: dict[str, int] = {}
            if node_message is not None:
                usages = getattr(node_message, "usages", None)
                if usages is not None:
                    token_data = {
                        "input_tokens": getattr(usages, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(usages, "completion_tokens", 0) or 0,
                        "cache_read_tokens": getattr(usages, "cache_read_input_tokens", 0) or 0,
                        "cache_creation_tokens": getattr(usages, "cache_creation_input_tokens", 0)
                        or 0,
                    }

            return EventModel(
                event=Event.NODE_EXECUTION,
                event_type=EventType.END,
                node_name=context.node_name or "",
                data={
                    "input_messages": input_messages,
                    "response_text": response_text,
                    "has_tool_calls": has_tool_calls,
                    "tool_call_names": tool_call_names,
                    "is_final": not has_tool_calls,
                    "token_usage": token_data,
                },
                content_type=[ContentType.MESSAGE],
                thread_id=self._config.get("thread_id", ""),
                run_id=self._config.get("run_id", ""),
                timestamp=time.time(),
            )

        return None
