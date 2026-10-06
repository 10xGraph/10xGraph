"""
TrajectoryCollector — accumulates execution trajectory from EventModel events.

Wires into graph.compile() via a CallbackManager rather than requiring
changes at construction time.

Example:
    collector = TrajectoryCollector()
    _, callback_mgr = make_trajectory_callback(collector, config={"thread_id": "run-1"})

    compiled = graph.compile(callback_manager=callback_mgr)
    await compiled.ainvoke(state, config)

    print(collector.tool_calls)
    print(collector.node_visits)
    print(f"duration: {collector.duration:.3f}s")
"""

import logging
import time
from dataclasses import dataclass
from dataclasses import field as datafield
from typing import Any

from agentflow.qa.evaluation.collectors.publisher_callback import PublisherCallback
from agentflow.qa.evaluation.dataset.eval_set import StepType, ToolCall, TrajectoryStep
from agentflow.qa.evaluation.token_usage import TokenUsage
from agentflow.runtime.publisher.base_publisher import BasePublisher
from agentflow.runtime.publisher.events import Event, EventModel, EventType
from agentflow.utils.callbacks import CallbackManager, InvocationType


logger = logging.getLogger("agentflow.evaluation.collectors")


@dataclass
class NodeResponse:
    """Snapshot of input and output at a single AI node invocation.

    input_messages is extracted from input_data["state"].context BEFORE
    process_node_result updates the state.  response_text is extracted via
    output_data.invoke() because the LLM response is NOT yet written to state
    at callback time.

    Attributes:
        node_name:       Name of the graph node that ran (e.g. "MAIN").
        input_messages:  Conversation history going into this node.
        response_text:   LLM text output; empty when this is a tool-call turn.
        has_tool_calls:  True when the LLM decided to call a tool this turn.
        tool_call_names: Names of tools requested.
        is_final:        True when no further tool calls — last text turn.
        timestamp:       Wall-clock time when this node invocation completed.
    """

    node_name: str
    input_messages: list[dict] = datafield(default_factory=list)
    response_text: str = ""
    has_tool_calls: bool = False
    tool_call_names: list[str] = datafield(default_factory=list)
    is_final: bool = False
    timestamp: float = 0.0
    token_usage: TokenUsage = datafield(default_factory=TokenUsage)


class TrajectoryCollector(BasePublisher):
    """Collects execution trajectory from graph events.

    Captures the sequence of nodes visited, tools called, and LLM outputs
    produced during agent graph execution.  Extends BasePublisher so it can
    be used as the target for PublisherCallback.

    Attributes:
        trajectory:      Complete execution trajectory as TrajectoryStep objects.
        tool_calls:      List of tool calls made during execution.
        node_visits:     List of node names visited in order.
        node_responses:  Per-node input/output snapshots (one per node visit).
        final_response:  LLM text from the last non-tool-call node invocation.
        events:          All raw EventModel objects (when capture_all_events=True).
        start_time:      When the first event was received.
        end_time:        When the last event was received.

    Example:
        ```python
        collector = TrajectoryCollector()
        _, mgr = make_trajectory_callback(collector, config={"thread_id": "run-1"})
        compiled = graph.compile(callback_manager=mgr)
        await compiled.ainvoke(state, config)

        print(f"Visited nodes: {collector.node_visits}")
        print(f"Tool calls: {collector.tool_calls}")
        print(f"Duration: {collector.duration:.3f}s")
        ```
    """

    def __init__(self, capture_all_events: bool = False):
        super().__init__(config={})
        self.trajectory: list[TrajectoryStep] = []
        self.tool_calls: list[ToolCall] = []
        self.node_visits: list[str] = []
        self.node_responses: list[NodeResponse] = []
        self.final_response: str = ""
        self.events: list[EventModel] = []
        self.capture_all_events = capture_all_events
        self.start_time: float | None = None
        self.end_time: float | None = None
        self._pending_tool_calls: dict[str, dict] = {}

    async def publish(self, event: EventModel) -> None:
        await self.on_event(event)

    async def on_event(self, event: EventModel) -> None:
        now = time.time()
        if self.start_time is None:
            self.start_time = now
        self.end_time = now

        if self.capture_all_events:
            self.events.append(event)

        if event.event == Event.NODE_EXECUTION:
            self._process_node_event(event)
        elif event.event == Event.TOOL_EXECUTION:
            self._process_tool_event(event)
        elif event.event == Event.GRAPH_EXECUTION:
            self._process_graph_event(event)

    def on_event_sync(self, event: EventModel) -> None:
        now = time.time()
        if self.start_time is None:
            self.start_time = now
        self.end_time = now

        if self.capture_all_events:
            self.events.append(event)

        if event.event == Event.NODE_EXECUTION:
            self._process_node_event(event)
        elif event.event == Event.TOOL_EXECUTION:
            self._process_tool_event(event)
        elif event.event == Event.GRAPH_EXECUTION:
            self._process_graph_event(event)

    def _process_node_event(self, event: EventModel) -> None:
        node_name = event.node_name or event.data.get("node_name", "")
        if not node_name:
            logger.debug("Dropping node event with empty node_name: %s", event.event_type)
            return

        self.node_visits.append(node_name)
        self.trajectory.append(
            TrajectoryStep.node(
                name=node_name,
                timestamp=event.timestamp,
                event_type=event.event_type.value
                if hasattr(event.event_type, "value")
                else str(event.event_type),
            )
        )
        response_text = event.data.get("response_text", "")
        has_tool_calls = event.data.get("has_tool_calls", False)
        is_final = event.data.get("is_final", not has_tool_calls)
        raw_token = event.data.get("token_usage", {})
        node_token_usage = TokenUsage(
            input_tokens=raw_token.get("input_tokens", 0),
            output_tokens=raw_token.get("output_tokens", 0),
            cache_read_tokens=raw_token.get("cache_read_tokens", 0),
            cache_creation_tokens=raw_token.get("cache_creation_tokens", 0),
        )
        nr = NodeResponse(
            node_name=node_name,
            input_messages=event.data.get("input_messages", []),
            response_text=response_text,
            has_tool_calls=has_tool_calls,
            tool_call_names=event.data.get("tool_call_names", []),
            is_final=is_final,
            timestamp=event.timestamp or time.time(),
            token_usage=node_token_usage,
        )
        self.node_responses.append(nr)
        if is_final and response_text:
            self.final_response = response_text

    def _process_tool_event(self, event: EventModel) -> None:
        """Process a tool execution event.

        Handles both END-only events (PublisherCallback pathway) and separate
        START/END events.  START events are parked in _pending_tool_calls;
        the ToolCall is committed once the END event arrives.
        """
        function_name = event.data.get("function_name", event.node_name or "")
        args = {
            k: v
            for k, v in (event.data.get("args", {})).items()
            if k not in ("tool_call_id", "state")
        }
        tool_call_id = event.data.get("tool_call_id", "")
        result = event.data.get("result", "")

        if not function_name:
            logger.debug("Dropping tool event with empty function_name")
            return

        is_start = getattr(event, "event_type", None) == EventType.START
        is_end = getattr(event, "event_type", None) == EventType.END

        if is_start and tool_call_id:
            self._pending_tool_calls[tool_call_id] = {"name": function_name, "args": args}
            return

        if is_end and tool_call_id and tool_call_id in self._pending_tool_calls:
            pending = self._pending_tool_calls.pop(tool_call_id)
            args = pending.get("args", {}) or args
            function_name = pending.get("name", "") or function_name

        tc = ToolCall(
            name=function_name,
            args=args,
            call_id=tool_call_id if tool_call_id else None,
            result=result,
        )
        self.tool_calls.append(tc)
        self.trajectory.append(
            TrajectoryStep.tool(
                name=function_name,
                args=args,
                timestamp=event.timestamp,
                tool_call_id=tool_call_id,
            )
        )

    def _process_graph_event(self, event: EventModel) -> None:
        if event.event_type == EventType.END:
            self.end_time = time.time()

    def get_tool_names(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]

    def get_trajectory_steps(self, step_type: StepType | None = None) -> list[TrajectoryStep]:
        if step_type is None:
            return self.trajectory
        return [s for s in self.trajectory if s.step_type == step_type]

    def reset(self) -> None:
        self.trajectory.clear()
        self.tool_calls.clear()
        self.node_visits.clear()
        self.node_responses.clear()
        self.final_response = ""
        self.events.clear()
        self._pending_tool_calls.clear()
        self.start_time = None
        self.end_time = None

    @property
    def duration(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory": [s.model_dump() for s in self.trajectory],
            "tool_calls": [tc.model_dump() for tc in self.tool_calls],
            "node_visits": self.node_visits,
            "node_responses": [
                {
                    "node_name": nr.node_name,
                    "input_messages": nr.input_messages,
                    "response_text": nr.response_text,
                    "has_tool_calls": nr.has_tool_calls,
                    "tool_call_names": nr.tool_call_names,
                    "is_final": nr.is_final,
                    "timestamp": nr.timestamp,
                }
                for nr in self.node_responses
            ],
            "final_response": self.final_response,
            "duration_seconds": self.duration,
        }

    async def close(self) -> None:
        self.end_time = time.time()

    def sync_close(self) -> None:
        self.end_time = time.time()

    def __repr__(self) -> str:
        return (
            f"TrajectoryCollector("
            f"nodes={len(self.node_visits)}, "
            f"tools={len(self.tool_calls)}, "
            f"responses={len(self.node_responses)}, "
            f"steps={len(self.trajectory)})"
        )


def make_trajectory_callback(
    collector: TrajectoryCollector,
    config: dict | None = None,
) -> tuple[TrajectoryCollector, CallbackManager]:
    """Wire a TrajectoryCollector into a CallbackManager.

    Registers hooks for TOOL, MCP, and AI invocation types so all tool calls
    and node visits are captured automatically during graph execution.

    Args:
        collector: The TrajectoryCollector that will receive events.
        config:    Optional dict with thread_id / run_id to stamp on each event.

    Returns:
        Tuple of (collector, callback_manager). Pass the callback_manager
        to graph.compile().

    Example:
        ```python
        collector = TrajectoryCollector()
        _, mgr = make_trajectory_callback(collector, config={"thread_id": "eval-1"})
        compiled = graph.compile(callback_manager=mgr)
        await compiled.ainvoke(state, config)
        ```
    """
    cb = PublisherCallback(collector, config=config)
    mgr = CallbackManager()
    mgr.register_after_invoke(InvocationType.TOOL, cb)
    mgr.register_after_invoke(InvocationType.MCP, cb)
    mgr.register_after_invoke(InvocationType.AI, cb)
    return collector, mgr
