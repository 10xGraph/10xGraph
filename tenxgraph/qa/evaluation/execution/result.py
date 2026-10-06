"""
Execution result container for agent evaluation.

Holds extracted tool calls, trajectory and response
built by AgentEvaluator from a TrajectoryCollector.
"""

from __future__ import annotations

import warnings
from typing import Any

from pydantic import BaseModel, Field

from agentflow.qa.evaluation.dataset.eval_set import StepType, ToolCall, TrajectoryStep
from agentflow.qa.evaluation.token_usage import TokenUsage


class NodeResponseData(BaseModel):
    """Serialisable snapshot of a single AI-node invocation.

    Matches the fields of the ``NodeResponse`` dataclass in the
    trajectory collector but as a Pydantic model so it survives
    serialisation into ``EvalCaseResult`` and reporters.

    Attributes:
        node_name:       Name of the graph node that ran.
        input_messages:  Conversation history going into this node.
        response_text:   LLM text output; empty on tool-call turns.
        has_tool_calls:  Whether the LLM decided to call tools.
        tool_call_names: Names of tools requested.
        is_final:        True when no further tool calls.
        timestamp:       Wall-clock time when this invocation completed.
    """

    node_name: str = ""
    input_messages: list[dict[str, Any]] = Field(default_factory=list)
    response_text: str = ""
    has_tool_calls: bool = False
    tool_call_names: list[str] = Field(default_factory=list)
    is_final: bool = False
    timestamp: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class ExecutionResult(BaseModel):
    """Holds extracted execution data built from a TrajectoryCollector.

    Populated by ``AgentEvaluator._execution_from_collector()`` and passed
    directly to criteria for evaluation, then carried through to
    ``EvalCaseResult`` and reporters.

    Attributes:
        tool_calls:      Tool calls captured during graph execution.
        trajectory:      Full ordered trajectory (NODE + TOOL steps).
        messages:        Flat, de-duplicated message history from node inputs.
        actual_response: Final text response from the agent.
        node_responses:  Per-node input/output snapshots for intermediate audit.
        node_visits:     Ordered list of node names visited.
        duration_seconds: Wall-clock execution time (graph only, not criteria).
    """

    tool_calls: list[ToolCall] = Field(default_factory=list)
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    actual_response: str = ""
    node_responses: list[NodeResponseData] = Field(default_factory=list)
    node_visits: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0

    # --- Computed helpers --------------------------------------------------

    @property
    def token_usage(self) -> TokenUsage:
        """Aggregate token usage across all node invocations in this case."""
        total = TokenUsage()
        for nr in self.node_responses:
            total = total + nr.token_usage
        return total

    @property
    def tool_trajectory(self) -> list[TrajectoryStep]:
        """Return only TOOL steps — used by TrajectoryMatchCriterion."""
        return [s for s in self.trajectory if s.step_type == StepType.TOOL]

    def get_tool_names(self) -> list[str]:
        """Get list of tool names called in order.

        Used by ToolNameMatchCriterion.

        Returns:
            List of tool name strings in call order.
        """
        return [tc.name for tc in self.tool_calls]

    # --- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize execution result for logging or reporting.

        .. deprecated::
            Use ``model_dump()`` instead.
        """
        warnings.warn(
            "ExecutionResult.to_dict() is deprecated, use model_dump()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.model_dump()

    def __repr__(self) -> str:
        return (
            f"ExecutionResult("
            f"tools={len(self.tool_calls)}, "
            f"steps={len(self.trajectory)}, "
            f"nodes={len(self.node_responses)}, "
            f"response_len={len(self.actual_response)})"
        )
