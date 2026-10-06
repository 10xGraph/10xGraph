"""
Evaluation set and case data models.

This module defines the core data structures for agent evaluation:
    - EvalSet: Collection of evaluation cases
    - EvalCase: Single test scenario with expected outcomes
    - Invocation: Single turn in a conversation
    - ToolCall: Expected or actual tool call
    - TrajectoryStep: Single step in execution trajectory
    - SessionInput: Initial session configuration
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StepType(StrEnum):
    """Type of step in execution trajectory."""

    NODE = "node"
    TOOL = "tool"
    MESSAGE = "message"
    CONDITIONAL = "conditional"


class ToolCall(BaseModel):
    """Represents a tool call (expected or actual)."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None
    result: Any | None = None

    def matches(
        self,
        other: ToolCall,
        check_args: bool = True,
        check_call_id: bool = False,
    ) -> bool:
        if self.name != other.name:
            return False
        # Only enforce args comparison when the *expected* spec (other) has args
        # defined.  If other.args is empty it means "any args are acceptable".
        if check_args and other.args and self.args != other.args:
            return False
        return not (check_call_id and self.call_id != other.call_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return False
        return self.name == other.name and self.args == other.args

    def __hash__(self) -> int:
        return hash((self.name, frozenset(self.args.items())))


class TrajectoryStep(BaseModel):
    """Single step in an execution trajectory."""

    step_type: StepType
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    timestamp: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def node(
        cls,
        name: str,
        timestamp: float | None = None,
        **metadata: Any,
    ) -> TrajectoryStep:
        """Create a node step."""
        return cls(
            step_type=StepType.NODE,
            name=name,
            timestamp=timestamp,
            metadata=metadata,
        )

    @classmethod
    def tool(
        cls,
        name: str,
        args: dict[str, Any] | None = None,
        timestamp: float | None = None,
        **metadata: Any,
    ) -> TrajectoryStep:
        """Create a tool step."""
        return cls(
            step_type=StepType.TOOL,
            name=name,
            args=args or {},
            timestamp=timestamp,
            metadata=metadata,
        )


class MessageContent(BaseModel):
    """Content of a message (simplified from full Message model)."""

    role: str
    content: str | list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def user(cls, text: str) -> MessageContent:
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, text: str) -> MessageContent:
        return cls(role="assistant", content=text)

    def get_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        texts = []
        for block in self.content:
            if isinstance(block, dict):
                if "text" in block:
                    texts.append(block["text"])
                elif block.get("type") == "text":
                    texts.append(block.get("text", ""))
        return " ".join(texts)


class Invocation(BaseModel):
    """A single turn in the conversation."""

    invocation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_content: MessageContent
    expected_tool_trajectory: list[ToolCall] = Field(default_factory=list)
    expected_node_order: list[str] = Field(default_factory=list)
    expected_intermediate_responses: list[MessageContent] = Field(default_factory=list)
    expected_final_response: MessageContent | None = None

    @classmethod
    def simple(
        cls,
        user_query: str,
        expected_response: str | None = None,
        expected_tools: list[ToolCall] | None = None,
        expected_node_order: list[str] | None = None,
    ) -> Invocation:
        return cls(
            user_content=MessageContent.user(user_query),
            expected_final_response=(
                MessageContent.assistant(expected_response) if expected_response else None
            ),
            expected_tool_trajectory=expected_tools or [],
            expected_node_order=expected_node_order or [],
        )


class SessionInput(BaseModel):
    """Initial session configuration for evaluation."""

    app_name: str = ""
    user_id: str = "test_user"
    state: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    """A single evaluation case representing one test scenario."""

    eval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    conversation: list[Invocation] = Field(default_factory=list)
    session_input: SessionInput = Field(default_factory=SessionInput)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def single_turn(
        cls,
        eval_id: str,
        user_query: str,
        expected_response: str | None = None,
        expected_tools: list[ToolCall] | None = None,
        expected_node_order: list[str] | None = None,
        name: str = "",
        description: str = "",
    ) -> EvalCase:
        """Create a single-turn evaluation case."""
        return cls(
            eval_id=eval_id,
            name=name,
            description=description,
            conversation=[
                Invocation.simple(
                    user_query=user_query,
                    expected_response=expected_response,
                    expected_tools=expected_tools,
                    expected_node_order=expected_node_order,
                )
            ],
        )

    @classmethod
    def multi_turn(
        cls,
        eval_id: str,
        conversation: list[tuple[str, str]],
        expected_tools: list[ToolCall] | None = None,
        name: str = "",
        description: str = "",
    ) -> EvalCase:
        """Create a multi-turn evaluation case.

        Args:
            eval_id: Unique identifier for this case.
            conversation: List of (user_query, expected_response) tuples.
            expected_tools: Expected tool calls for the first invocation.
            name: Human-readable name.
            description: Description of the test.

        Returns:
            Configured EvalCase with one Invocation per conversation turn.

        Example:
            ```python
            case = EvalCase.multi_turn(
                eval_id="chat_test",
                conversation=[
                    ("Hello", "Hi there!"),
                    ("What is the weather?", "It is sunny."),
                ],
                expected_tools=[ToolCall(name="get_weather", args={})],
            )
            ```
        """
        invocations = []
        for i, (user_query, expected_response) in enumerate(conversation):
            # Only attach expected_tools to the first invocation
            tools = expected_tools if i == 0 else None
            invocations.append(
                Invocation.simple(
                    user_query=user_query,
                    expected_response=expected_response,
                    expected_tools=tools,
                )
            )

        return cls(
            eval_id=eval_id,
            name=name,
            description=description,
            conversation=invocations,
        )


class EvalSet(BaseModel):
    """A collection of evaluation cases."""

    eval_set_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    eval_cases: list[EvalCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.eval_cases)

    def __iter__(self):
        return iter(self.eval_cases)

    def add_case(self, case: EvalCase) -> None:
        self.eval_cases.append(case)

    def get_case(self, eval_id: str) -> EvalCase | None:
        for case in self.eval_cases:
            if case.eval_id == eval_id:
                return case
        return None

    def filter_by_tags(self, tags: list[str]) -> list[EvalCase]:
        return [case for case in self.eval_cases if all(tag in case.tags for tag in tags)]

    @classmethod
    def from_file(cls, path: str) -> EvalSet:
        """Load an EvalSet from a JSON file."""
        import json
        from pathlib import Path

        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_file(self, path: str) -> None:
        """Save the EvalSet to a JSON file."""
        import json
        from pathlib import Path

        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    # Alias so builder.save() works
    def save(self, path: str) -> None:
        """Alias for to_file() — save EvalSet to JSON file."""
        self.to_file(path)
