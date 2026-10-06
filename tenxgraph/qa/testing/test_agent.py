"""TestAgent for unit testing - returns predefined responses.

This module provides a TestAgent class that can be used to replace the
production Agent in tests, allowing for predictable and controlled testing
without making actual LLM API calls.
"""

import json
import logging
import uuid
from typing import Any

from agentflow.core.graph.base_agent import BaseAgent
from agentflow.core.state import AgentState
from agentflow.core.state.message import Message
from agentflow.core.state.message_block import ToolCallBlock
from agentflow.runtime.adapters.llm.base_converter import BaseConverter
from agentflow.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.utils.converter import convert_messages


logger = logging.getLogger("agentflow.testing")


class MockLLMResponse:
    """Mock response object used by TestAgent."""

    def __init__(
        self, content: str, test_id: str = "test-response", tools_calls: list | None = None
    ):
        """Initialize mock response.

        Args:
            content: The text content of the response
            id: Response ID (default: "test-response")
            tools_calls: Optional list of tool calls to include in response
        """
        self.id = test_id
        self.content = content
        self.tools_calls = tools_calls


class _TestAgentConverter(BaseConverter):
    """Convert TestAgent mock responses into assistant messages."""

    async def convert_response(self, response: MockLLMResponse) -> Message:
        msg = Message.text_message(response.content, role="assistant")
        if response.tools_calls:
            msg.tools_calls = response.tools_calls
            # Also add ToolCallBlock to content
            for tc in response.tools_calls:
                func_info = tc.get("function", {})
                args = func_info.get("arguments", {})
                # Parse JSON string args if needed
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                msg.content.append(
                    ToolCallBlock(
                        id=tc.get("id", str(uuid.uuid4())),
                        name=func_info.get("name", ""),
                        args=args,
                    )
                )
        return msg

    async def convert_streaming_response(
        self,
        config: dict,
        node_name: str,
        response: MockLLMResponse,
        meta: dict | None = None,
    ):
        yield await self.convert_response(response)


class TestAgent(BaseAgent):
    """Test agent for unit testing - returns predefined responses.

    Use this to swap out the production Agent in tests for predictable behavior
    without making actual LLM API calls.

    Attributes:
        responses: List of predefined responses to return
        call_count: Number of times the agent was called
        call_history: List of call details for assertions

    Example:
        ```python
        # Production code uses Agent
        agent = Agent(model="gpt-4", system_prompt=[...])

        # Test code uses TestAgent
        test_agent = TestAgent(model="gpt-4", system_prompt=[...], responses=["Hello from test!"])

        # Use in graph
        graph = StateGraph()
        graph.add_node("MAIN", test_agent)
        # ... or override existing node
        graph.override_node("MAIN", test_agent)
        ```
    """

    def __init__(
        self,
        model: str = "test-model",
        system_prompt: list[dict[str, Any]] | None = None,
        responses: list[str] | None = None,
        tools: list | None = None,
        simulate_tool_calls: bool = False,
        **kwargs: Any,
    ):
        """Initialize a TestAgent.

        Args:
            model: Model identifier (for compatibility, defaults to "test-model")
            system_prompt: System prompt configuration (optional for testing)
            responses: List of predefined responses to return. Cycles through
                the list on subsequent calls. Defaults to ["Test response"].
            tools: Optional tool configuration (for compatibility)
            simulate_tool_calls: If True, first call returns tool calls, second returns response
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            **kwargs,
        )
        self.responses = responses or ["Test response"]
        self.call_count = 0
        self.call_history: list[dict[str, Any]] = []
        self.simulate_tool_calls = simulate_tool_calls or (tools is not None and len(tools) > 0)
        self._tools_config = tools

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        **kwargs: Any,
    ) -> MockLLMResponse:
        """Return predefined response instead of calling LLM.

        This method simulates an LLM response by returning a predefined
        response from the responses list, cycling through if multiple
        calls are made.

        If simulate_tool_calls is True and tools are configured, the first
        call will return tool calls, and subsequent calls will return the
        actual response.

        Args:
            messages: List of message dicts (recorded for assertions)
            tools: Tool specifications (recorded for assertions)
            **kwargs: Additional parameters (recorded for assertions)

        Returns:
            MockLLMResponse object for test conversion
        """
        self.call_count += 1
        self.call_history.append(
            {
                "messages": messages,
                "tools": tools,
                "kwargs": kwargs,
            }
        )

        # Check if we should simulate tool calls
        if self.simulate_tool_calls and self._tools_config and self.call_count == 1:
            # First call: return tool calls
            tools_calls = []
            for i, tool in enumerate(self._tools_config):
                tool_name = (
                    tool if isinstance(tool, str) else getattr(tool, "__name__", f"tool_{i}")
                )
                tool_call = {
                    "id": f"tool-call-{i}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps({"query": "test query"}),
                    },
                }
                tools_calls.append(tool_call)

            logger.debug("TestAgent returning tool calls on first call")
            return MockLLMResponse(
                content="",
                test_id=f"test-response-{self.call_count}",
                tools_calls=tools_calls,
            )

        # Get next response (cycles through list)
        idx = (self.call_count - 1) % len(self.responses)
        content = self.responses[idx]

        logger.debug(
            "TestAgent returning response %d/%d: %s...",
            idx + 1,
            len(self.responses),
            content[:50] if len(content) > 50 else content,  # noqa: PLR2004
        )

        # Return MockLLMResponse that has model_dump() method
        return MockLLMResponse(
            content=content,
            test_id=f"test-response-{self.call_count}",
        )

    async def execute(
        self,
        state: AgentState,
        config: dict[str, Any],
    ) -> ModelResponseConverter:
        """Execute test agent - returns mock response.

        Args:
            state: Current agent state
            config: Execution configuration

        Returns:
            ModelResponseConverter wrapping the mock response
        """
        messages = convert_messages(
            state=state,
            system_prompts=self.system_prompt,
        )
        response = await self._call_llm(messages)

        return ModelResponseConverter(response, converter=_TestAgentConverter())

    # Assertion helpers for testing

    def assert_called(self) -> None:
        """Assert the agent was called at least once.

        Raises:
            AssertionError: If the agent was never called
        """
        assert self.call_count > 0, "TestAgent was never called"  # noqa: S101

    def assert_called_times(self, n: int) -> None:
        """Assert the agent was called exactly n times.

        Args:
            n: Expected number of calls

        Raises:
            AssertionError: If call count doesn't match
        """
        assert self.call_count == n, f"Expected {n} calls, got {self.call_count}"  # noqa: S101

    def assert_not_called(self) -> None:
        """Assert the agent was never called.

        Raises:
            AssertionError: If the agent was called
        """
        assert self.call_count == 0, f"Expected no calls, but got {self.call_count}"  # noqa: S101

    def get_last_messages(self) -> list[dict[str, Any]]:
        """Get messages from the last call.

        Returns:
            List of message dicts from the most recent call,
            or empty list if never called
        """
        if not self.call_history:
            return []
        return self.call_history[-1]["messages"]

    def get_last_tools(self) -> list | None:
        """Get tools from the last call.

        Returns:
            Tool specifications from the most recent call,
            or None if never called or no tools provided
        """
        if not self.call_history:
            return None
        return self.call_history[-1]["tools"]

    def reset(self) -> None:
        """Reset call count and history.

        Use this between tests or test cases to clear state.
        """
        self.call_count = 0
        self.call_history.clear()
