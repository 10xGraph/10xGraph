"""Base Agent abstract class for Agentflow.

This module provides the BaseAgent abstract class that defines the common interface
for all agents - both production and test agents. This allows swapping between
Agent and TestAgent seamlessly for testability.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from agentflow.core.graph.tool_node.base import ToolNode

from agentflow.core.state import AgentState
from agentflow.core.state.message import Message


logger = logging.getLogger("agentflow.agent")


class BaseAgent(ABC):
    """Base class for all agents - production and test.

    Provides the common interface that all agents must implement.
    This allows swapping between Agent and TestAgent seamlessly.

    Subclasses must implement:
    - execute(): Main execution logic
    - _call_llm(): LLM API call (can be mocked in tests)

    Attributes:
        model: LLM model identifier
        system_prompt: System prompt configuration
        tool_node: ToolNode instance or named graph-node reference (str)
        kwargs: Additional configuration parameters

    Example:
        ```python
        # Production agent
        agent = Agent(model="gpt-4", system_prompt=[...])

        # Test agent (same interface)
        test_agent = TestAgent(model="gpt-4", responses=["Hello!"])
        ```
    """

    def __init__(
        self,
        model: str,
        provider: str | None = None,
        system_prompt: list[dict[str, Any]] | None = None,
        tool_node: "str | ToolNode | None" = None,
        extra_messages: list[Message] | None = None,
        client: Any = None,  # Escape hatch: allow custom client
        base_url: str | None = None,  # For OpenAI-compatible APIs (ollama, vllm, etc.)
        trim_context: bool = False,
        tools_tags: set[str] | None = None,
        **llm_kwargs,
    ):
        """Initialize a BaseAgent.

        Args:
            model: LLM model identifier (e.g., "gpt-4", "gemini/gemini-2.0-flash")
            system_prompt: System prompt as list of message dicts
            tool_node: ToolNode instance, or a string naming an existing graph node
                whose ``func`` is a ToolNode.  Pass ``None`` for no tools.
            **kwargs: Additional LLM or agent configuration parameters
        """
        self.model = model
        self.system_prompt = system_prompt or []
        self.tool_node = tool_node
        self.llm_kwargs = llm_kwargs

    @abstractmethod
    async def execute(
        self,
        state: AgentState,
        config: dict[str, Any],
    ) -> Any:
        """Execute the agent logic.

        This is the main entry point called by the graph during node execution.

        Args:
            state: Current agent state with context
            config: Execution configuration

        Returns:
            ModelResponseConverter, list[Message], or dict
        """

    @abstractmethod
    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        **kwargs: Any,
    ) -> Any:
        """Make the actual LLM call.

        This is the method that differs between production and test agents.
        Production agents call real LLM APIs, while test agents return
        predefined responses.

        Args:
            messages: List of message dicts for the LLM
            tools: Optional list of tool specifications
            **kwargs: Additional LLM call parameters

        Returns:
            LLM response (format depends on implementation)
        """

    def get_tool_node(self) -> "ToolNode | None":
        """Return the agent's internal ToolNode, or None if not configured.

        Use this instead of accessing ``agent._tool_node`` directly when you
        need to register the tool node as a separate graph node.

        Example::

            tool_node = ToolNode([my_tool])
            agent = Agent(model="gpt-4o", tool_node=tool_node)
            graph.add_node("TOOL", agent.get_tool_node())
        """
        return None

    async def __call__(
        self,
        state: AgentState,
        config: dict[str, Any],
    ) -> Any:
        """Make the agent callable like a function.

        This allows using the agent directly as a node function in the graph.
        Delegates to the execute() method.

        Args:
            state: Current agent state with context
            config: Execution configuration

        Returns:
            Result from execute() - ModelResponseConverter, list[Message], or dict
        """
        return await self.execute(state, config)
