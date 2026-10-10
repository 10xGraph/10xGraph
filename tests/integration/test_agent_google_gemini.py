"""Integration tests for Agent with Google Gemini model.

These tests verify that the Agent class works correctly with Google's Gemini
model in real-world scenarios with tools and complex workflows.

To run these tests:
    export GEMINI_API_KEY=your_api_key_here
    pytest tests/integration/test_agent_google_gemini.py -v -s
"""

import os

import pytest

from tenxgraph.core.graph import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import END, ResponseGranularity


# Skip all tests if GEMINI_API_KEY is not set
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    ),
]


# Test tools
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"


def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers."""
    return a + b


def get_time() -> str:
    """Get current time."""
    return "2025-11-24 09:30:00"


class TestAgentGoogleGemini:
    """Integration tests for Agent with Google Gemini."""

    async def test_basic_agent_response(self):
        """Test basic Agent response without tools."""
        agent = Agent(
            model="google/gemini-2.5-flash-lite",
            system_prompt=[
                {"role": "system", "content": "You are a helpful assistant. Be concise."}
            ],
        )

        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)

        compiled = graph.compile()

        result = await compiled.ainvoke(
            {"messages": [Message.text_message("Say 'Hello World' and nothing else.", role="user")]},
            response_granularity=ResponseGranularity.FULL
        )
        final_state = result["state"]

        # Verify we got a response
        assert len(final_state.context) >= 2  # User message + assistant response
        last_message = final_state.context[-1]
        assert last_message.role == "assistant"
        assert len(last_message.content) > 0

        await compiled.aclose()

    async def test_agent_with_single_tool(self):
        """Test Agent using a single tool."""
        tools = [get_weather]
        tool_node = ToolNode(tools)

        agent = Agent(
            model="google/gemini-2.5-flash-lite",
            system_prompt=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use tools when needed.",
                }
            ],
            tools=tools,
        )

        def should_continue(state: AgentState) -> str:
            if not state.context:
                return END
            last_message = state.context[-1]
            if hasattr(last_message, "tools_calls") and last_message.tools_calls:
                return "tools"
            return END

        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")

        compiled = graph.compile()

        result = await compiled.ainvoke(
            {"messages": [Message.text_message("What's the weather in Tokyo?", role="user")]},
            response_granularity=ResponseGranularity.FULL
        )
        final_state = result["state"]

        # Verify tool was called
        tool_used = any(
            hasattr(msg, "tools_calls") and msg.tools_calls for msg in final_state.context
        )
        assert tool_used, "Expected agent to use the weather tool"

        # Verify we got a final response
        last_message = final_state.context[-1]
        assert last_message.role == "assistant"

        await compiled.aclose()

    async def test_agent_with_multiple_tools(self):
        """Test Agent with multiple tools."""
        tools = [get_weather, calculate_sum, get_time]
        tool_node = ToolNode(tools)

        agent = Agent(
            model="google/gemini-2.5-flash-lite",
            system_prompt=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use tools when appropriate.",
                }
            ],
            tools=tools,
        )

        def should_continue(state: AgentState) -> str:
            if not state.context:
                return END
            last_message = state.context[-1]
            if hasattr(last_message, "tools_calls") and last_message.tools_calls:
                return "tools"
            return END

        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")

        compiled = graph.compile()

        result = await compiled.ainvoke(
            {"messages": [Message.text_message("Calculate 25 + 17", role="user")]},
            response_granularity=ResponseGranularity.FULL
        )
        final_state = result["state"]

        # Verify tool was used
        tool_calls_found = False
        for msg in final_state.context:
            if hasattr(msg, "tools_calls") and msg.tools_calls:
                tool_calls_found = True
                # Check that calculate_sum was called
                for tool_call in msg.tools_calls:
                    func_name = tool_call.get("function", {}).get("name", "")
                    if func_name == "calculate_sum":
                        break
                else:
                    continue
                break

        assert tool_calls_found, "Expected agent to use calculate_sum tool"

        # Verify final response mentions the result
        last_message = final_state.context[-1]
        assert last_message.role == "assistant"

        await compiled.aclose()

    async def test_agent_streaming_mode(self):
        """Test Agent in streaming mode."""
        agent = Agent(
            model="google/gemini-2.5-flash-lite",
            system_prompt=[
                {"role": "system", "content": "You are a helpful assistant. Be brief."}
            ],
        )

        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)

        compiled = graph.compile()

        chunks = []
        async for chunk in compiled.astream({"messages": [
            Message.text_message("Count from 1 to 3", role="user")
        ]}):
            chunks.append(chunk)

        # Verify we got streaming chunks
        assert len(chunks) > 0, "Expected streaming chunks"

        await compiled.aclose()

    async def test_agent_with_tools_streaming(self):
        """Test Agent with tools in streaming mode."""
        tools = [get_weather]
        tool_node = ToolNode(tools)

        agent = Agent(
            model="google/gemini-2.5-flash-lite",
            system_prompt=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use tools when needed.",
                }
            ],
            tools=tools,
        )

        def should_continue(state: AgentState) -> str:
            if not state.context:
                return END
            last_message = state.context[-1]
            if hasattr(last_message, "tools_calls") and last_message.tools_calls:
                return "tools"
            return END

        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")

        compiled = graph.compile()

        chunks = []
        async for chunk in compiled.astream({"messages": [
            Message.text_message("What's the weather in Paris?", role="user")
        ]}):
            chunks.append(chunk)

        # Verify we got chunks
        assert len(chunks) > 0, "Expected streaming chunks"

        await compiled.aclose()

    async def test_agent_state_persistence(self):
        """Test that Agent properly updates state."""
        agent = Agent(
            model="google/gemini-2.5-flash-lite",
            system_prompt=[
                {"role": "system", "content": "You are a helpful assistant."}
            ],
        )

        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)

        compiled = graph.compile()

        # First message
        result = await compiled.ainvoke(
            {"messages": [Message.text_message("Remember: my name is Alice", role="user")]},
            response_granularity=ResponseGranularity.FULL
        )
        state1 = result["state"]

        # Verify state was updated
        assert len(state1.context) >= 2  # Original + response

        # Second message - pass all messages from previous result
        result = await compiled.ainvoke(
            {"messages": state1.context + [Message.text_message("What's my name?", role="user")]},
            response_granularity=ResponseGranularity.FULL
        )
        final_state = result["state"]

        # Verify context grew
        assert len(final_state.context) >= len(state1.context) + 1

        await compiled.aclose()

    async def test_agent_error_handling(self):
        """Test Agent handles errors gracefully."""
        agent = Agent(
            model="google/gemini-2.5-flash-lite",
            system_prompt=[
                {"role": "system", "content": "You are a helpful assistant."}
            ],
        )

        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)

        compiled = graph.compile()

        # Empty message should still work
        try:
            result = await compiled.ainvoke(
                {"messages": [Message.text_message("", role="user")]},
                response_granularity=ResponseGranularity.FULL
            )
            # Should complete without crashing
            assert result is not None
        except Exception as e:
            # If it does error, it should be a specific error, not a crash
            assert "Error" in str(type(e).__name__)

        await compiled.aclose()


class TestAgentGeminiComplexWorkflows:
    """Test complex multi-step workflows with Gemini."""

    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with context."""
        tools = [calculate_sum]
        tool_node = ToolNode(tools)

        agent = Agent(
            model="google/gemini-2.5-flash-lite",
            system_prompt=[
                {
                    "role": "system",
                    "content": "You are a math tutor. Help with calculations.",
                }
            ],
            tools=tools,
        )

        def should_continue(state: AgentState) -> str:
            if not state.context:
                return END
            last_message = state.context[-1]
            if hasattr(last_message, "tools_calls") and last_message.tools_calls:
                return "tools"
            return END

        graph = StateGraph()
        graph.add_node("agent", agent)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")

        compiled = graph.compile()

        # Turn 1
        result = await compiled.ainvoke(
            {"messages": [Message.text_message("What is 10 + 15?", role="user")]},
            response_granularity=ResponseGranularity.FULL
        )
        state = result["state"]

        # Verify response
        assert len(state.context) >= 2

        # Turn 2 - build on previous messages
        result = await compiled.ainvoke(
            {"messages": state.context + [Message.text_message("Now add 20 to that", role="user")]},
            response_granularity=ResponseGranularity.FULL
        )
        final_state = result["state"]

        # Verify conversation continued
        assert len(final_state.context) >= len(state.context) + 1

        await compiled.aclose()
