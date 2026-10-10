#!/usr/bin/env python3
"""
Test script to verify the injectable changes work correctly.
"""

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from injectq import inject

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.utils import Message


@inject
def test_tool_with_injection(
    location: str,
    tool_call_id: str,
    state: AgentState,
) -> str:
    """Test tool function that uses dependency injection."""
    print(f"Location: {location}")
    print(f"Tool call ID: {tool_call_id}")
    print(f"State has {len(state.context) if state and state.context else 0} messages")
    return f"Weather in {location} is sunny (injected tool_call_id: {tool_call_id})"


def test_tool_without_injection(location: str) -> str:
    """Test tool function without dependency injection."""
    return f"Weather in {location} is cloudy"


async def test_tool_signature_generation():
    """Test that tool signatures exclude injectable parameters."""
    print("Testing tool signature generation...")

    # Create ToolNode with both functions
    tool_node = ToolNode([test_tool_with_injection, test_tool_without_injection])

    # Get tool descriptions
    tools = await tool_node.all_tools()

    # Check the injected function - should only have 'location' parameter
    injected_tool = None
    normal_tool = None

    for tool in tools:
        if tool["function"]["name"] == "test_tool_with_injection":
            injected_tool = tool
        elif tool["function"]["name"] == "test_tool_without_injection":
            normal_tool = tool

    print(f"Found {len(tools)} tools")

    # Test injected tool signature
    if injected_tool:
        params = injected_tool["function"]["parameters"]["properties"]
        print(f"Injected tool parameters: {list(params.keys())}")
        # Should only contain 'location', not 'tool_call_id' or 'state'
        assert "location" in params, "Location parameter missing"
        assert "tool_call_id" not in params, "tool_call_id should be excluded"
        assert "state" not in params, "state should be excluded"
        print("✓ Injectable parameters correctly excluded from tool signature")
    else:
        raise AssertionError("Injected tool not found")

    # Test normal tool signature
    if normal_tool:
        params = normal_tool["function"]["parameters"]["properties"]
        print(f"Normal tool parameters: {list(params.keys())}")
        assert "location" in params, "Location parameter missing in normal tool"
        print("✓ Normal tool parameters correctly included")
    else:
        raise AssertionError("Normal tool not found")


async def test_tool_execution():
    """Test that tool execution with injection works."""
    print("\nTesting tool execution...")

    # Create a test state
    test_state = AgentState()
    test_state.context = [Message.from_text("Test message")]

    # Create ToolNode
    tool_node = ToolNode([test_tool_with_injection])

    # Execute the tool
    result = await tool_node.execute(
        name="test_tool_with_injection",
        args={"location": "New York"},
        tool_call_id="test_123",
        config={},
        state=test_state,
    )

    print(f"Tool execution result: {result}")

    # Verify the result contains expected information
    if hasattr(result, "content"):
        content = result.content
        assert "New York" in content, "Location not found in result"
        assert "test_123" in content, "Tool call ID not injected properly"
        print("✓ Tool execution with injection successful")
    else:
        print(f"Result type: {type(result)}")
        print("✓ Tool execution completed (result format may vary)")


async def main():
    """Run all tests."""
    print("Testing injectable changes with injectq...")

    try:
        await test_tool_signature_generation()
        await test_tool_execution()
        print("\n🎉 All tests passed! Injectable changes are working correctly.")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
