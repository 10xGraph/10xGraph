"""Examples demonstrating QuickTest - simplified testing."""

import asyncio

from tenxgraph.qa.testing import QuickTest


async def example_single_turn():
    """Example: Single-turn Q&A test."""
    print("=" * 60)
    print("Example 1: Single Turn Test")
    print("=" * 60)

    # BEFORE: Would need ~20 lines of setup
    # NOW: Just 3 lines!
    result = await QuickTest.single_turn(
        agent_response="Hello! How can I help you today?",
        user_message="Hi there",
    )

    # Chainable assertions
    result.assert_contains("Hello")
    result.assert_contains("help")
    result.assert_no_errors()

    print(f"✓ Test passed! Response: {result.final_response}\n")


async def example_multi_turn():
    """Example: Multi-turn conversation test."""
    print("=" * 60)
    print("Example 2: Multi-Turn Conversation")
    print("=" * 60)

    result = await QuickTest.multi_turn(
        [
            ("Hello", "Hi! How can I help you?"),
            ("What's the weather?", "I'll check the weather for you."),
            ("Thank you", "You're welcome!"),
        ]
    )

    result.assert_contains("welcome")
    result.assert_message_count(6)  # 3 user + 3 assistant

    print(f"✓ Multi-turn test passed! Final: {result.final_response}\n")


async def example_with_tools():
    """Example: Test agent with tool calls."""
    print("=" * 60)
    print("Example 3: Agent with Tools")
    print("=" * 60)

    result = await QuickTest.with_tools(
        query="What's the weather in New York?",
        response="The weather in New York is sunny, 72°F",
        tools=["get_weather"],
        tool_responses={"get_weather": "Sunny, 72°F"},
    )

    result.assert_contains("sunny")
    result.assert_tool_called("get_weather")

    print(f"✓ Tool test passed! Response: {result.final_response}\n")


async def example_assertions():
    """Example: Various assertion methods."""
    print("=" * 60)
    print("Example 4: Assertion Methods")
    print("=" * 60)

    result = await QuickTest.single_turn(
        agent_response="I can help you with Python programming.",
        user_message="Can you help with coding?",
    )

    # Multiple assertion styles
    (
        result.assert_contains("Python")
        .assert_contains("programming")
        .assert_not_contains("Java")
        .assert_no_errors()
    )

    print("✓ All assertions passed!\n")


async def main():
    """Run all examples."""
    print("\n🚀 QuickTest Examples\n")
    print("Demonstrating simplified testing with 90% less boilerplate\n")

    await example_single_turn()
    await example_multi_turn()
    await example_with_tools()
    await example_assertions()

    print("=" * 60)
    print("✓ All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
