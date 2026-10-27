"""QuickTest: one-call tests for common agent scenarios.

QuickTest builds the graph, runs it with a canned agent, and hands back a
TestResult with chainable assertions.
Run: pytest examples/testing/test_quick_test.py -q
"""

import pytest

from tenxgraph.qa.testing import QuickTest
from tenxgraph.qa.testing import TestAgent as CannedAgent


async def test_single_turn():
    result = await QuickTest.single_turn(
        agent_response="Hello! How can I help?",
        user_message="Hi",
    )

    # Assertions return the result, so they chain.
    result.assert_contains("Hello").assert_not_contains("goodbye").assert_no_errors()
    assert result.final_response == "Hello! How can I help?"


async def test_multi_turn():
    result = await QuickTest.multi_turn(
        [
            ("Hello", "Hi! How can I help?"),
            ("What is 2 + 2?", "4"),
            ("Thanks", "You're welcome!"),
        ]
    )

    result.assert_contains("welcome")
    # 3 user messages + 3 assistant replies
    result.assert_message_count(6)


async def test_with_tools():
    # Strings become mock tools that record their calls and return a canned value.
    result = await QuickTest.with_tools(
        query="Weather in NYC?",
        response="It is sunny in NYC.",
        tools=["get_weather"],
        tool_responses={"get_weather": "sunny, 24C"},
    )

    result.assert_tool_called("get_weather").assert_contains("sunny")
    result.assert_tool_not_called("send_email")


async def test_custom_agent():
    agent = CannedAgent(responses=["custom reply"])
    result = await QuickTest.custom(agent=agent, user_message="anything")

    result.assert_equals("custom reply")
    agent.assert_called_times(1)


async def test_failed_assertion_message():
    result = await QuickTest.single_turn(agent_response="abc", user_message="x")
    with pytest.raises(AssertionError, match="Expected response to contain"):
        result.assert_contains("missing")
