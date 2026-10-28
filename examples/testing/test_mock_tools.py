"""MockToolRegistry: fake tools that record every call.

Register plain functions by name, hand them to a ToolNode, and assert on how
they were called afterwards. No network, no LLM.
Run: pytest examples/testing/test_mock_tools.py -q
"""

import pytest

from tenxgraph.core.graph import ToolNode
from tenxgraph.qa.testing import MockToolRegistry


@pytest.fixture
def tools() -> MockToolRegistry:
    registry = MockToolRegistry()
    registry.register(
        "get_weather",
        lambda city: f"Sunny in {city}",
        description="Get the weather for a city.",
    )
    registry.register("send_email", lambda to, body: "sent")

    async def lookup(order_id: str) -> str:
        return f"order {order_id}: shipped"

    registry.register_async("lookup_order", lookup)
    return registry


def test_calls_are_recorded(tools: MockToolRegistry):
    fn = tools.functions["get_weather"]

    assert fn(city="NYC") == "Sunny in NYC"
    fn(city="Paris")

    assert tools.was_called("get_weather")
    assert not tools.was_called("send_email")
    assert tools.call_count("get_weather") == 2
    assert tools.get_last_call("get_weather")["kwargs"] == {"city": "Paris"}
    tools.assert_called("get_weather")
    tools.assert_called_with("get_weather", city="Paris")
    tools.assert_call_count("get_weather", 2)


async def test_async_tool(tools: MockToolRegistry):
    result = await tools.functions["lookup_order"](order_id="A1")

    assert result == "order A1: shipped"
    tools.assert_called_with("lookup_order", order_id="A1")


def test_reset_keeps_tools_but_clears_calls(tools: MockToolRegistry):
    tools.functions["send_email"](to="a@b.c", body="hi")
    tools.reset()

    assert tools.call_count("send_email") == 0
    assert "send_email" in tools.functions  # reset() keeps registrations

    tools.clear()
    assert tools.functions == {}  # clear() drops them too


def test_assertion_failures(tools: MockToolRegistry):
    with pytest.raises(AssertionError):
        tools.assert_called("send_email")


async def test_plugs_into_tool_node(tools: MockToolRegistry):
    # get_tool_list() returns the tracked functions, ready for ToolNode.
    node = ToolNode(tools.get_tool_list())

    schemas = await node.all_tools()
    names = {s["function"]["name"] for s in schemas}
    assert {"get_weather", "send_email", "lookup_order"} <= names
