"""A realtime session may only run the tools it advertised, and only narrow its tag filter.

The API passes the client's init frame through as per-session overrides. A client that could
widen ``tools_tags`` (or clear it, which advertises everything) or call a tool by name would
reach tools the agent was built to hide.
"""

import pytest

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.realtime.base import (
    RealtimeConfig,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
)
from tenxgraph.core.realtime.live_agent import LiveAgent, _advertised_tool_names
from tenxgraph.core.realtime.queue import LiveInputQueue
from tenxgraph.utils import tool
from tests.realtime.test_live_agent import MODEL, FakeRealtimeClient


@tool(tags=["weather"])
def get_weather(city: str) -> str:
    """Weather."""
    return f"sunny in {city}"


@tool(tags=["admin"])
def delete_everything() -> str:
    """Admin only."""
    return "deleted"


def _agent(client, tags=None):
    return LiveAgent(
        MODEL,
        realtime_config=RealtimeConfig(model=MODEL, tools_tags=tags),
        tool_node=ToolNode([get_weather, delete_everything]),
        realtime_client_factory=lambda: client,
    )


async def _run(agent, config):
    queue = LiveInputQueue()
    queue.close()
    return [event async for event in agent.arun(queue, config)]


def _advertised(client):
    return sorted(t["function"]["name"] for t in client.connected_config.tools)


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", [["admin"], [], ["weather", "admin"], "admin"])
async def test_session_cannot_widen_the_tag_filter(requested):
    client = FakeRealtimeClient([TurnCompleteEvent()])
    await _run(_agent(client, tags=["weather"]), {"realtime": {"tools_tags": requested}})
    assert _advertised(client) == ["get_weather"]


@pytest.mark.asyncio
async def test_session_can_narrow_an_unfiltered_agent():
    client = FakeRealtimeClient([TurnCompleteEvent()])
    await _run(_agent(client), {"realtime": {"tools_tags": ["weather"]}})
    assert _advertised(client) == ["get_weather"]


@pytest.mark.asyncio
async def test_call_to_an_unadvertised_tool_is_refused():
    client = FakeRealtimeClient([ToolCallEvent(id="c1", name="delete_everything", args={})])
    events = await _run(_agent(client, tags=["weather"]), {})

    result = next(e for e in events if isinstance(e, ToolResultEvent)).result
    assert "not available" in result["error"]
    assert client.tool_responses[0][2] == result


@pytest.mark.asyncio
async def test_call_to_an_advertised_tool_runs():
    client = FakeRealtimeClient([ToolCallEvent(id="c1", name="get_weather", args={"city": "Oslo"})])
    await _run(_agent(client, tags=["weather"]), {})
    assert client.tool_responses[0][2] == {"result": "sunny in Oslo"}


def test_advertised_names_from_every_tool_shape():
    class Decl:
        name = "native"

    class NativeTool:
        function_declarations = [Decl()]

    tools = [
        {"type": "function", "function": {"name": "openai_style"}},
        {"name": "flat"},
        NativeTool(),
        {"sentinel": True},
    ]
    assert _advertised_tool_names(tools) == {"openai_style", "flat", "native"}
