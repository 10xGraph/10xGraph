"""override_node: swap a production node for a test double.

The graph factory below is the "production" code. In a test we override its
LLM node with a TestAgent, keeping the real tool node and routing.
Run: pytest examples/testing/test_override_node.py -q
"""

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.qa.testing import MockToolRegistry
from tenxgraph.qa.testing import TestAgent as CannedAgent
from tenxgraph.utils.constants import END


async def production_llm(state: AgentState) -> list[Message]:
    # Stands in for a real Agent; calling it in a test would need network access.
    raise RuntimeError("tests must not reach the real model")


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


def route(state: AgentState) -> str:
    last = state.context[-1] if state.context else None
    if last and getattr(last, "tools_calls", None) and last.role == "assistant":
        return "TOOL"
    return END


def create_graph() -> StateGraph:
    """Production factory: returns the uncompiled graph so callers can override nodes."""
    graph = StateGraph()
    graph.add_node("MAIN", production_llm)
    graph.add_node("TOOL", ToolNode([get_weather]))
    graph.set_entry_point("MAIN")
    graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MAIN")
    return graph


async def test_override_before_compile():
    graph = create_graph()
    graph.override_node("MAIN", CannedAgent(responses=["stubbed answer"]))
    app = graph.compile()

    result = await app.ainvoke({"messages": [Message.text_message("hi")]})

    assert result["messages"][-1].text() == "stubbed answer"


async def test_override_after_compile():
    app = create_graph().compile()
    app.override_node("MAIN", CannedAgent(responses=["post-compile stub"]))

    result = await app.ainvoke({"messages": [Message.text_message("hi")]})

    assert result["messages"][-1].text() == "post-compile stub"


async def test_override_with_mock_tool_node():
    # Replace the tool node too, so we can assert the agent's tool call reached it.
    registry = MockToolRegistry().register("get_weather", lambda **kw: "mock sunny")
    graph = create_graph()
    graph.override_node(
        "MAIN",
        CannedAgent(responses=["It is sunny."], tools=["get_weather"]),
    )
    graph.override_node("TOOL", ToolNode(registry.get_tool_list()))
    app = graph.compile()

    result = await app.ainvoke(
        {"messages": [Message.text_message("Weather?")]},
        config={"recursion_limit": 10},
    )

    assert result["messages"][-1].text() == "It is sunny."
    registry.assert_called("get_weather")


def test_unknown_node_raises():
    with pytest.raises(KeyError):
        create_graph().override_node("MISSING", CannedAgent())
