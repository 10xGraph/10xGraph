"""Inject services into a hand-written graph node and read them from the container.

Nodes take state and config plus any Inject[...] defaults. No LLM is called, so this file runs
without an API key.
Run: python examples/dependency_injection/nodes.py
"""

import asyncio

from di import OrderRepository, Settings, build_container, list_orders
from injectq import Inject, InjectQ

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import TextBlock, ToolCallBlock
from tenxgraph.utils.constants import END
from tenxgraph.utils.injection import fresh


async def planner(
    state: AgentState,
    config: dict,
    settings: Settings = Inject[Settings],
) -> Message:
    """A scripted 'model': asks for alice's orders first, then summarises the tool result."""
    settings = fresh(settings)
    if state.context[-1].role == "tool":
        return Message(
            role="assistant",
            content=[TextBlock(text=f"Done. Amounts are in {settings.currency}.")],
        )
    return Message(
        role="assistant",
        content=[ToolCallBlock(id="call-1", name="list_orders", args={"customer": "alice"})],
        tools_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "list_orders", "arguments": '{"customer": "alice"}'},
            }
        ],
    )


def route(state: AgentState) -> str:
    last = state.context[-1]
    return "TOOL" if last.role == "assistant" and last.tools_calls else END


def build_graph(container: InjectQ) -> StateGraph:
    graph = StateGraph(container=container)
    graph.add_node("MAIN", planner)
    graph.add_node("TOOL", ToolNode([list_orders]))
    graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")
    return graph


class FakeRepository(OrderRepository):
    """Test double: fixed data instead of the real repository."""

    def __init__(self) -> None:
        self._orders = {"alice": [("TEST-1", 1.0)]}


async def run(container: InjectQ, label: str) -> None:
    app = build_graph(container).compile()
    result = await app.ainvoke(
        {"messages": [Message.text_message("What did alice order?")]},
        config={"thread_id": f"di-{label}"},
    )
    tool_message = next(m for m in result["messages"] if m.role == "tool")
    print(f"{label} tool result:", tool_message.text())
    print(f"{label} assistant:", result["messages"][-1].text())
    await app.aclose()


async def main() -> None:
    # Production wiring.
    await run(build_container(), "real")

    # Test wiring: bind a different implementation, or swap a whole node with override_node.
    await run(build_container(repo=FakeRepository()), "fake-repo")

    graph = build_graph(build_container())
    stub = Message.text_message("stub", role="assistant")
    graph.override_node("MAIN", lambda state, config: stub)
    app = graph.compile()
    result = await app.ainvoke(
        {"messages": [Message.text_message("hi")]}, config={"thread_id": "t"}
    )
    print("override_node assistant:", result["messages"][-1].text())
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
