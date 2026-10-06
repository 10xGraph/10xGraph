"""Test that graph resets to entry point after successful completion."""

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.utils.constants import END
from tenxgraph.utils import ResponseGranularity


def node_a(state: AgentState, config: dict) -> Message:
    """First node in the graph."""
    return Message.text_message("Node A executed", role="assistant")


def node_b(state: AgentState, config: dict) -> Message:
    """Second node in the graph."""
    return Message.text_message("Node B executed", role="assistant")


def route_to_b_or_end(state: AgentState) -> str:
    """Route to node B on first call, END on subsequent calls."""
    # Check the last message to determine routing
    if state.context and len(state.context) >= 2:
        last_message = state.context[-1]
        if "Node A executed" in last_message.text():
            return "B"
    return END


def test_graph_resets_to_entry_point_after_completion():
    """Test that a completed graph resets to entry point on next invocation."""
    # Build a simple graph: START -> A -> (B or END)
    graph = StateGraph()
    graph.add_node("A", node_a)
    graph.add_node("B", node_b)

    # Add conditional edge from A
    graph.add_conditional_edges(
        "A",
        route_to_b_or_end,
        {"B": "B", END: END},
    )

    # B goes to END
    graph.add_edge("B", END)
    graph.set_entry_point("A")

    app = graph.compile()

    # First execution: should go A -> B -> END
    config = {"thread_id": "test-reset-123", "recursion_limit": 10}
    inp = {"messages": [Message.text_message("First message", role="user")]}
    res = app.invoke(inp, config=config, response_granularity=ResponseGranularity.FULL)

    # Verify first execution completed successfully
    assert res["state"].execution_meta.status == ExecutionStatus.COMPLETED
    assert len(res["messages"]) == 3  # user message + A message + B message
    assert "Node A executed" in res["messages"][1].text()
    assert "Node B executed" in res["messages"][2].text()

    # Second execution with same thread_id: should reset and start from A again
    inp2 = {"messages": [Message.text_message("Second message", role="user")]}
    res2 = app.invoke(inp2, config=config, response_granularity=ResponseGranularity.FULL)

    # Verify second execution also completed successfully
    assert res2["state"].execution_meta.status == ExecutionStatus.COMPLETED
    # Should have accumulated messages from both runs
    # First run: user1, A, B | Second run: user2, A (no B because route goes to END)
    assert len(res2["messages"]) >= 2  # At least user2 message + A message

    # Most importantly: verify the second execution went through node A
    # (proving it reset to entry point rather than staying at B or END)
    messages_text = [msg.text() for msg in res2["messages"]]
    assert "Node A executed" in messages_text[-1] or any(
        "Node A executed" in txt for txt in messages_text[-2:]
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

