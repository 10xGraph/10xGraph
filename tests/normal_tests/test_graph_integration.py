#!/usr/bin/env python3

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.utils import Message


def test_graph_integration():
    """Test that the graph module works with the converted Pydantic models"""
    print("=== Testing Graph Integration ===")

    # Create a simple state
    state = AgentState()
    print(f"Initial state: {type(state)}")

    # Create a graph
    graph = StateGraph[AgentState](state)
    print(f"Graph created with state type: {type(graph.state)}")

    # Add a simple node
    def simple_node(state: AgentState, config: dict) -> AgentState:
        print(f"Node received state type: {type(state)}")
        state.context.append(Message.from_text("Hello from node"))
        return state

    graph.add_node("simple", simple_node)
    graph.set_entry_point("simple")

    # Test compilation
    compiled = graph.compile()
    print(f"Graph compiled successfully: {type(compiled)}")

    # Test execution
    result = compiled.invoke({"messages": [Message.from_text("Test input")]})
    print(f"Execution result keys: {list(result.keys())}")
    print(f"Messages in result: {len(result.get('messages', []))}")

    return True


if __name__ == "__main__":
    try:
        success = test_graph_integration()
        if success:
            print("\n✅ Graph integration test passed!")
        else:
            print("\n❌ Graph integration test failed!")
    except Exception as e:
        print(f"\n❌ Graph integration test failed with error: {e}")
        import traceback

        traceback.print_exc()
