from dataclasses import dataclass, field
from typing import Any

from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.utils import Message


@dataclass
class MinimalTestState(AgentState):
    items: list[str] = field(default_factory=list)
    count: int = 0


async def minimal_test_node(state: MinimalTestState, config: dict[str, Any]) -> MinimalTestState:
    print("=== Inside minimal_test_node ===")
    print(f"state parameter type: {type(state)}")
    print(f"state.items type: {type(state.items)}")
    print(f"state.items value: {state.items}")
    print(f"state.count: {state.count}")

    # Try to modify the state
    try:
        state.items.append("test_item")
        state.count += 1
        print(f"Successfully modified state. items: {state.items}, count: {state.count}")
    except Exception as e:
        print(f"Error modifying state: {e}")
        raise

    return state


# Test step by step
print("=== Creating graph ===")
initial_state = MinimalTestState()
print(f"Initial state type: {type(initial_state)}")
print(f"Initial state.items type: {type(initial_state.items)}")

graph = StateGraph[MinimalTestState](initial_state)
print(f"Graph state type: {type(graph.state)}")
print(f"Graph state.items type: {type(graph.state.items)}")

graph.add_node("TEST", minimal_test_node)
graph.set_entry_point("TEST")

print("\n=== Compiling graph ===")
checkpointer = InMemoryCheckpointer[MinimalTestState]()
app = graph.compile(checkpointer=checkpointer)

print("\n=== Invoking graph ===")
config = {"thread_id": "minimal_test"}
try:
    result = app.invoke({"messages": [Message.from_text("Test")]}, config=config)
    print(f"Final result: {result}")
except Exception as e:
    print(f"Error during invoke: {e}")
