from dataclasses import dataclass, field
from typing import Any

from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.utils import Message


@dataclass
class DebugState(AgentState):
    items: list[str] = field(default_factory=list)
    count: int = 0


async def debug_node(state: DebugState, config: dict[str, Any]) -> DebugState:
    print("=== Inside debug_node ===")
    print(f"state parameter type: {type(state)}")
    print(f"state.__dict__: {state.__dict__}")

    # Check each field specifically
    for field_name in ["items", "count", "context", "context_summary", "execution_meta"]:
        if hasattr(state, field_name):
            attr_value = getattr(state, field_name)
            print(f"state.{field_name} type: {type(attr_value)}")
            print(f"state.{field_name} value: {attr_value}")
        else:
            print(f"state.{field_name}: NOT FOUND")

    # Try to access items specifically
    try:
        items_attr = state.items
        print(f"Direct access to state.items: {type(items_attr)} = {items_attr}")
    except Exception as e:
        print(f"Error accessing state.items: {e}")

    return state


# Test step by step
print("=== Creating graph ===")
initial_state = DebugState()
print(f"Initial state type: {type(initial_state)}")
print(f"Initial state.items type: {type(initial_state.items)}")

graph = StateGraph[DebugState](initial_state)
print(f"Graph state type: {type(graph.state)}")
print(f"Graph state.items type: {type(graph.state.items)}")

graph.add_node("DEBUG", debug_node)
graph.set_entry_point("DEBUG")

print("\n=== Compiling graph ===")
checkpointer = InMemoryCheckpointer[DebugState]()
app = graph.compile(checkpointer=checkpointer)

print("\n=== Invoking graph ===")
config = {"thread_id": "debug"}
try:
    result = app.invoke({"messages": [Message.from_text("Debug")]}, config=config)
    print(f"Final result: {result}")
except Exception as e:
    print(f"Error during invoke: {e}")
    raise
