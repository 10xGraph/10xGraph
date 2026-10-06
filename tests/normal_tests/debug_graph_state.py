from dataclasses import dataclass, field

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state.agent_state import AgentState


@dataclass
class TestState(AgentState):
    items: list[str] = field(default_factory=list)
    count: int = 0


print("=== Creating StateGraph ===")
graph = StateGraph[TestState](TestState())

print(f"Graph state type: {type(graph.state)}")
print(f"Graph state.items type: {type(graph.state.items)}")
print(f"Graph state.items value: {graph.state.items}")
print(f"Graph state.count: {graph.state.count}")

# Create new state using the same method as compiled_graph
print("\n=== Creating state via type() ===")
new_state = type(graph.state)()
print(f"New state type: {type(new_state)}")
print(f"New state.items type: {type(new_state.items)}")
print(f"New state.items value: {new_state.items}")
print(f"New state.count: {new_state.count}")

try:
    new_state.items.append("test")
    print("Append successful!")
    print(f"New state.items after append: {new_state.items}")
except Exception as e:
    print(f"Append failed: {e}")

# Check what happens if I modify the original graph state
print("\n=== Checking original state ===")
try:
    graph.state.items.append("original_test")
    print("Original state append successful!")
    print(f"Graph state.items after append: {graph.state.items}")
except Exception as e:
    print(f"Original state append failed: {e}")
    print(f"Graph state.items attributes: {dir(graph.state.items)[:5]}...")
