from dataclasses import dataclass, field

from tenxgraph.core.state.agent_state import AgentState


@dataclass
class TestState(AgentState):
    items: list[str] = field(default_factory=list)
    count: int = 0


# Test 1: Direct instantiation
print("=== Test 1: Direct instantiation ===")
state1 = TestState()
print(f"state1.items type: {type(state1.items)}")
print(f"state1.items value: {state1.items}")

# Test 2: type() constructor
print("\n=== Test 2: type() constructor ===")
StateClass = type(state1)
state2 = StateClass()
print(f"state2.items type: {type(state2.items)}")
print(f"state2.items value: {state2.items}")

# Test 3: __class__ constructor
print("\n=== Test 3: __class__ constructor ===")
state3 = state1.__class__()
print(f"state3.items type: {type(state3.items)}")
print(f"state3.items value: {state3.items}")

# Test 4: Check fields
print("\n=== Test 4: Check dataclass fields ===")
import dataclasses


fields = dataclasses.fields(TestState)
for f in fields:
    print(f"Field {f.name}: {f}")

# Test 5: What happens with the actual state from the graph
from tenxgraph.core.graph import StateGraph


print("\n=== Test 5: StateGraph state ===")
graph = StateGraph[TestState](TestState())
print(f"Graph state type: {type(graph.state)}")
print(f"Graph state.items type: {type(graph.state.items)}")

# Try to create new state the same way
StateClass = type(graph.state)
print(f"StateClass: {StateClass}")
new_state = StateClass()
print(f"New state type: {type(new_state)}")
print(f"New state.items type: {type(new_state.items)}")
print(f"New state.items value: {new_state.items}")

# Test 6: Check if the graph state itself is corrupted
print("\n=== Test 6: Check graph state ===")
print(f"graph.state.__dict__: {graph.state.__dict__}")
print(f"hasattr items: {hasattr(graph.state, 'items')}")
items_attr = getattr(graph.state, "items", None)
print(f"getattr items: {items_attr}")
print(f"getattr items type: {type(items_attr)}")
