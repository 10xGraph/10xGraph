from dataclasses import dataclass, field

from tenxgraph.core.state.agent_state import AgentState


@dataclass
class TestState(AgentState):
    items: list[str] = field(default_factory=list)
    count: int = 0


# Test direct instantiation
print("=== Direct instantiation ===")
state1 = TestState()
print(f"state1.items type: {type(state1.items)}")
print(f"state1.items value: {state1.items}")
print(f"state1.count: {state1.count}")

# Test type() constructor
print("\n=== type() constructor ===")
state2 = type(state1)()
print(f"state2.items type: {type(state2.items)}")
print(f"state2.items value: {state2.items}")
print(f"state2.count: {state2.count}")

# Test appending
print("\n=== Testing append ===")
try:
    state2.items.append("test")
    print("Append successful!")
    print(f"state2.items after append: {state2.items}")
except Exception as e:
    print(f"Append failed: {e}")
    print(
        f"state2.items.__dict__: {state2.items.__dict__ if hasattr(state2.items, '__dict__') else 'No __dict__'}"
    )
