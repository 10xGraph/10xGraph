from dataclasses import dataclass, field
from typing import Any

from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.utils import Message


@dataclass
class TestState(AgentState):
    items: list[str] = field(default_factory=list)
    count: int = 0


async def debug_agent(state: TestState, config: dict[str, Any]) -> TestState:
    print(f"debug_agent received state type: {type(state)}")
    print(f"debug_agent state.items type: {type(state.items)}")
    print(f"debug_agent state.items value: {state.items}")
    print(f"debug_agent state.count: {state.count}")

    # Try to access items
    try:
        print(f"Items before append: {state.items}")
        state.items.append(f"item_{state.count}")
        print(f"Items after append: {state.items}")
        state.count += 1
        return state
    except Exception as e:
        print(f"Error in debug_agent: {e}")
        print(f"state.items dir: {dir(state.items)[:10]}")
        raise


# Test minimal case
print("=== Creating minimal test ===")
graph = StateGraph[TestState](TestState())
graph.add_node("DEBUG", debug_agent)
graph.set_entry_point("DEBUG")

checkpointer = InMemoryCheckpointer[TestState]()
app = graph.compile(checkpointer=checkpointer)

config = {"thread_id": "debug_test"}
result = app.invoke({"messages": [Message.from_text("Debug test")]}, config=config)

print(f"Final result: {result}")
