#!/usr/bin/env python3

from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.execution_state import ExecutionState
from tenxgraph.utils import START


def test_execution_state():
    print("=== Testing ExecutionState ===")
    exec_state = ExecutionState(current_node=START)
    print(f"Created ExecutionState: {exec_state}")
    print(f"Status: {exec_state.status}")
    print(f"Current node: {exec_state.current_node}")

    # Test serialization
    state_dict = exec_state.model_dump()
    print(f"Serialized keys: {list(state_dict.keys())}")

    # Test deserialization
    restored_state = ExecutionState.from_dict(state_dict)
    print(f"Restored current_node: {restored_state.current_node}")
    print(f"Restored status: {restored_state.status}")


def test_agent_state():
    print("\n=== Testing AgentState ===")
    agent_state = AgentState()
    print(f"Created AgentState: type={type(agent_state)}")
    print(f"Context length: {len(agent_state.context)}")
    print(f"Execution meta current node: {agent_state.execution_meta.current_node}")

    # Test methods
    agent_state.advance_step()
    print(f"After advance_step: step={agent_state.execution_meta.step}")

    agent_state.set_current_node("test_node")
    print(f"After set_current_node: {agent_state.execution_meta.current_node}")

    # Test serialization
    state_dict = agent_state.model_dump()
    print(f"State dict keys: {list(state_dict.keys())}")


def test_pydantic_features():
    print("\n=== Testing Pydantic Features ===")

    # Test model_dump()
    exec_state = ExecutionState(current_node=START, step=5)
    model_data = exec_state.model_dump()
    print(f"model_dump() keys: {list(model_data.keys())}")

    # Test model_validate()
    new_state = ExecutionState.model_validate({"current_node": "test", "step": 10})
    print(f"model_validate() created state with node: {new_state.current_node}")

    # Test AgentState Pydantic features
    agent_state = AgentState()
    agent_model_data = agent_state.model_dump()
    print(f"AgentState model_dump() keys: {list(agent_model_data.keys())}")


if __name__ == "__main__":
    try:
        test_execution_state()
        test_agent_state()
        test_pydantic_features()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
