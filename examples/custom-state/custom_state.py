from typing import Any

from dotenv import load_dotenv

from tenxgraph.core import Agent, StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END


load_dotenv()


class MyState(AgentState):
    """Custom state with additional fields for resume matching."""

    candidate_cv: str = ""
    jd: str = ""
    match_score: float = 0.0
    analysis_results: dict[str, Any] = {}


# Create checkpointer with custom state type
checkpointer = InMemoryCheckpointer[MyState]()


# Create agent instance
def create_main_agent() -> Agent:
    """Create the main agent for CV and job description analysis."""
    return Agent(
        model="gemini-2.5-flash",
        provider="google",
        system_prompt=[
            {
                "role": "system",
                "content": """
                    You are a helpful HR assistant.
                    Your task is to analyze candidate CVs against job descriptions.
                    Please provide a helpful response to the user's query.
                """,
            },
        ],
        trim_context=True,
    )


def should_use_tools(state: AgentState) -> str:
    """Determine if we should use tools or end the conversation."""
    if not state.context or len(state.context) == 0:
        return "TOOL"  # No context, might need tools

    last_message = state.context[-1]

    # If the last message is from assistant and has tool calls, go to TOOL
    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    # If last message is a tool result, we should be done (AI will make final response)
    if last_message.role == "tool":
        return END

    # Default to END for other cases
    return END


def create_app(initial_state: MyState | None = None) -> Any:
    """Create and compile the graph application."""
    state = initial_state or MyState()
    main_agent = create_main_agent()

    graph = StateGraph[MyState](state)
    graph.add_node("MAIN", main_agent)
    graph.set_entry_point("MAIN")

    return graph.compile(checkpointer=checkpointer)


def test_basic_functionality():
    """Test basic functionality with default state."""
    print("=== Testing Basic Functionality ===")
    app = create_app()
    inp = {"messages": [Message.text_message("Hello, can you help me with CV analysis?")]}
    config = {"thread_id": "basic_test", "recursion_limit": 10}

    res = app.invoke(inp, config=config)
    print("Basic test result:", res)
    return res


def test_custom_state_fields():
    """Test with custom state fields populated."""
    print("\n=== Testing Custom State Fields ===")

    # Create a custom state with some data
    custom_state = MyState()
    custom_state.candidate_cv = "John Doe - Software Engineer with 5 years Python experience"
    custom_state.jd = "Looking for Senior Python Developer with 3+ years experience"
    custom_state.match_score = 0.85
    custom_state.analysis_results = {"skills_match": True, "experience_match": True}

    custom_app = create_app(custom_state)

    inp = {"messages": [Message.text_message("What's the match score for this candidate?")]}
    config = {"thread_id": "custom_test", "recursion_limit": 10}

    res = custom_app.invoke(inp, config=config)
    print("Custom state test result:", res)
    return res


def test_partial_state_update():
    """Test that only provided fields in input_data['state'] are updated, others remain unchanged."""
    print("\n=== Testing Partial State Update ===")

    # Initial state with all fields set
    initial_state = MyState()
    initial_state.candidate_cv = "Alice - Data Scientist with 3 years ML experience"
    initial_state.jd = "Looking for Data Scientist with ML background"
    initial_state.match_score = 0.7
    initial_state.analysis_results = {"skills_match": False, "experience_match": True}

    app_partial = create_app(initial_state)

    # Only update 'jd' field via input_data['state']
    partial_update = {"jd": "Looking for Data Scientist with deep learning experience"}
    inp = {
        "messages": [Message.text_message("Update the job description only.")],
        "state": partial_update,
    }
    config = {"thread_id": "partial_update_test", "recursion_limit": 10}

    # Save old values for comparison
    old_cv = initial_state.candidate_cv
    old_score = initial_state.match_score
    old_analysis = initial_state.analysis_results.copy()

    res = app_partial.invoke(inp, config=config, response_granularity=ResponseGranularity.FULL)
    print("Partial state update result:", res)

    # After invoke, check that only 'jd' changed in the returned state
    updated_state = res["state"]
    print("Returned state keys:", list(updated_state.model_dump().keys()))
    print("Returned state dict:", updated_state.model_dump())
    assert hasattr(updated_state, "jd"), f"Returned state missing 'jd': {updated_state}"
    assert updated_state.jd == partial_update["jd"], f"JD should be updated, got {updated_state.jd}"
    assert updated_state.candidate_cv == old_cv, "CV should remain unchanged"
    assert updated_state.match_score == old_score, "Score should remain unchanged"
    assert (
        updated_state.analysis_results == old_analysis
    ), "Analysis results should remain unchanged"
    print("Partial state update test passed!")
    return res


if __name__ == "__main__":
    # Run tests
    try:
        test_basic_functionality()
        test_custom_state_fields()
        test_partial_state_update()
        print("\n=== All tests completed successfully! ===")
    except Exception as e:
        print(f"Error during testing: {e}")
        raise
