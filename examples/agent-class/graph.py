import os

from dotenv import load_dotenv

from tenxgraph.core.graph import Agent, StateGraph, ToolNode
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.message import Message
from tenxgraph.utils.constants import END


load_dotenv()


def get_weather(
    location: str,
) -> str:
    """
    Get the current weather for a specific location.
    This demo shows injectable parameters: tool_call_id and state are automatically injected.
    """
    return f"The weather in {location} is sunny"


tool_node = ToolNode([get_weather])


graph = StateGraph()
graph.add_node(
    "MAIN",
    Agent(
        model="google/gemini-2.5-flash",
        system_prompt=[
            {
                "role": "system",
                "content": "You are a helpful assistant, Help user queries effectively.",
            }
        ],
        tool_node="TOOL",
    ),
)
graph.add_node("TOOL", tool_node)


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
        return "MAIN"

    # Default to END for other cases
    return END


# Add conditional edges from MAIN
graph.add_conditional_edges(
    "MAIN",
    should_use_tools,
    {"TOOL": "TOOL", END: END},
)

# Always go back to MAIN after TOOL execution
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile()


if __name__ == "__main__":
    inp = {"messages": [Message.text_message("How is weather in London?")]}
    config = {"thread_id": "12345", "recursion_limit": 10}

    res = app.invoke(inp, config=config)

    for i in res["messages"]:
        print("**********************")
        print("Message Type: ", i.role)
        print(i)
        print("**********************")
        print("\n\n")
