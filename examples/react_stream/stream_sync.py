import time

from dotenv import load_dotenv

from tenxgraph.core import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.stream_emitter import StreamEmitter
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END


load_dotenv()

checkpointer = InMemoryCheckpointer()


class CustomAgentState(AgentState):
    jd_text: str = ""
    cv_text: str = ""


def get_weather(
    location: str,
    tool_call_id: str | None = None,
    state: AgentState | None = None,
    emit: StreamEmitter | None = None,
) -> str:
    """
    Get the current weather for a specific location.
    This demo shows injectable parameters: tool_call_id and state are automatically injected.
    """
    # You can access injected parameters here

    if emit:
        emit.progress("Fetching weather data...", data={"location": location})

    time.sleep(1)  # Simulate a delay in fetching weather data

    if emit:
        emit.progress("Processing weather data...", data={"location": location})

    if tool_call_id:
        print(f"Tool call ID: {tool_call_id}")
    if state and hasattr(state, "context"):
        print(f"Number of messages in context: {len(state.context)}")  # type: ignore

    if emit:
        emit.progress("Finalizing response...", data={"location": location})

    return f"The weather in {location} is sunny"


tool_node = ToolNode([get_weather])


main_agent = Agent(
    model="gemini-2.5-flash",
    provider="google",
    system_prompt=[
        {
            "role": "system",
            "content": """
                You are a helpful assistant with access to a get_weather tool.
                When asked about weather for any location you MUST call the get_weather tool.
                Do not answer weather questions from your own knowledge.
            """,
        },
        {"role": "user", "content": "Today Date is 2024-06-15"},
    ],
    tool_node=tool_node,
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
        return "MAIN"

    # Default to END for other cases
    return END


graph = StateGraph(CustomAgentState())
graph.add_node("MAIN", main_agent)
graph.add_node("TOOL", tool_node)

# Add conditional edges from MAIN
graph.add_conditional_edges(
    "MAIN",
    should_use_tools,
    {"TOOL": "TOOL", END: END},
)

# Always go back to MAIN after TOOL execution
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")


app = graph.compile(
    checkpointer=checkpointer,
)


# now run it
inp = {"messages": [Message.text_message("Please call the get_weather function for Paris")]}
config = {"thread_id": "12345", "recursion_limit": 10, "is_stream": True}

res = app.stream(inp, config=config)
print("Streaming response:")

message_count = 0
for i in res:
    message_count += 1
    print(i.model_dump())
    print("\n\n")

print(f"Total messages received: {message_count}")
