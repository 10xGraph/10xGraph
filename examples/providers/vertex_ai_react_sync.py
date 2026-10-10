"""Vertex AI provider example using a ReAct agent with tools.

Prerequisites:
    1. A GCP project with the Vertex AI API enabled.
    2. A service account JSON key file.
    3. Environment variables (set in .env or shell):
        GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
        GOOGLE_CLOUD_LOCATION=us-central1          # optional, defaults to us-central1
        GOOGLE_APPLICATION_CREDENTIALS=./path/to/service_account.json
"""

from dotenv import load_dotenv

from tenxgraph.core import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END

load_dotenv()

checkpointer = InMemoryCheckpointer()


def get_weather(location: str) -> str:
    """Get the current weather for a specific location."""
    return f"The weather in {location} is sunny"


tool_node = ToolNode([get_weather])

agent = Agent(
    model="gemini-2.5-flash",
    provider="vertex_ai",
    system_prompt=[{"role": "system", "content": "You are a helpful assistant."}],
    trim_context=True,
    reasoning_config=True,
    tool_node=tool_node,
)


def should_use_tools(state: AgentState) -> str:
    """Determine if we should use tools or end the conversation."""
    if not state.context or len(state.context) == 0:
        return "TOOL"

    last_message = state.context[-1]

    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    if last_message.role == "tool":
        return "MAIN"

    return END


graph = StateGraph()
graph.add_node("MAIN", agent)
graph.add_node("TOOL", tool_node)
graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    inp = {"messages": [Message.text_message("What is the weather in New York City?")]}
    config = {"thread_id": "12345", "recursion_limit": 10}

    res = app.invoke(inp, config=config)

    for msg in res["messages"]:
        print(f"[{msg.role}] {msg}")
