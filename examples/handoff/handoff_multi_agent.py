"""
Multi-agent handoff example demonstrating agent-to-agent transfers.

This example shows how to create a multi-agent system where agents can
transfer control to each other using handoff tools. Similar to react_sync.py
but with multiple specialized agents that hand off work between each other.
"""

from dotenv import load_dotenv

from tenxgraph.core import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.prebuilt.tools import create_handoff_tool
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END


load_dotenv()

checkpointer = InMemoryCheckpointer()


# ============================================================================
# Define Regular Tools
# ============================================================================


def get_weather(
    location: str,
    tool_call_id: str | None = None,
    state: AgentState | None = None,
) -> str:
    """
    Get the current weather for a specific location.
    This demo shows injectable parameters: tool_call_id and state are automatically injected.
    """
    if tool_call_id:
        print(f"Tool call ID: {tool_call_id}")
    if state and hasattr(state, "context"):
        print(f"Number of messages in context: {len(state.context)}")  # type: ignore

    return f"The weather in {location} is sunny, 25°C"


def search_web(
    query: str,
    tool_call_id: str | None = None,
) -> str:
    """
    Search the web for information.
    """
    print(f"Searching web for: {query}")
    return f"Search results for '{query}': Found relevant information about quantum computing"


def write_document(
    content: str,
    title: str,
    tool_call_id: str | None = None,
) -> str:
    """
    Write a document with the given content and title.
    """
    print(f"Writing document: {title}")
    return f"Document '{title}' written successfully with {len(content)} characters"


# ============================================================================
# Create Agent-Specific Tool Nodes
# ============================================================================

# Coordinator has access to handoff tools for delegation
coordinator_tools = ToolNode(
    [
        create_handoff_tool(
            "researcher", "Transfer to research specialist for detailed investigation"
        ),
        create_handoff_tool("writer", "Transfer to writing specialist for content creation"),
        get_weather,  # Also has regular tools
    ]
)

# Researcher has access to search and can handoff to writer or back to coordinator
researcher_tools = ToolNode(
    [
        search_web,
        create_handoff_tool("coordinator", "Transfer back to coordinator for delegation"),
        create_handoff_tool("writer", "Transfer to writer with research findings"),
    ]
)

# Writer has document writing capability and can handoff back to coordinator
writer_tools = ToolNode(
    [
        write_document,
        create_handoff_tool("coordinator", "Transfer back to coordinator"),
    ]
)


# ============================================================================
# Create Agents using Agent class
# ============================================================================

coordinator_agent = Agent(
    model="gemini-2.5-flash",
    provider="google",
    system_prompt=[
        {
            "role": "system",
            "content": """
                You are a coordinator agent. Your job is to:
                1. Understand user requests
                2. Delegate tasks to specialized agents:
                   - Use transfer_to_researcher for investigation and research tasks
                   - Use transfer_to_writer for content creation and writing tasks
                3. You can also check weather using get_weather tool

                Always explain your decision to delegate and why you're choosing a specific agent.
            """,
        },
    ],
    tool_node="COORDINATOR_TOOLS",
    trim_context=True,
)

researcher_agent = Agent(
    model="gemini-2.5-flash",
    provider="google",
    system_prompt=[
        {
            "role": "system",
            "content": """
                You are a research specialist. Your job is to:
                1. Investigate topics thoroughly using the search_web tool
                2. Gather comprehensive information
                3. When research is complete, you can:
                   - Transfer to writer agent if content needs to be created
                   - Transfer back to coordinator if task is complete
                Be thorough in your research and explain your findings clearly.
            """,
        },
    ],
    tool_node="RESEARCHER_TOOLS",
    trim_context=True,
)

writer_agent = Agent(
    model="gemini-2.5-flash",
    provider="google",
    system_prompt=[
        {
            "role": "system",
            "content": """
                You are a writing specialist. Your job is to:
                1. Create clear, engaging content
                2. Use the write_document tool to save content
                3. Transfer back to coordinator when writing is complete

                Focus on clarity, structure, and engaging writing.
            """,
        },
    ],
    tool_node="WRITER_TOOLS",
    trim_context=True,
)


# ============================================================================
# Define Routing Logic
# ============================================================================


def should_continue_coordinator(state: AgentState) -> str:
    """Route from coordinator to tools or end."""
    if not state.context or len(state.context) == 0:
        return "COORDINATOR_TOOLS"

    last_message = state.context[-1]

    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "COORDINATOR_TOOLS"

    if last_message.role == "tool":
        return "COORDINATOR"

    return END


def should_continue_researcher(state: AgentState) -> str:
    """Route from researcher to tools or end."""
    if not state.context or len(state.context) == 0:
        return "RESEARCHER_TOOLS"

    last_message = state.context[-1]

    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "RESEARCHER_TOOLS"

    if last_message.role == "tool":
        return "RESEARCHER"

    return END


def should_continue_writer(state: AgentState) -> str:
    """Route from writer to tools or end."""
    if not state.context or len(state.context) == 0:
        return "WRITER_TOOLS"

    last_message = state.context[-1]

    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "WRITER_TOOLS"

    if last_message.role == "tool":
        return "WRITER"

    return END


# ============================================================================
# Build Graph
# ============================================================================

graph = StateGraph()

# Add agent nodes
graph.add_node("COORDINATOR", coordinator_agent)
graph.add_node("COORDINATOR_TOOLS", coordinator_tools)
graph.add_node("RESEARCHER", researcher_agent)
graph.add_node("RESEARCHER_TOOLS", researcher_tools)
graph.add_node("WRITER", writer_agent)
graph.add_node("WRITER_TOOLS", writer_tools)

# Set entry point
graph.set_entry_point("COORDINATOR")

# Add edges for coordinator
graph.add_conditional_edges(
    "COORDINATOR",
    should_continue_coordinator,
    {
        "COORDINATOR_TOOLS": "COORDINATOR_TOOLS",
        END: END,
    },
)

# Add edges for researcher
graph.add_conditional_edges(
    "RESEARCHER",
    should_continue_researcher,
    {
        "RESEARCHER_TOOLS": "RESEARCHER_TOOLS",
        END: END,
    },
)

# Add edges for writer
graph.add_conditional_edges(
    "WRITER",
    should_continue_writer,
    {
        "WRITER_TOOLS": "WRITER_TOOLS",
        END: END,
    },
)

# Tool nodes will automatically navigate to target agents via handoff detection
# The handoff tools will return Command(goto=target_agent) which the graph handles
# No need to add explicit edges from tool nodes back to agents!

# Compile the graph
app = graph.compile(checkpointer=checkpointer)


# ============================================================================
# Run Example
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MULTI-AGENT HANDOFF EXAMPLE")
    print("=" * 80 + "\n")

    # Test case: Research and write about quantum computing
    inp = {
        "messages": [
            Message.text_message(
                "Please research quantum computing and write a brief article about it."
            )
        ]
    }
    config = {"thread_id": "handoff-demo-001", "recursion_limit": 15}

    print("User Request:")
    print("  'Please research quantum computing and write a brief article about it.'\n")
    print("Expected Flow:")
    print("  1. Coordinator → delegates to Researcher")
    print("  2. Researcher → searches web, then transfers to Writer")
    print("  3. Writer → creates document, transfers back to Coordinator")
    print("  4. Coordinator → provides final summary\n")
    print("=" * 80 + "\n")

    res = app.invoke(inp, config=config)

    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80 + "\n")

    print("Message History:")
    print("-" * 80)
    for i, msg in enumerate(res["messages"], 1):
        print(f"\n[{i}] Message Type: {msg.role}")
        print(f"    Content: {msg.text()[:200]}...")
        if hasattr(msg, "tools_calls") and msg.tools_calls:
            for tool_call in msg.tools_calls:
                tool_name = tool_call.get("function", {}).get("name", "")
                print(f"    Tool Called: {tool_name}")

    print("\n" + "=" * 80)
    print("END OF EXAMPLE")
    print("=" * 80 + "\n")
