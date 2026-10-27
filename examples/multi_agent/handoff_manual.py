"""Manual handoff: wire agent-to-agent transfers on your own StateGraph with create_handoff_tool.

Use this when SwarmAgent's wiring is not what you need. Each agent gets a ToolNode that
holds its regular tools and transfer tools. When the model calls `transfer_to_<node>`,
the tool node returns a Command that jumps to that node without running the tool.
Agents: coordinator on Gemini, researcher on DeepSeek, writer on Anthropic.

Run: python examples/multi_agent/handoff_manual.py
     (needs GEMINI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY)
"""

import asyncio
import os

from dotenv import load_dotenv

from tenxgraph.core.graph import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.prebuilt.tools import create_handoff_tool
from tenxgraph.utils.constants import END


load_dotenv()


def search_web(query: str) -> str:
    """Search the web for information."""
    return (
        f"Results for '{query}': quantum computers use qubits; error correction is the main hurdle."
    )


def write_document(title: str, content: str) -> str:
    """Save a document."""
    return f"Document '{title}' saved ({len(content)} characters)."


# The target name in create_handoff_tool must equal a node name in the graph. It is used as
# written, so these nodes are lowercase on purpose.
coordinator = Agent(
    model="gemini-2.5-flash",  # Google
    system_prompt=[
        {
            "role": "system",
            "content": "You coordinate. Send research to the researcher and writing to the writer. "
            "When the writer returns, give the user a one-line wrap-up.",
        }
    ],
    tool_node="coordinator_tools",
)
researcher = Agent(
    model="deepseek-chat",  # DeepSeek
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    system_prompt=[
        {"role": "system", "content": "Research with search_web, then transfer to the writer."}
    ],
    tool_node="researcher_tools",
)
writer = Agent(
    model="claude-sonnet-5-5",  # Anthropic
    system_prompt=[
        {
            "role": "system",
            "content": "Write a short article, save it with write_document, "
            "then transfer back to the coordinator.",
        }
    ],
    tool_node="writer_tools",
)


def route_to(tool_node: str):
    """Go to the agent's tool node when the last reply has tool calls, else finish."""

    def route(state: AgentState) -> str:
        last = state.context[-1] if state.context else None
        return tool_node if last and last.role == "assistant" and last.tools_calls else END

    return route


def build_graph() -> StateGraph:
    graph = StateGraph()
    agents = {
        "coordinator": (
            coordinator,
            [
                create_handoff_tool("researcher", "Hand off a research task."),
                create_handoff_tool("writer", "Hand off a writing task."),
            ],
        ),
        "researcher": (
            researcher,
            [search_web, create_handoff_tool("writer", "Hand off findings to the writer.")],
        ),
        "writer": (
            writer,
            [write_document, create_handoff_tool("coordinator", "Return to the coordinator.")],
        ),
    }
    for name, (agent, tools) in agents.items():
        graph.add_node(name, agent)
        graph.add_node(f"{name}_tools", ToolNode(tools))
        graph.add_conditional_edges(
            name, route_to(f"{name}_tools"), {f"{name}_tools": f"{name}_tools", END: END}
        )
        # After a regular tool, return to the same agent. A transfer tool skips this edge
        # because its Command already names the next node.
        graph.add_edge(f"{name}_tools", name)

    graph.set_entry_point("coordinator")
    return graph


async def main() -> None:
    app = build_graph().compile()
    result = await app.ainvoke(
        {
            "messages": [
                Message.text_message("Research quantum computing and write a short article.")
            ]
        },
        config={"thread_id": "handoff-1", "recursion_limit": 30},
    )
    for msg in result["messages"]:
        if msg.role == "assistant" and msg.text():
            model = (msg.metadata or {}).get("model") or "unknown model"
            print(f"[{model}] {msg.text()[:300]}\n")
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
