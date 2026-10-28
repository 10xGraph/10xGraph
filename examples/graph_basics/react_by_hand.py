"""The ReAct loop wired by hand with Agent + ToolNode + a conditional edge.

ReactAgent builds exactly this graph for you. Read this file to see what is inside,
or copy it when you need a different shape (extra nodes, custom routing).

Run: python examples/graph_basics/react_by_hand.py   (needs GEMINI_API_KEY)
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.core.graph import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils.constants import END


load_dotenv()


def get_weather(location: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {location} is sunny, 24 C."


tool_node = ToolNode([get_weather])

# Agent is the LLM node. tool_node="TOOL" refers to the graph node registered below,
# resolved at compile(); passing the ToolNode object itself also works.
main_agent = Agent(
    model="gemini-2.5-flash",
    system_prompt=[{"role": "system", "content": "You are a concise weather assistant."}],
    tool_node="TOOL",
)


def route(state: AgentState) -> str:
    """Go to TOOL when the last assistant message asked for tools, otherwise stop."""
    last = state.context[-1] if state.context else None
    if last is not None and last.role == "assistant" and last.tools_calls:
        return "TOOL"
    return END


graph = StateGraph()
graph.add_node("MAIN", main_agent)
graph.add_node("TOOL", tool_node)

graph.set_entry_point("MAIN")
graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
# The loop: tool results go back to the model, which either calls more tools or answers.
graph.add_edge("TOOL", "MAIN")

app = graph.compile()


async def main() -> None:
    async with app:
        result = await app.ainvoke(
            {"messages": [Message.text_message("What is the weather in Berlin?")]},
            config={"thread_id": "by-hand-1", "recursion_limit": 10},
        )
        for message in result["messages"]:
            print(f"{message.role}:", message.text() or message.tools_calls)


if __name__ == "__main__":
    asyncio.run(main())
