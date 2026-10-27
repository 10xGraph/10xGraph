"""The @tool decorator: description, tags, and filtering tools with tools_tags.

The first half runs without an LLM and prints which tools each tag filter exposes.
The second half runs a DeepSeek agent that can only see the "read" tools.

Run: python examples/tools/tool_decorator.py   (agent half needs DEEPSEEK_API_KEY)
"""

import asyncio
import os

from dotenv import load_dotenv

from tenxgraph.core.graph import ToolNode
from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.utils import get_tool_metadata, tool


load_dotenv()


# Keep name equal to the function name: ToolNode dispatches calls by function name.
@tool(name="get_order", description="Look up an order by id.", tags=["orders", "read"])
def get_order(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped", "total": 59.0}


@tool(name="cancel_order", description="Cancel an order by id.", tags=["orders", "write"])
def cancel_order(order_id: str) -> str:
    return f"Order {order_id} cancelled."


@tool  # bare form: name and description come from the function and its docstring
def current_time() -> str:
    """Return the current server time."""
    return "2026-10-28T12:00:00Z"


async def show_tag_filtering() -> None:
    node = ToolNode([get_order, cancel_order, current_time])
    print("metadata:", get_tool_metadata(cancel_order))
    for tags in (None, {"read"}, {"write"}):
        schemas = await node.all_tools(tags=tags)
        print(f"tags={tags}: {[s['function']['name'] for s in schemas]}")
    # A tool with no tags (current_time) is shown under every filter.


async def run_agent() -> None:
    agent = ReactAgent(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        tools=[get_order, cancel_order, current_time],
        # The model is only shown tools tagged "read" (plus untagged ones). A call to a
        # hidden tool, such as cancel_order, is refused at dispatch.
        tools_tags={"read"},
    )
    async with agent.compile() as app:
        result = await app.ainvoke(
            {"messages": [Message.text_message("Cancel order A-17, then tell me its status.")]},
            config={"thread_id": "tags-1"},
        )
        print("assistant:", result["messages"][-1].text())


async def main() -> None:
    await show_tag_filtering()
    await run_agent()


if __name__ == "__main__":
    asyncio.run(main())
