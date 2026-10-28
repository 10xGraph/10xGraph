"""Client-side tools: the model calls a tool that the client executes, not the server.

The graph pauses after the tool node and hands back the call. The client runs it
(here a stand-in for a browser or app function) and resumes the same thread with a
ToolResultBlock. A checkpointer is required so the paused thread is saved.

Run: python examples/tools/remote_tools.py   (needs DEEPSEEK_API_KEY)
"""

import asyncio
import json
import os

from dotenv import load_dotenv

from tenxgraph.core.state import Message, ToolResultBlock
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()


def server_time() -> str:
    """Return the server time."""
    return "2026-10-28T12:00:00Z"


# ReactAgent needs at least one server tool to create its TOOL node.
agent = ReactAgent(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    tools=[server_time],
)
app = agent.compile(checkpointer=InMemoryCheckpointer())

CLIENT_TOOL = {
    "name": "read_clipboard",
    "description": "Read the text currently in the user's clipboard.",
    "parameters": {"type": "object", "properties": {}},
}

# Option 1: attach to the compiled graph for every run. "node" is the ToolNode name.
app.attach_remote_tools([{**CLIENT_TOOL, "node": "TOOL"}])

# Option 2: offer tools for one run only, through the run config:
#   config = {"thread_id": "...", "remote_tools": [CLIENT_TOOL]}
# This leaves the shared graph unchanged, which suits per-user frontend tools.


def run_on_client(name: str, args: dict) -> str:
    # In a real app this is JavaScript in the browser or code in a desktop app.
    return "Meeting moved to 15:00"


async def main() -> None:
    config = {"thread_id": "remote-1"}
    async with app:
        first = await app.ainvoke(
            {"messages": [Message.text_message("Summarise what is in my clipboard.")]}, config
        )

        # The run stopped before answering. The assistant message holds the tool call.
        call = next(
            m.tools_calls[0]
            for m in reversed(first["messages"])
            if m.role == "assistant" and m.tools_calls
        )
        name, args = call["function"]["name"], json.loads(call["function"]["arguments"] or "{}")
        print(f"paused: model wants the client to run {name}({args})")

        output = run_on_client(name, args)

        # Resume the same thread with the result keyed by the tool call id.
        result_msg = Message(
            role="tool", content=[ToolResultBlock(call_id=call["id"], output=output)]
        )
        final = await app.ainvoke({"messages": [result_msg]}, config)
        print("assistant:", final["messages"][-1].text())


if __name__ == "__main__":
    asyncio.run(main())
