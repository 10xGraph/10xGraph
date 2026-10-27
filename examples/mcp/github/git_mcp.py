"""List commits from a GitHub repository through GitHub's remote MCP server.

A ReactAgent gets its tools from the MCP server, so no tool is written by hand.
Contributed example, modernised to use ReactAgent(client=...).

Run: GITHUB_TOKEN=... OPENAI_API_KEY=... python examples/mcp/github/git_mcp.py
"""

import asyncio
import os

from dotenv import load_dotenv
from fastmcp import Client

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()

# GitHub hosts an MCP server over streamable HTTP. The token goes in an
# Authorization header, so it never appears in the prompt.
mcp_config = {
    "mcpServers": {
        "github": {
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"},
            "transport": "streamable-http",
        },
    }
}

agent = ReactAgent(
    model="gpt-5-mini",
    system_prompt=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the GitHub tools to answer questions.",
        }
    ],
    client=Client(mcp_config),  # tools are discovered from the server at run time
    trim_context=True,
)


async def main() -> None:
    app = agent.compile(checkpointer=InMemoryCheckpointer())
    question = (
        "Call the list_commits tool for the repo 'portfolio' owned by 'suchith83' "
        "and give me all commits in that repo."
    )
    result = await app.ainvoke(
        {"messages": [Message.text_message(question)]},
        config={"thread_id": "github-commits", "recursion_limit": 10},
    )
    for message in result["messages"]:
        print(f"[{message.role}]", message.text()[:500])
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
