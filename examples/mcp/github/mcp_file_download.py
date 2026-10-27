"""Fetch a file (a repository README) through GitHub's remote MCP server.

Same setup as git_mcp.py with a different question: the agent picks the right
MCP tool on its own.

Run: GITHUB_TOKEN=... OPENAI_API_KEY=... python examples/mcp/github/mcp_file_download.py
"""

import asyncio
import os

from dotenv import load_dotenv
from fastmcp import Client

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()

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
    client=Client(mcp_config),
    trim_context=True,
)


async def main() -> None:
    app = agent.compile(checkpointer=InMemoryCheckpointer())
    question = "Get the README.md file from the repo 'portfolio' owned by 'suchith83'."
    result = await app.ainvoke(
        {"messages": [Message.text_message(question)]},
        config={"thread_id": "github-readme", "recursion_limit": 10},
    )
    print("assistant:", result["messages"][-1].text())
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
