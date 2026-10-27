"""Give a ReactAgent the Xquik MCP tools.

Reuses build_client() from client.py. Run from the repo root:
XQUIK_API_KEY=... ANTHROPIC_API_KEY=... python examples/mcp/xquik/agent.py
"""

import asyncio

from client import build_client, require_api_key
from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent


load_dotenv()


async def main() -> None:
    agent = ReactAgent(
        model="claude-sonnet-5-5",
        system_prompt=[{"role": "system", "content": "Use the Xquik tools to answer."}],
        client=build_client(require_api_key()),
    )
    app = agent.compile()
    result = await app.ainvoke(
        {"messages": [Message.text_message("Which Xquik tools are available to you?")]},
        config={"thread_id": "xquik-demo", "recursion_limit": 10},
    )
    print("assistant:", result["messages"][-1].text())
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
