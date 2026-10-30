"""A ReactAgent that uses tools from two local MCP servers.

Start two servers first (separate terminals):
    python examples/mcp/local/server.py --name us --port 8000
    python examples/mcp/local/server.py --name eu --port 8001
Then: GEMINI_API_KEY=... python examples/mcp/local/agent.py
"""

import asyncio

from dotenv import load_dotenv
from fastmcp import Client

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent


load_dotenv()

# Each server needs its own URL. With more than one server, fastmcp prefixes every tool
# with the server key, so the agent sees `us_get_weather` and `eu_get_weather`
# (a single-server config keeps the plain name).
mcp_config = {
    "mcpServers": {
        "us": {"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"},
        "eu": {"url": "http://127.0.0.1:8001/mcp", "transport": "streamable-http"},
    }
}

agent = ReactAgent(
    model="gemini-2.5-flash",
    system_prompt=[
        {"role": "system", "content": "Answer weather questions with the available tools."}
    ],
    client=Client(mcp_config),
    # Sends config["user"] (or {"user_id": ...}) with every MCP call as request metadata (meta["user"]).
    pass_user_info_to_mcp=True,
)


async def main() -> None:
    app = agent.compile()
    result = await app.ainvoke(
        {"messages": [Message.text_message("Compare the weather in New York and Berlin.")]},
        config={"thread_id": "mcp-local", "user_id": "alice", "recursion_limit": 10},
    )
    print("assistant:", result["messages"][-1].text())
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
