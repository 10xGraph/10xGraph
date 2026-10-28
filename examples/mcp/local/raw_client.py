"""Talk to an MCP server directly: list its tools and call one, with no LLM involved.

Useful to check a server before giving it to an agent.
Start `python examples/mcp/local/server.py` first, then run this file.
"""

import asyncio

from dotenv import load_dotenv
from fastmcp import Client


load_dotenv()

config = {
    "mcpServers": {
        "weather": {
            "url": "http://127.0.0.1:8000/mcp",
            "transport": "streamable-http",
            # Remote servers usually want a header like this one.
            "headers": {"Authorization": "Bearer TEST_WEATHER_API_KEY"},
        },
    },
}


async def main() -> None:
    async with Client(config) as client:
        for tool in await client.list_tools():
            tags = (tool.meta or {}).get("_fastmcp", {}).get("tags", [])
            print(f"tool: {tool.name} tags={tags}")
            print("  schema:", tool.inputSchema)

        result = await client.call_tool("get_weather", {"location": "New York"})
        print("data:", result.data)


if __name__ == "__main__":
    asyncio.run(main())
