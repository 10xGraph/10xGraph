"""A small FastMCP weather server over streamable HTTP.

Run it in its own terminal first:
    python examples/mcp/local/server.py                      # port 8000
    python examples/mcp/local/server.py --port 8001 --name eu  # a second server
"""

import argparse

from fastmcp import Context, FastMCP


def build_server(name: str) -> FastMCP:
    mcp = FastMCP(f"{name} weather server")

    # With pass_user_info_to_mcp=True (see agent.py) the ToolNode sends the caller as request
    # metadata. A `Context` parameter is injected by FastMCP and never shows in the tool
    # schema the LLM sees, so the model cannot invent or change the caller.
    @mcp.tool(
        description="Get the current weather for a location",
        tags={"weather", "information"},
    )
    def get_weather(location: str, ctx: Context) -> dict:
        # ToolNode sends the caller as request metadata under the key "user".
        meta = ctx.request_context.meta
        user = (meta.model_extra or {}).get("user") if meta else None
        print(f"[{name}] weather for {location}, caller: {user}")
        return {"location": location, "temperature": "22C", "description": f"Sunny ({name})"}

    @mcp.tool(description="Get a 3-day forecast for a location")
    def get_forecast(location: str) -> list[dict]:
        return [{"day": d, "temperature": f"{20 + d}C", "description": "Clear"} for d in range(3)]

    return mcp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="us")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    build_server(args.name).run(transport="streamable-http", host="127.0.0.1", port=args.port)
