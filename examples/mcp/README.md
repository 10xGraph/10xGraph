---
title: MCP Tools in a Python AI Agent with 10xGraph
description: Connect a 10xGraph ReactAgent to local and remote MCP servers over streamable HTTP, use several servers at once, and send auth headers in Python.
section: Examples
group: "Tools and integrations"
order: 50
label: MCP
updated: "2026-10-28"
faq:
  - question: How do I give a 10xGraph agent tools from an MCP server?
    answer: Build a fastmcp Client from an mcpServers config and pass it as ReactAgent(client=...). The agent discovers the server's tools at run time, so you do not write tool functions.
  - question: What happens to tool names when I connect more than one MCP server?
    answer: fastmcp prefixes each tool with its server key, for example us_get_weather and eu_get_weather. A config with a single server keeps the original tool name.
  - question: How do I authenticate to a remote MCP server?
    answer: Add a headers entry to that server in the mcpServers config, such as an Authorization bearer token read from an environment variable.
  - question: How does the MCP server know which user made the call?
    answer: Set pass_user_info_to_mcp=True on ReactAgent or ToolNode. The ToolNode adds config["user"], or {"user_id": ...} when only user_id is set, as request metadata (`meta["user"]`) on each MCP tool call.
---

## What you will build

A ReactAgent whose tools come from MCP (Model Context Protocol) servers instead of Python functions. `local/` runs a FastMCP weather server on your machine and shows an agent that uses two of them at once. `github/` and `xquik/` connect to hosted servers that need a token.

All of them share one pattern: describe the servers in an `mcpServers` dict, wrap it in a fastmcp `Client`, and pass the client to `ReactAgent`.

## Key concepts

- `fastmcp.Client(config)`: the MCP client. The config lists servers by name with `url`, `transport` and optional `headers`.
- `ReactAgent(client=...)`: builds a `ToolNode` around the client and wires the model and tool loop.
- `ToolNode.all_tools()`: lists the tool schemas the client discovered.
- `pass_user_info_to_mcp`: forwards the caller identity to the server as request metadata (`meta["user"]`).
- Tool name prefixes: with several servers, each tool is named `<server>_<tool>`.

## Prerequisites

```bash
pip install "10xgraph[mcp,google-genai]" python-dotenv
export GEMINI_API_KEY=...        # local/ examples
export OPENAI_API_KEY=...        # github/ examples (plus GITHUB_TOKEN)
```

## Run it

Start the weather server in one terminal, then the agent in another:

```bash
python examples/mcp/local/server.py
python examples/mcp/local/raw_client.py   # check the server, no LLM
```

For the two-server agent:

```bash
python examples/mcp/local/server.py --name us --port 8000
python examples/mcp/local/server.py --name eu --port 8001
python examples/mcp/local/agent.py
```

## How it works

`local/server.py` builds a FastMCP server and serves it over streamable HTTP:

```python
@mcp.tool(description="Get the current weather for a location")
def get_weather(location: str, ctx: Context) -> dict: ...  # reads ctx.request_context.meta

build_server(args.name).run(transport="streamable-http", host="127.0.0.1", port=args.port)
```

`ctx: Context` is injected by FastMCP and is not in the schema the model sees. The agent sends the caller as request metadata, and the tool reads `meta.model_extra["user"]`.

`local/raw_client.py` talks to the server without an LLM: `client.list_tools()` and `client.call_tool("get_weather", {...})`. Run it first when a server misbehaves.

`local/agent.py` lists two servers under different keys and URLs:

```python
mcp_config = {
    "mcpServers": {
        "us": {"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"},
        "eu": {"url": "http://127.0.0.1:8001/mcp", "transport": "streamable-http"},
    }
}
agent = ReactAgent(model="gemini-2.5-flash", client=Client(mcp_config), pass_user_info_to_mcp=True)
```

The model sees `us_get_weather` and `eu_get_weather`. Pointing two keys at the same URL, as an older version of this example did, only duplicates tools.

For a remote server with authentication, put the credential in `headers`, as `github/` does:

```python
"github": {
    "url": "https://api.githubcopilot.com/mcp/",
    "headers": {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
    "transport": "streamable-http",
}
```

Contributed examples for hosted servers:

- [`github/`](github/): list commits and read files from GitHub's MCP server with `gpt-5-mini`.
- [`xquik/`](xquik/): list and use the tools of the Xquik MCP server.

## Expected output

Illustrative, from `agent.py`:

```text
[us] weather for New York, caller: {'user_id': 'alice'}
[eu] weather for Berlin, caller: {'user_id': 'alice'}
assistant: New York is 22C and sunny. Berlin is 22C and sunny.
```

## Using a different model

The model is a string in the code. Change `gemini-2.5-flash` to `gpt-5-mini` (`pip install "10xgraph[mcp,openai]"`) or `claude-sonnet-5-5` (`pip install "10xgraph[mcp,anthropic]"`) and set the matching API key. The MCP setup does not change.

## Next steps

- [`../tools/`](../tools/) for tools written as Python functions.
- [`../providers/`](../providers/) for the model options.
- [`../dependency_injection/`](../dependency_injection/) for passing your own services to tools.
