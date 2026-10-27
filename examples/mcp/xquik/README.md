---
title: Xquik MCP Server with a 10xGraph Agent
description: Connect a 10xGraph ToolNode or ReactAgent to the remote Xquik MCP server with an API key and list its tools in Python.
section: Examples
group: "Tools and integrations"
order: 52
label: Xquik MCP
updated: "2026-10-28"
faq:
  - question: How does the example authenticate?
    answer: It sends the XQUIK_API_KEY environment variable in the x-api-key header. Xquik also supports OAuth 2.1 for MCP clients that handle it.
  - question: Can I use the tools in an agent?
    answer: Yes. Pass the same client to ReactAgent(client=...), as agent.py does with claude-sonnet-5-5.
---

# Xquik MCP Example

This example connects 10xGraph's `ToolNode` to the remote Xquik MCP server and
loads its tool schemas without changing any framework defaults.

Install 10xGraph's MCP extra, provide an API key through the environment, and
run the example:

```bash
pip install "10xgraph[mcp]"
export XQUIK_API_KEY="your-api-key"
python examples/mcp/xquik/client.py
```

The script prints the available tool names. Use the same `ToolNode` as an
agent's `tool_node` when building a graph. `agent.py` does this with
`ReactAgent(model="claude-sonnet-5-5", client=build_client(...))`; install
`10xgraph[mcp,anthropic]` and set `ANTHROPIC_API_KEY` to run it. See the
[Xquik MCP overview](https://docs.xquik.com/mcp/overview) for authentication,
tool behavior, and safety requirements.

This API-key example sends the `x-api-key` header. Xquik also supports OAuth
2.1 for compatible MCP clients.

Xquik is an independent third-party service. Not affiliated with X Corp.
"Twitter" and "X" are trademarks of X Corp.

See also the [MCP overview](../README.md).
