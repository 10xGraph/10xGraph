---
title: GitHub MCP Server with a 10xGraph Agent
description: Use GitHub's remote MCP server from a 10xGraph ReactAgent to list commits and download a README, authenticated with a GitHub token in Python.
section: Examples
group: "Tools and integrations"
order: 51
label: GitHub MCP
updated: "2026-10-28"
faq:
  - question: Where does the GitHub token go?
    answer: In the headers of the server entry in the mcpServers config, as an Authorization bearer token read from GITHUB_TOKEN. It is never part of the prompt.
  - question: Do I write the GitHub tools myself?
    answer: No. ReactAgent(client=...) discovers the tools the GitHub MCP server exposes, such as list_commits, at run time.
---

## What you will build

Two small scripts, contributed by the community, that point a ReactAgent at GitHub's hosted MCP server. `git_mcp.py` asks for the commits of a repository and `mcp_file_download.py` reads its `README.md`. The agent uses `gpt-5-mini`.

## Key concepts

- `fastmcp.Client` with a remote `streamable-http` server and an `Authorization` header.
- `ReactAgent(client=...)` to expose the server's tools to the model.
- `InMemoryCheckpointer` to keep the thread state between steps.

## Prerequisites

```bash
pip install "10xgraph[mcp,openai]" python-dotenv
export OPENAI_API_KEY=...
export GITHUB_TOKEN=...   # a personal access token
```

## Run it

```bash
python examples/mcp/github/git_mcp.py
python examples/mcp/github/mcp_file_download.py
```

## How it works

Both files build the same agent. The only difference is the question:

```python
agent = ReactAgent(model="gpt-5-mini", client=Client(mcp_config), trim_context=True)
app = agent.compile(checkpointer=InMemoryCheckpointer())
result = await app.ainvoke({"messages": [Message.text_message(question)]}, config={...})
```

The model decides which GitHub tool to call, `ReactAgent` runs it through the MCP client, and the loop ends when the model answers in text.

## Expected output

Illustrative:

```text
[user] Call the list_commits tool for the repo 'portfolio' ...
[assistant]
[tool] [{"sha": "3f2a...", "commit": {"message": "Add contact form"}}, ...]
[assistant] The repo has 12 commits. The latest is "Add contact form".
```

## Using a different model

Change `gpt-5-mini` to `gemini-2.5-flash` (`pip install "10xgraph[mcp,google-genai]"`) or `claude-sonnet-5-5` (`pip install "10xgraph[mcp,anthropic]"`) and set that provider's key.

## Next steps

- [MCP overview](../README.md)
- [`../local/`](../local/) to run your own MCP server.
- [`../../providers/`](../../providers/) for other models.
