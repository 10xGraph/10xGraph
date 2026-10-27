---
title: Tools in 10xGraph - @tool Decorator, Injected Params, ToolResult and Remote Tools
description: Define Python agent tools in 10xGraph: the @tool decorator with tags, injected state and config, ToolResult state updates, prebuilt tools and client-side remote tools.
section: Examples
group: "Tools and integrations"
order: 40
label: Tools
updated: "2026-10-28"
faq:
  - question: How does the model learn what my tool does?
    answer: ToolNode builds an OpenAI-style function schema from the function name, type hints and docstring. The @tool decorator lets you set the description and tags explicitly.
  - question: Which parameters are injected instead of chosen by the model?
    answer: Parameters named state, config, tool_call_id and emit, plus dependencies injected through InjectQ. They are removed from the schema the model sees.
  - question: How do I let the model call a function that runs in the browser?
    answer: Register it as a remote tool with attach_remote_tools or config["remote_tools"]. The graph pauses after the tool node and hands the call to the client, which resumes the thread with a ToolResultBlock.
---

## What you will build

Five short scripts, one per tool feature. They use DeepSeek (`deepseek-chat`) through its OpenAI-compatible endpoint, which shows that any such model works by passing `base_url` and `api_key`.

## Key concepts

- `@tool(name, description, tags)`: attaches metadata to a function.
- `Agent(tools_tags=...)` / `ReactAgent(tools_tags=...)`: show the model only tools matching a tag. Untagged tools are always shown.
- Injected parameters: `state`, `config`, `tool_call_id` are filled by `ToolNode`.
- `ToolResult(message, state)`: returns text to the model and writes fields into state.
- `safe_calculator`, `fetch_url`, `create_file_tools(root)`: prebuilt tools in `tenxgraph.prebuilt.tools`.
- `attach_remote_tools` / `config["remote_tools"]`: tools the client executes.

## Prerequisites

```bash
pip install "10xgraph[openai]" python-dotenv
export DEEPSEEK_API_KEY=...
```

`tool_decorator.py` prints its tag-filtering demo before any API call, so that part works without a key.

## Run it

```bash
python examples/tools/tool_decorator.py
python examples/tools/injected_params.py
python examples/tools/tool_result_state.py
python examples/tools/prebuilt_tools.py
python examples/tools/remote_tools.py
```

## How it works

The agents in this folder are built like this:

```python
ReactAgent(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    tools=[...],
)
```

`ReactAgent` forwards extra keyword arguments to its `Agent`, which uses the OpenAI client for unknown model names.

### tool_decorator.py

```python
@tool(name="get_order", description="Look up an order by id.", tags=["orders", "read"])
def get_order(order_id: str) -> dict: ...
```

`ToolNode.all_tools(tags={"read"})` returns `get_order` and any untagged tools, not `cancel_order`. The agent uses `tools_tags={"read"}`, and a call to a hidden tool is refused. Keep the decorator `name` equal to the function name: `ToolNode` looks tools up by `__name__` when dispatching.

### injected_params.py

```python
def whoami(topic: str, state: AgentState, config: dict[str, Any], tool_call_id: str) -> str:
```

Only `topic` appears in the schema. Extra keys you put in the run config, such as `plan`, are visible in `config`.

### tool_result_state.py

```python
return ToolResult(message="Added keyboard...", state={"cart": cart, "total": total})
```

Keys in `state` must be fields on your `AgentState` subclass (here `CartState`, passed as `ReactAgent(state=CartState())`). Use `ResponseGranularity.FULL` to read them back.

### prebuilt_tools.py

`safe_calculator(expression, precision)` evaluates arithmetic through an AST, `fetch_url(url)` fetches public pages and blocks private hosts, and `create_file_tools(root, read=, write=, search=)` returns `file_read`, `file_write` and `file_search` fixed to `root`. Files such as `.env` and `*.pem` are hidden by default.

### remote_tools.py

```python
app.attach_remote_tools([{**CLIENT_TOOL, "node": "TOOL"}])
```

When the model calls `read_clipboard`, the tool node does not run it. The graph pauses and returns; the call is in the last assistant message's `tools_calls`. The client runs it, then resumes the same thread:

```python
result_msg = Message(role="tool", content=[ToolResultBlock(call_id=call["id"], output=output)])
final = await app.ainvoke({"messages": [result_msg]}, config)
```

A checkpointer is required so the paused thread is stored. To offer tools for a single run, put the same schemas in `config["remote_tools"]` instead.

### Using a different model

Replace the model string and drop `base_url`/`api_key`. For example `model="gpt-5-mini"` with `10xgraph[openai]` and `OPENAI_API_KEY`, or `model="claude-sonnet-5-5"` with `10xgraph[anthropic]`.

## Expected output

Illustrative output of `tool_decorator.py`:

```text
tags=None: ['get_order', 'cancel_order', 'current_time']
tags={'read'}: ['get_order', 'current_time']
tags={'write'}: ['cancel_order', 'current_time']
assistant: Order A-17 is shipped. I could not cancel it with the tools available.
```

## Next steps

- [Quickstart](../quickstart/): the smallest agent with one tool.
- [Streaming](../streaming/): stream tool progress with `StreamEmitter`.
- [Graph basics](../graph_basics/): how `ToolNode` fits into a graph.
