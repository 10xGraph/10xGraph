---
title: Test 10xGraph Agents Without an LLM Using pytest
description: Unit test a 10xGraph Python AI agent offline with TestAgent, QuickTest, MockToolRegistry, MockMCPClient and override_node, with no API keys or network calls.
section: Examples
group: "Quality"
order: 220
label: Testing agents
updated: "2026-10-28"
faq:
  - question: How do I test a 10xGraph agent without calling an LLM?
    answer: Replace the Agent node with TestAgent, which returns the responses you list. You can pass it to add_node, or swap it into an existing graph with override_node. The graph, routing, state and tool nodes all run for real.
  - question: Can I check which tools an agent called?
    answer: Yes. MockToolRegistry wraps each function and records its calls, so you can use assert_called_with and assert_call_count. QuickTest.with_tools returns a TestResult with assert_tool_called.
  - question: Why does pytest warn about TestAgent being collected?
    answer: Pytest collects classes named Test*. Import it under another name, for example "from tenxgraph.qa.testing import TestAgent as CannedAgent", as these files do.
  - question: Does MockMCPClient work end to end inside ToolNode?
    answer: ToolNode accepts it, but ToolNode drives a real fastmcp client (async context manager, ping and tool objects). MockMCPClient only provides list_tools and call_tool, so use it to test the code around your MCP client rather than ToolNode's MCP execution.
---

# Test 10xGraph agents without an LLM

## What you will build

A pytest suite for a 10xGraph agent that runs in about two seconds with no API key. The tests script the model's replies, fake the tools, and swap nodes in a production graph, then assert on the result.

Five files cover the five tools in `tenxgraph.qa.testing`. Each file is a normal pytest module you can copy into your own project.

## Key concepts

- `TestAgent`: a stand-in for `Agent` that returns canned `responses` in order and records every call (`call_count`, `get_last_messages()`).
- `QuickTest`: one-call helpers (`single_turn`, `multi_turn`, `with_tools`, `custom`) that build a graph, run it and return a `TestResult`.
- `TestResult`: chainable assertions such as `assert_contains`, `assert_tool_called` and `assert_message_count`.
- `MockToolRegistry`: registers fake tools by name and records their calls.
- `MockMCPClient`: a fake MCP server with `add_tool`, `list_tools` and `call_tool`.
- `override_node`: replaces a node in a `StateGraph` before compile, or in a `CompiledGraph` after.

## Prerequisites

```bash
pip install "10xgraph" pytest pytest-asyncio
pip install "10xgraph[mcp]"   # only for test_mock_mcp.py, which builds a ToolNode with a client
```

No API keys and no Docker. Nothing in this folder makes a network call.

## Run it

From the repository root:

```bash
pytest examples/testing -q -c examples/testing/pytest.ini
```

`pytest.ini` turns on `asyncio_mode = auto` so `async def test_...` works without decorators, and it avoids the coverage options the library's own `pyproject.toml` adds. Copy it into your project if you use the same layout.

| File | Shows |
|---|---|
| `test_with_test_agent.py` | `TestAgent` responses, call history and assertions |
| `test_quick_test.py` | `QuickTest` single turn, multi turn, tools and chained assertions |
| `test_mock_tools.py` | `MockToolRegistry` call recording and `ToolNode` wiring |
| `test_mock_mcp.py` | `MockMCPClient` protocol methods and call assertions |
| `test_override_node.py` | `override_node` before and after compile |

## How it works

`TestAgent` plugs into a graph like the real agent. The `MAIN` node is the only thing that changes:

```python
agent = CannedAgent(responses=["Paris is the capital of France."])
graph = StateGraph()
graph.add_node("MAIN", agent)
graph.set_entry_point("MAIN")
graph.add_edge("MAIN", END)
app = graph.compile()
```

Because `responses` cycles, one agent can script a whole conversation, and `agent.get_last_messages()` shows the system prompt and user text the model would have received.

`QuickTest` collapses that setup into one call and returns a result you can chain:

```python
result = await QuickTest.with_tools(
    query="Weather in NYC?",
    response="It is sunny in NYC.",
    tools=["get_weather"],
    tool_responses={"get_weather": "sunny, 24C"},
)
result.assert_tool_called("get_weather").assert_contains("sunny")
```

`MockToolRegistry` records calls made through the functions it hands out, which you can give to a `ToolNode`:

```python
tools = MockToolRegistry().register("get_weather", lambda city: f"Sunny in {city}")
node = ToolNode(tools.get_tool_list())
tools.assert_called_with("get_weather", city="Paris")
```

`override_node` is the way to test a graph you do not want to rebuild. `test_override_node.py` keeps the production routing and tool node, and replaces only the model call:

```python
graph = create_graph()                      # production factory
graph.override_node("MAIN", CannedAgent(responses=["stubbed answer"]))
app = graph.compile()
```

`TestAgent` returns tool calls with the fixed argument `{"query": "test query"}` on its first call when it has `tools`. A tool it triggers must therefore accept a `query` parameter, or the tool node returns an error message instead of a result.

## Expected output

Illustrative:

```text
$ pytest examples/testing -q -c examples/testing/pytest.ini
........................                                                 [100%]
24 passed in 1.7s
```

## Using a different model

These tests never call a model, so the model name does not matter. `TestAgent(model=...)` defaults to `"test-model"`. Your production `Agent` can use any provider (`gpt-5-mini`, `gemini-2.5-flash`, `claude-sonnet-5-5`) and the tests stay the same.

## Next steps

- [Evaluation](../evaluation/) scores a real model with trajectory and LLM-judge criteria.
- [Serving](../serving/) exposes a graph over HTTP so you can test it end to end.
