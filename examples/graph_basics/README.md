---
title: 10xGraph StateGraph Basics - Custom State, Routing and Command
description: Learn the 10xGraph StateGraph in Python with no API key: custom AgentState fields, function nodes, conditional edges and Command, plus a hand-built ReAct agent.
section: Examples
group: "Getting started"
order: 20
label: Graph basics
updated: "2026-10-28"
faq:
  - question: Can I run a 10xGraph StateGraph without an LLM?
    answer: Yes. A node is any function, so a graph of plain functions runs with no provider SDK or API key. state_graph_no_llm.py does this.
  - question: When should I build the ReAct loop by hand instead of using ReactAgent?
    answer: Use ReactAgent unless you need a different shape, such as extra nodes between the model and the tools or custom routing. react_by_hand.py shows the exact graph ReactAgent builds, so you can copy and change it.
  - question: What does Command do?
    answer: A node returns Command(update=..., goto=...) to change state and choose the next node in one step, instead of relying on an edge defined at build time.
---

## What you will build

Two small graphs. The first routes support tickets with plain functions and a custom state, and runs without any key. The second wires the ReAct loop (model, tools, back to the model) by hand with `Agent` and `ToolNode`, which is what `ReactAgent` does for you.

## Key concepts

- `StateGraph`: collects nodes and edges, then `compile()` produces a `CompiledGraph`.
- `AgentState` subclass: adds your own fields to `context`; the extra fields are saved with the state.
- `add_conditional_edges(node, fn, path_map)`: `fn(state)` returns a key, `path_map` maps it to a node.
- `Command(update=..., goto=...)`: a node updates state and picks the next node itself.
- `START` / `END`: graph entry and exit markers. `add_edge(START, "X")` sets the entry point.
- `ResponseGranularity`: `LOW` returns messages, `FULL` also returns the final state.
- `Agent` and `ToolNode`: the LLM node and the tool executor used in `react_by_hand.py`.

## Prerequisites

```bash
pip install 10xgraph                       # state_graph_no_llm.py
pip install "10xgraph[google-genai]" python-dotenv   # react_by_hand.py
export GEMINI_API_KEY=...                  # react_by_hand.py only
```

## Run it

```bash
python examples/graph_basics/state_graph_no_llm.py
python examples/graph_basics/react_by_hand.py
```

## How it works

### state_graph_no_llm.py

The state adds three fields. Nodes receive the state by parameter name.

```python
class TicketState(AgentState):
    intent: str = ""
    amount: float = 0.0
    visited: list[str] = Field(default_factory=list)
```

`INTAKE` sets `intent`, then a conditional edge routes on it:

```python
graph.add_edge(START, "INTAKE")
graph.add_conditional_edges(
    "INTAKE", route_by_intent,
    {"faq": "FAQ", "refund": "REFUND", "other": "FALLBACK"},
)
```

`REFUND` has no outgoing edge. It returns a `Command`, so the node chooses between review and the end:

```python
return Command(update=state, goto="review")   # or goto=END
```

A node can also return a `Message`, which is appended to `state.context`. Note that the engine applies the `add_messages` reducer to `context`; custom fields are replaced by assignment.

The script runs each ticket twice: once with the default `LOW` granularity, and once with `ResponseGranularity.FULL` to read `result["state"].visited`.

### react_by_hand.py

```python
main_agent = Agent(model="gemini-2.5-flash", tool_node="TOOL", ...)
graph.add_node("MAIN", main_agent)
graph.add_node("TOOL", ToolNode([get_weather]))
graph.set_entry_point("MAIN")
graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
graph.add_edge("TOOL", "MAIN")
```

`route` returns `"TOOL"` when the last assistant message has `tools_calls`, otherwise `END`. The edge from `TOOL` back to `MAIN` closes the loop. `ReactAgent(model=..., tools=[...])` builds this same graph.

### Using a different model

Change `model="gemini-2.5-flash"` to another string, for example `"gpt-5-mini"` with `10xgraph[openai]` and `OPENAI_API_KEY`, or `"claude-sonnet-5-5"` with `10xgraph[anthropic]`.

## Expected output

Illustrative output of `state_graph_no_llm.py`:

```text
user: Large refund please
  user: Large refund please
  assistant: Large refund, sending to review.
  assistant: A person will review it today.
  path=['intake', 'refund', 'review'] intent=refund amount=250.0
```

## Next steps

- [Quickstart](../quickstart/): the same loop with `ReactAgent`.
- [Streaming](../streaming/): stream a graph run chunk by chunk.
- [Tools](../tools/): the tool features available to `ToolNode`.
