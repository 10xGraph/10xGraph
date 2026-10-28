---
title: Streaming a 10xGraph Python AI Agent with astream and StreamEmitter
description: Stream agent output in 10xGraph with astream: read StreamChunk events, print text as it arrives, report tool progress with StreamEmitter, and stop a run with astop.
section: Examples
group: "Getting started"
order: 30
label: Streaming
updated: "2026-10-28"
faq:
  - question: What does astream yield?
    answer: StreamChunk objects. Each has an event (message, state, updates or error), and for message events a Message that may be a partial delta of the model's text. Tool progress chunks carry a data dict instead of a message.
  - question: How do I show progress from inside a tool?
    answer: Add a parameter named emit typed StreamEmitter | None. During astream it is a StreamEmitter, so emit.progress(...) appears in the stream. During ainvoke it is None.
  - question: Does astop need a checkpointer?
    answer: Yes. astop records the stop request on the thread through the checkpointer and returns ok False with reason no-checkpointer without one. The running graph checks the flag between nodes.
---

## What you will build

A travel agent whose tool reports progress while it works, streamed to the terminal with `astream`. A second script starts a long run and stops it from another task with `astop`.

Both use Anthropic `claude-sonnet-5-5`.

## Key concepts

- `astream(input, config)`: async generator of `StreamChunk`.
- `StreamEvent`: `MESSAGE`, `STATE`, `UPDATES`, `ERROR`.
- `Message.delta`: true for partial text; the complete message arrives later with `delta=False`.
- `StreamEmitter`: injected into tools via a parameter named `emit`; methods `progress`, `message`, `update`, `error`.
- `astop(config)`: asks a running thread to stop; needs a checkpointer and the run's `thread_id`.

## Prerequisites

```bash
pip install "10xgraph[anthropic]" python-dotenv
export ANTHROPIC_API_KEY=...
```

## Run it

```bash
python examples/streaming/stream_agent.py
python examples/streaming/stop_stream.py
```

## How it works

### stream_agent.py

The tool takes `emit`, which is hidden from the model's schema:

```python
async def search_flights(origin: str, destination: str, emit: StreamEmitter | None = None) -> str:
    if emit:
        emit.progress("Contacting airlines", data={"route": f"{origin}-{destination}"})
```

The loop reads the chunks:

```python
async for chunk in app.astream(inp, config={"thread_id": "stream-1"}):
    if chunk.event == StreamEvent.MESSAGE:
        if chunk.message is None:      # StreamEmitter chunk, see chunk.data
            ...
        elif chunk.message.delta:      # partial model text
            print(chunk.message.text(), end="", flush=True)
```

Progress chunks use `event=message` with `message=None` and a `data` dict whose `status` is `tool_progress`. Tool calls arrive as assistant messages with `tools_calls`; tool output arrives as `role == "tool"` messages.

### stop_stream.py

The graph is compiled with `InMemoryCheckpointer()`. One task consumes `astream`; after five seconds the main task calls:

```python
status = await app.astop({"thread_id": "stop-demo"})
```

The stop flag is checked between nodes, so the run ends after the current tool call or model call finishes, not in the middle of a token stream.

### Using a different model

Change `model="claude-sonnet-5-5"`, for example to `"gpt-5-mini"` (`10xgraph[openai]`, `OPENAI_API_KEY`) or `"gemini-2.5-flash"` (`10xgraph[google-genai]`, `GEMINI_API_KEY`).

## Expected output

Illustrative; text and timing will vary.

```text
[model asked for tools: ['search_flights']]
[progress] search_flights: Contacting airlines
[progress] search_flights: Comparing prices
[progress] search_flights: Ranking results
[tool result] Cheapest Oslo to Rome: $212 nonstop, departs 08:15.
The cheapest option is a $212 nonstop flight at 08:15.
```

## Next steps

- [Quickstart](../quickstart/): the non-streaming version.
- [Tools](../tools/): injected parameters and client-side tools.
- [Graph basics](../graph_basics/): how the graph routes between nodes.
