---
title: 10xGraph Quickstart - Your First Python AI Agent with Tools
description: Build a Python AI agent with 10xGraph in about 30 lines. A ReactAgent with one tool, run with ainvoke or invoke, using a thread_id and clean shutdown.
section: Examples
group: "Getting started"
order: 10
label: Quickstart
updated: "2026-10-28"
faq:
  - question: What is the difference between invoke and ainvoke?
    answer: ainvoke is the async method and the one to use inside async code such as a web server. invoke is a blocking wrapper that runs ainvoke with asyncio.run, so it only works when no event loop is running.
  - question: Why do I pass a thread_id?
    answer: The thread_id names a conversation. The checkpointer stores state per thread, so calling again with the same thread_id continues the earlier chat. If you omit it, a random one is generated and a warning is logged.
  - question: Do I have to call aclose()?
    answer: Call it, or use async with, when you are done. It flushes background checkpoint writes and releases the checkpointer, store and publisher.
---

## What you will build

The smallest useful agent: a `ReactAgent` with one weather tool. It sends a question, lets the model call the tool, and prints the final answer. A second file does the same with the blocking `invoke()`.

The model is `gpt-5-mini` from OpenAI, written directly in the code.

## Key concepts

- `ReactAgent`: prebuilt agent that wires the model node, the tool node and the tool-loop routing.
- `agent.compile()`: returns a `CompiledGraph`. The default checkpointer is in memory.
- `ainvoke()` / `invoke()`: run the graph once and return the new messages.
- `thread_id`: identifies the conversation in the checkpointer.
- `async with app` / `aclose()`: releases resources on exit.

## Prerequisites

```bash
pip install "10xgraph[openai]" python-dotenv
export OPENAI_API_KEY=...
```

## Run it

```bash
python examples/quickstart/quickstart.py
python examples/quickstart/quickstart_sync.py
```

## How it works

A tool is a plain function. Its type hints and docstring become the schema the model sees.

```python
def get_weather(location: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {location} is sunny, 24 C."
```

`ReactAgent` takes the model string and the tools. The provider is detected from the model name.

```python
agent = ReactAgent(
    model="gpt-5-mini",
    system_prompt=[{"role": "system", "content": "You are a concise weather assistant."}],
    tools=[get_weather],
)
app = agent.compile()
```

Run it inside `async with app` so `aclose()` runs even if the call raises:

```python
async with app:
    result = await app.ainvoke(
        {"messages": [Message.text_message("What is the weather in Tokyo?")]},
        config={"thread_id": "quickstart-1"},
    )
```

The default response granularity is `LOW`, so `result["messages"]` holds the messages produced by the run. `message.text()` returns the text of a message.

`quickstart_sync.py` calls `app.invoke(...)` twice with the same `thread_id`; the second call sees the first turn.

### Using a different model

Change the model string and install the matching extra. For example `model="gemini-2.5-flash"` with `pip install "10xgraph[google-genai]"` and `GEMINI_API_KEY`, or `model="claude-sonnet-5-5"` with `10xgraph[anthropic]` and `ANTHROPIC_API_KEY`.

## Expected output

Illustrative; the wording will vary.

```text
user: What is the weather in Tokyo?
assistant: [{'id': 'call_1', 'type': 'function', 'function': {'name': 'get_weather', ...}}]
tool: The weather in Tokyo is sunny, 24 C.
assistant: It is sunny and 24 C in Tokyo.

assistant: It is sunny and 24 C in Tokyo.
```

## Next steps

- [Graph basics](../graph_basics/): what `ReactAgent` builds, and graphs without an LLM.
- [Streaming](../streaming/): print text as it is generated.
- [Tools](../tools/): decorators, injected parameters and prebuilt tools.
