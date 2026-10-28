---
title: Context Window Management for Python AI Agents - Trim and Summarize with 10xGraph
description: Keep long agent conversations inside the token limit in 10xGraph with MessageContextManager trimming and SummaryContextManager summaries from a cheaper model.
section: Examples
group: "Production"
order: 120
label: Context management
updated: "2026-10-28"
faq:
  - question: Why is my context manager not doing anything?
    answer: The agent only calls it when trim_context=True is set on the agent (ReactAgent or Agent). Without the flag the context manager is registered but unused during LLM calls.
  - question: What does max_messages count in MessageContextManager?
    answer: User messages. It keeps the most recent max_messages user turns with their replies, and always keeps system messages. Set remove_tool_msgs=True to also drop tool calls and tool results.
  - question: Can the summary model come from a different provider than the agent?
    answer: Yes. SummaryContextManager takes its own model string and detects the provider from the name, so a Claude agent can use gemini-2.5-flash or an OpenAI model for summaries. Set api_style="chat" only for OpenAI-compatible models without Responses API support.
  - question: Where does the summary go?
    answer: Into state.context_summary. It is a rolling summary, so each new summarisation is appended to the earlier one, and it is placed in front of the recent messages when the next prompt is built.
---

## What you will build

Two ways to stop a long chat from outgrowing the model's context window.

- `message_window.py` keeps only the last 3 user turns. Older turns are dropped, which is cheap but forgetful.
- `summarize.py` replaces old turns with a short summary and keeps the recent messages word for word. The conversation is answered by Anthropic `claude-sonnet-5-5`, while the summaries are written by Google `gemini-2.5-flash`. Summarising is a simple task, so a smaller, cheaper model from another company does it at lower cost, and the context manager has its own model setting, independent of the agent's.

Both print the size of the stored context after each turn, and the second prints `context_summary`.

## Key concepts

- `MessageContextManager(max_messages=10, remove_tool_msgs=False)`: window by user-message count.
- `SummaryContextManager(model, max_messages=30, token_budget=None, keep_recent=8, remove_tool_msgs=False, summary_system_prompt=None, max_summary_tokens=600, api_style="responses")`: summarises when the message count or the estimated token count passes its limit.
- `ReactAgent(context_manager=..., trim_context=True)`: attaches the manager and turns on trimming. Both are needed.
- `state.context_summary`: the rolling summary text. Read it with `ResponseGranularity.PARTIAL` (keys `context`, `summary`, `messages`) or `FULL`.
- `ResponseGranularity`: how much of the state `ainvoke` returns.

## Prerequisites

```bash
pip install "10xgraph[anthropic,google-genai]" python-dotenv
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
```

`message_window.py` only needs the Anthropic key and extra.

## Run it

```bash
python examples/context_management/message_window.py
python examples/context_management/summarize.py
```

## How it works

### Windowing

```python
agent = ReactAgent(
    model="claude-sonnet-5-5",
    context_manager=MessageContextManager(max_messages=3),
    trim_context=True,
)
```

The manager counts user messages. When there are more than `max_messages`, it keeps the latest ones with everything after them, and keeps system messages. The trim runs before each LLM call and again when state is saved, so the stored thread stays small too. The last question in the script asks for the favourite colour from turn 1, which is no longer in context, so the agent cannot answer it.

### Summarising

```python
summariser = SummaryContextManager(
    model="gemini-2.5-flash",
    max_messages=6,
    keep_recent=4,
    max_summary_tokens=400,
)
agent = ReactAgent(model="claude-sonnet-5-5", context_manager=summariser, trim_context=True)
```

When the context passes `max_messages` (or `token_budget`, a rough estimate at 4 characters per token), everything except the last `keep_recent` non-system messages is sent to the summary model. System messages are never summarised. The result is stored in `state.context_summary`, and the older messages are removed. Later summaries are appended to the earlier text, so nothing is thrown away permanently. When the next prompt is built, the summary is injected as an assistant message before the recent messages. If the summary call fails, the context is left unchanged and the run continues.

`api_style` selects the OpenAI endpoint for the summary model (`"responses"` by default, `"chat"` for compatible providers without it). It has no effect with a Gemini model.

### Reading the summary

```python
result = await graph.ainvoke(inputs, config, response_granularity=ResponseGranularity.PARTIAL)
print(len(result["context"]), result["summary"])
```

## Expected output

Illustrative; model text and the exact turn where summarising starts will differ.

`message_window.py`:

```text
user: I have a dog called Biscuit.
assistant: Biscuit is a great name!
  -> context: 6 messages, 3 from the user
...
user: What is my favourite colour?
assistant: I do not have that in our conversation. What is it?
  -> context: 6 messages, 3 from the user
```

`summarize.py`:

```text
user: I am vegetarian.
assistant: Noted. I will plan vegetarian-friendly food stops.
  -> context: 5 messages
  -> context_summary: The user is planning a 5-day trip to Japan in April, likes food and quiet places, lands in Tokyo and departs from Osaka, and has about 2000 euros excluding flights.
```

## Using a different model

Change the model strings. For example an OpenAI agent, `ReactAgent(model="gpt-5-mini", ...)` with `pip install "10xgraph[openai]"`, and a summariser on a third company, `SummaryContextManager(model="claude-sonnet-5-5", ...)`. For an OpenAI summary model on an older endpoint, add `api_style="chat"`.

## Next steps

- [Persistence](../persistence/): store the trimmed thread in SQLite or Postgres.
- [Resilience](../resilience/): fallbacks when the agent model or the summary model fails.
- [Human in the loop](../human_in_the_loop/): pause long-running threads for approval.
