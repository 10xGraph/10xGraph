---
title: Callbacks, Validators and Lifecycle Hooks in 10xGraph
description: Add logging, timing, cost tracking, prompt injection validation, lifecycle hooks and secret redaction to a 10xGraph Python AI agent with CallbackManager.
section: Examples
group: "Production"
order: 130
label: Callbacks and validators
updated: "2026-10-28"
faq:
  - question: What is the difference between invoke callbacks and lifecycle hooks?
    answer: Invoke callbacks (before_invoke, after_invoke, on_error) wrap a single model or tool call. Lifecycle hooks (GraphLifecycleHook) fire at structural events of the whole run, such as start, end, interrupt, resume and checkpoint.
  - question: Do input validators check tool results?
    answer: No. Validators run on the messages you pass to invoke or stream. Tool results, node output and interrupt resume values are not validated, and the built-in prompt injection checks are heuristics, not a complete defence.
  - question: How do I attach a CallbackManager to a ReactAgent?
    answer: Pass it to compile, for example agent.compile(callback_manager=manager). StateGraph.compile accepts the same argument.
---

# Callbacks, validators and lifecycle hooks

## What you will build

Four small scripts that hook into a 10xGraph run. One logs, times and prices every model and tool call. One audits the run with graph lifecycle hooks. One rejects unsafe input with validators. One scrubs API keys from logs, without calling a model.

All hooks are collected in a single `CallbackManager` and handed to `compile(callback_manager=...)`.

## Key concepts

- `CallbackManager`: registry for invoke callbacks, input validators and lifecycle hooks.
- `register_before_invoke` / `register_after_invoke` / `register_on_error`: callbacks per `InvocationType` (`AI`, `TOOL`, `MCP`, `SKILL`).
- `CallbackContext`: tells a callback which node and function is running.
- `GraphLifecycleHook`: base class with no-op methods such as `on_graph_start`, `on_graph_end`, `on_graph_error`, `on_interrupt`, `on_resume`, `on_checkpoint`, `on_state_update`, `on_turn_start` and `on_turn_end`.
- `BaseValidator`, `PromptInjectionValidator`, `register_default_validators`: input validation that raises `ValidationError`.
- `mask_secrets`, `install_secret_redaction`: credential scrubbing for strings and for the `tenxgraph` logger.

## Prerequisites

```bash
pip install "10xgraph[google-genai]" python-dotenv
export GEMINI_API_KEY="your-key"
```

The agents in this folder use `gemini-2.5-flash`. `secret_redaction.py` needs no key.

## Run it

```bash
python examples/callbacks/invoke_hooks.py
python examples/callbacks/lifecycle_hooks.py
python examples/callbacks/validators.py
python examples/callbacks/secret_redaction.py
```

## How it works

### Invoke hooks (`invoke_hooks.py`)

The same `CallbackContext` object reaches `before_invoke` and `after_invoke` for one call, so its `metadata` dict carries the start time:

```python
async def start_timer(context: CallbackContext, input_data: Any) -> Any:
    context.metadata = context.metadata or {}
    context.metadata["t0"] = time.perf_counter()
    return input_data
```

Callbacks are registered for both `AI` and `TOOL` invocations, then the manager goes to `compile`:

```python
for invocation_type in (InvocationType.AI, InvocationType.TOOL):
    manager.register_before_invoke(invocation_type, start_timer)
    manager.register_after_invoke(invocation_type, stop_timer)
    manager.register_on_error(invocation_type, log_error)

app = agent.compile(callback_manager=build_callback_manager())
```

Token counts are not part of the callback payload, so the script sums `Message.usages` from the returned messages and multiplies by an illustrative price table. Replace the prices with your provider's current rates.

### Lifecycle hooks (`lifecycle_hooks.py`)

`AuditHook` subclasses `GraphLifecycleHook` and overrides the events it cares about. Returning `None` keeps the state unchanged. Returning a state object replaces it, which is how a hook can inject data at `on_graph_start`.

```python
manager.register_lifecycle_hook(AuditHook())
app = agent.compile(checkpointer=InMemoryCheckpointer(), callback_manager=manager)
```

`on_interrupt` and `on_resume` fire for human-in-the-loop pauses. `on_turn_start` and `on_turn_end` fire only in realtime sessions. The exception from a failing run is always re-raised after `on_graph_error`.

### Validators (`validators.py`)

`register_default_validators` adds `PromptInjectionValidator` and `MessageContentValidator`. A custom rule only needs `validate`:

```python
class BusinessPolicyValidator(BaseValidator):
    async def validate(self, messages: list[Message]) -> bool:
        ...
        raise ValidationError(f"Out of scope: {topic}", "forbidden_topic", {"topic": topic})
```

A rejected prompt raises `ValidationError` from `ainvoke` before any node runs. With `strict_mode=False` the prompt injection validator only logs a warning.

### Secret redaction (`secret_redaction.py`)

`mask_secrets(text)` replaces OpenAI, Google, GitHub, Slack and AWS keys, bearer tokens and `key=value` secrets. `install_secret_redaction("tenxgraph")` adds the filter to that logger and its handlers, so call it after you configure your handlers. It is a safety net, not a guarantee.

## Expected output

Illustrative output of `invoke_hooks.py`:

```text
[before] ai MAIN
[after]  ai MAIN took 1.42s
[before] tool get_weather
[after]  tool get_weather took 0.00s
[before] tool get_weather
[error]  tool get_weather: No weather station for Atlantis
[before] ai MAIN
[after]  ai MAIN took 1.10s
assistant: Paris is 21C and sunny. I could not find weather data for Atlantis.
model calls: 2 (2.52s)
tool calls:  1 (0.00s), errors: 1
tokens: in=412 out=63
estimated cost: $0.000281
```

## Using a different model

Change the model string and install the matching extra. For OpenAI use `model="gpt-5-mini"` with `pip install "10xgraph[openai]"` and `OPENAI_API_KEY`. For Anthropic use `model="claude-sonnet-5-5"` with `pip install "10xgraph[anthropic]"` and `ANTHROPIC_API_KEY`. Update the key in `PRICE_PER_M_TOKENS` too.

## Next steps

- [Observability](../observability/) sends the same run events to a console, OpenTelemetry or a message broker.
- [Structured output](../structured_output/) validates the model's output instead of its input.
- [Human in the loop](../human_in_the_loop/) uses the interrupt events that `on_interrupt` and `on_resume` observe.
