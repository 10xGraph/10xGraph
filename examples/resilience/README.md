---
title: Retries, Fallback Models and Circuit Breaker for a Python AI Agent in 10xGraph
description: Make a 10xGraph agent survive provider outages with RetryConfig backoff, a circuit breaker, cross-company fallback models, and a recursion limit.
section: Examples
group: "Production"
order: 90
label: Resilience
updated: "2026-10-28"
faq:
  - question: How do I fall back to a model from a different provider?
    answer: Pass a (model, provider) tuple in fallback_models, for example ("gpt-5-mini", "openai"). A plain string reuses the primary agent's provider, so a bare "gpt-5-mini" on a Claude agent would be sent to Anthropic.
  - question: Which errors are retried?
    answer: HTTP status codes in RetryConfig.retryable_status_codes (429, 500, 502, 503, 529 by default) and connection or timeout errors. Any other error skips the remaining retries for that model and moves straight to the next fallback.
  - question: What does the circuit breaker do?
    answer: When circuit_breaker_enabled is true, consecutive failed attempt cycles for a (provider, model) pair open its circuit. While it is open the agent skips that model and goes to the next fallback, until circuit_breaker_reset_timeout passes and one trial call is allowed.
  - question: How do I stop an agent that loops forever?
    answer: Set recursion_limit in the run config (default 25). When the step count is reached, the graph raises GraphRecursionError, which you can catch.
---

## What you will build

An agent that answers support questions on Anthropic `claude-sonnet-5-5`, retries transient failures with exponential backoff, and fails over to OpenAI `gpt-5-mini` and then Gemini `gemini-2.5-flash` if Anthropic stays down. A circuit breaker keeps the agent from hammering a provider that is already failing.

A second script, which needs no API key, shows the other half of resilience: capping a runaway loop with `recursion_limit` and handling the exceptions the library raises.

## Key concepts

- `RetryConfig`: `max_retries`, `initial_delay`, `backoff_factor`, `max_delay`, `retryable_status_codes`, and the circuit breaker fields `circuit_breaker_enabled`, `circuit_breaker_threshold`, `circuit_breaker_reset_timeout`.
- `retry_config=` on `ReactAgent` / `Agent`: `True` (default) uses `RetryConfig()`, `False` or `None` turns retries off.
- `fallback_models=`: ordered list of `"model"` strings (same provider as the primary) or `(model, provider)` tuples (any provider).
- `recursion_limit`: a run-config key that caps node steps.
- `tenxgraph.core.exceptions`: `GraphError` (base), `GraphRecursionError`, `NodeError`, `NodeTimeoutError`, `StorageError` and others. Each has `error_code`, `context` and `to_dict()`.

## Prerequisites

```bash
pip install "10xgraph[anthropic,openai,google-genai]" python-dotenv
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

`recursion_limit.py` needs only `pip install 10xgraph`. You only need keys for fallback providers you expect to reach.

## Run it

```bash
python examples/resilience/retry_and_fallback.py
python examples/resilience/recursion_limit.py
```

## How it works

### Retry order

For each LLM call the agent builds the attempt list: the primary model, then each fallback in order. For one model it makes up to `max_retries + 1` calls. After a retryable error it sleeps `initial_delay * backoff_factor ** n`, capped at `max_delay`. With the settings below that is 1 s, then 2 s.

```python
retry_config = RetryConfig(
    max_retries=2,
    initial_delay=1.0,
    backoff_factor=2.0,
    max_delay=20.0,
    circuit_breaker_enabled=True,
    circuit_breaker_threshold=3,
    circuit_breaker_reset_timeout=60.0,
)
```

When a model runs out of retries, or raises an error that is not retryable (a 400 or 401, for example), the agent moves to the next model at once. If every model fails, the last exception is re-raised as the provider SDK raised it.

### Circuit breaker

Each `(provider, model)` pair gets a breaker, stored on the agent instance so its state lasts across calls. One failed attempt cycle (retries exhausted or non-retryable error) counts as one failure. After `circuit_breaker_threshold` failures in a row the circuit opens and the agent skips that model without calling it. After `circuit_breaker_reset_timeout` seconds one trial call is allowed. Success closes the circuit, and another failure opens it again. The breaker is off by default.

### Fallback models across companies

```python
agent = ReactAgent(
    model="claude-sonnet-5-5",
    ...,
    retry_config=retry_config,
    fallback_models=[
        ("gpt-5-mini", "openai"),
        ("gemini-2.5-flash", "google"),
    ],
)
```

The provider for the primary model is detected from its name. A fallback given as a plain string inherits that provider, which is only right within one company. The tuple form is `(model, provider)` with provider `"openai"`, `"google"` or `"anthropic"`. The fallback client is created on first use, so the OpenAI and Google SDKs are only needed if a fallback is actually reached. Fallbacks also reuse the primary agent's `base_url`, so do not combine them with a custom endpoint unless it serves every model in the list.

### Recursion limit and library exceptions

```python
config = {"thread_id": "loop-1", "recursion_limit": 6}
try:
    await app.ainvoke(inputs, config)
except GraphRecursionError as exc:
    print(exc.error_code, exc.context)
```

`GraphRecursionError` carries `max_steps`, `current_step` and `current_node`. All library errors subclass `GraphError`, so `except GraphError` catches any of them, and `exc.to_dict()` gives a JSON-friendly form for an API response. The checkpointer keeps the state saved before the stop. The library also logs these errors with a traceback at ERROR level, so expect log noise when you run the script.

## Expected output

Illustrative. `recursion_limit.py` printed this when run locally (log lines omitted):

```text
stopped by recursion limit
  code:    RECURSION_001
  context: {'max_steps': 6, 'current_step': 6, 'current_node': 'ACT'}
  dict:    {'error_type': 'GraphRecursionError', 'error_code': 'RECURSION_001', ...}
caught as GraphError: GraphRecursionError - Graph execution exceeded recursion limit: 4
attempts recorded before the stop: 3
```

`retry_and_fallback.py` prints one line when it works:

```text
assistant: Ticket T-884 is open and assigned to the billing team.
```

## Using a different model

Change the `model` string and the first tuple entries. For example `model="gpt-5-mini"` with `pip install "10xgraph[openai]"`, and fallbacks `[("claude-sonnet-5-5", "anthropic"), ("gemini-2.5-flash", "google")]`.

## Next steps

- [Graceful shutdown](../graceful_shutdown/): finish and flush work when the process is told to stop.
- [Human in the loop](../human_in_the_loop/): pause risky steps for approval.
- [Context management](../context_management/): keep long conversations under the token limit.
