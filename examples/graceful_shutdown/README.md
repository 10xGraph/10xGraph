---
title: Graceful Shutdown for a Python AI Agent Service with 10xGraph
description: Handle SIGINT and SIGTERM in a long-running 10xGraph agent service, flush checkpoint writes, and release resources with graph.aclose() and async with.
section: Examples
group: "Production"
order: 80
label: Graceful shutdown
updated: "2026-10-28"
faq:
  - question: What does graph.aclose() do, and in what order?
    answer: It waits for background tasks (including fire-and-forget checkpoint writes) up to shutdown_timeout, then releases the checkpointer, then the publisher, then the store. It returns a dict with one stats entry per component and a total_duration. A second call returns {"status": "already_closed"}.
  - question: Do I need GracefulShutdownManager if I use async with graph?
    answer: They solve different problems. async with graph guarantees aclose() runs when the block exits. GracefulShutdownManager turns SIGINT and SIGTERM into a flag your loop can check, so the process leaves the block on purpose instead of being killed mid-run.
  - question: How does this work in Kubernetes or Docker?
    answer: Both send SIGTERM first and SIGKILL after a grace period. Set terminationGracePeriodSeconds (Kubernetes) or stop_grace_period (Docker Compose) to more than your longest job plus shutdown_timeout, and make sure the Python process receives the signal.
---

## What you will build

A small support service that takes jobs from a list, answers each with a Gemini-backed `ReactAgent`, and appends the result to a ledger file. When you press Ctrl+C or the container receives SIGTERM, the service finishes the job it is on, stops taking new ones, and calls `graph.aclose()`. The stats dict that `aclose()` returns is printed so you can see what was flushed and released.

A second script shows the short form, `async with graph:`, which calls `aclose()` for you even when the body raises.

## Key concepts

- `GracefulShutdownManager.register_signal_handlers()`: installs asyncio handlers for SIGINT and SIGTERM that set `shutdown_requested`. They do not raise, so your loop chooses where to stop.
- `manager.wait_for_shutdown()`: awaitable that returns once a signal arrives. Wrap it in `asyncio.wait_for` for an interruptible idle sleep.
- `delayed_keyboard_interrupt()` / `DelayedKeyboardInterrupt`: a context manager that holds back SIGINT and SIGTERM for a short critical section and re-delivers them on exit.
- `compile(shutdown_timeout=...)`: the time budget `aclose()` has. The default is 30 seconds.
- `await graph.aclose()`: releases everything the graph owns and returns a stats dict.
- `async with graph:`: calls `aclose()` on exit, including after an exception.

## Prerequisites

```bash
pip install "10xgraph[google-genai]" python-dotenv
export GEMINI_API_KEY="..."
```

## Run it

```bash
python examples/graceful_shutdown/service_loop.py
# press Ctrl+C, or from another terminal: kill -TERM <pid>

python examples/graceful_shutdown/async_with.py
```

## How it works

### 1. Turn signals into a flag

```python
manager = GracefulShutdownManager(shutdown_timeout=20.0)
manager.register_signal_handlers()
```

The handler only sets `manager.shutdown_requested = True`. The loop checks it between jobs, so a job that is already running is never cut off. Where the platform lacks `add_signal_handler` (Windows), the manager logs a warning instead of failing.

### 2. Protect the critical section

```python
def commit_result(job: str, answer: str) -> None:
    with delayed_keyboard_interrupt(), LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"job": job, "answer": answer}) + "\n")
        f.flush()
        os.fsync(f.fileno())
```

Inside the block, SIGINT and SIGTERM are recorded instead of acted on. When the block exits, the original handler runs. Keep these blocks short: a long one makes the service ignore shutdown requests.

### 3. Close the graph in `finally`

```python
graph = agent.compile(checkpointer=InMemoryCheckpointer(), shutdown_timeout=20.0)
...
finally:
    stats = await graph.aclose()
```

`aclose()` runs in a fixed order:

1. **Background tasks.** The task manager waits for pending work, mainly checkpoint writes, which run in the background so they do not slow streaming. This step gets the full `shutdown_timeout`.
2. **Checkpointer.** `arelease()` closes its connections. This step gets `shutdown_timeout / 3`.
3. **Publisher.** `close()` flushes and disconnects the event bus, with `shutdown_timeout / 3`.
4. **Store.** `arelease()` on the vector store, with `shutdown_timeout / 3`.

The order matters because the checkpointer has to stay open until the background writes that use it are done. A component that was not configured reports `{"status": "skipped", ...}`. A component that exceeds its time reports `{"status": "timeout", ...}` and the next one still runs.

### 4. The simple form

```python
async with graph:
    result = await graph.ainvoke(...)
```

`__aexit__` calls `aclose()`. Calling it again, by hand or through a nested block, returns `{"status": "already_closed"}`.

## Kubernetes and Docker

Kubernetes sends SIGTERM when it stops a pod and sends SIGKILL after `terminationGracePeriodSeconds` (30 seconds by default). Docker does the same with `docker stop` and `stop_grace_period`. SIGKILL cannot be caught, so anything not flushed by then is lost.

- Set `terminationGracePeriodSeconds` higher than your longest job plus `shutdown_timeout`.
- Run the Python process as PID 1 with an exec-form `CMD ["python", "..."]`, or put an init such as `tini` in front. A shell wrapper may not forward SIGTERM.
- Stop accepting new work as soon as `shutdown_requested` is set, as the loop does here, and fail the readiness probe if you serve traffic.

## Expected output

Illustrative. Job text, timings and the answers will differ.

```text
Service running. Press Ctrl+C (or send SIGTERM) to stop.
[job 1] Where is order A-1001? -> Order A-1001 left the warehouse and arrives in 2 days.
[job 2] Where is order B-2002? -> Order B-2002 left the warehouse and arrives in 2 days.
^C
Shutting down...
{
  "background_tasks": {"status": "completed", "initial_tasks": 2, "completed_tasks": 2, "remaining_tasks": 0, "duration_seconds": 0.01},
  "checkpointer": {"status": "completed", "duration": 0.001},
  "publisher": {"status": "skipped", "reason": "no publisher"},
  "store": {"status": "skipped", "reason": "no store"},
  "total_duration": 0.02
}
```

## Using a different model

Change the model string in `ReactAgent(model=...)` and install the matching extra. For example `model="gpt-5-mini"` with `pip install "10xgraph[openai]"` and `OPENAI_API_KEY`, or `model="claude-sonnet-5-5"` with `pip install "10xgraph[anthropic]"` and `ANTHROPIC_API_KEY`.

## Next steps

- [Persistence](../persistence/): checkpointers that need `aclose()` to release connections.
- [Resilience](../resilience/): retries, fallbacks and recursion limits for runs that must not fail.
- [Human in the loop](../human_in_the_loop/): pause a run and resume it later, even after a restart.
