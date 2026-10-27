---
title: Observability and Event Publishers for 10xGraph Agents
description: Stream 10xGraph run events to the console, OpenTelemetry, Logfire, LangSmith, Redis, Kafka or RabbitMQ and combine several publishers in Python.
section: Examples
group: "Production"
order: 140
label: Observability
updated: "2026-10-28"
faq:
  - question: How do I add a publisher to a ReactAgent?
    answer: Pass publisher= to the ReactAgent constructor, either one publisher or a list. ReactAgent forwards it to the StateGraph it builds. With a hand-built graph, use StateGraph(publisher=...).
  - question: Can I use more than one publisher?
    answer: Yes. Pass a list, and 10xGraph wraps it in a CompositePublisher. Every publisher receives every event, and a failure in one does not stop the others.
  - question: Why call aclose() at the end of a script?
    answer: Events are published from background tasks. aclose() waits for them to finish and closes the publisher connections, so short scripts do not exit with events still queued.
  - question: Does OtelPublisher send prompts and responses?
    answer: Only at ObservabilityLevel.FULL. SPANS sends timing and structure, STANDARD adds token counts and model names, and FULL adds prompts, responses and tool input and output, which may contain personal data.
---

# Observability and event publishers

## What you will build

Five scripts that send the events of a 10xGraph run to different places: your terminal, several sinks at once, OpenTelemetry, a message broker, and Logfire or LangSmith. Each script runs the same kind of OpenAI-backed ReAct agent with one small tool, so the only thing that changes is the publisher.

## Key concepts

- `BasePublisher`: base class for event sinks. Implement `publish`, `close` and `sync_close`.
- `ConsolePublisher`: prints events, or logs them to `tenxgraph.publisher` with `use_logger=True`.
- `CompositePublisher`: fans one event out to several publishers. A list passed as `publisher=` becomes one.
- `OtelPublisher` and `ObservabilityLevel`: map events to OpenTelemetry spans at `SPANS`, `STANDARD` or `FULL` detail.
- `RedisPublisher`, `KafkaPublisher`, `RabbitMQPublisher`: send JSON events to a broker from a config dict.
- `LogfirePublisher`, `LangsmithPublisher`: `OtelPublisher` subclasses that configure their own exporter.
- `app.aclose()`: flushes background publish tasks and closes the publishers.

### Event types

Each `EventModel` has an `event` (source), an `event_type` (phase) and a list of `content_type` values.

| Field | Values |
|---|---|
| `event` | `graph_execution`, `node_execution`, `llm_call`, `tool_execution`, `streaming`, `realtime` |
| `event_type` | `start`, `progress`, `result`, `end`, `update`, `error`, `interrupted` |
| `content_type` | `text`, `message`, `reasoning`, `tool_call`, `tool_result`, `image`, `audio`, `transcript`, `video`, `document`, `data`, `state`, `update`, `error` |

Events also carry `node_name`, `thread_id`, `run_id`, `user_id`, `timestamp`, `data` and `metadata`.

## Prerequisites

```bash
pip install "10xgraph[openai]" python-dotenv
export OPENAI_API_KEY="your-key"
```

Extras per script:

| Script | Extra | Other requirements |
|---|---|---|
| `console_publisher.py`, `composite_publisher.py` | none | none |
| `otel_tracing.py` | `otel` | none |
| `brokers.py` | `redis`, `kafka` or `rabbitmq` | a running broker |
| `logfire_langsmith.py` | `logfire` or `langsmith` | `LOGFIRE_TOKEN` or `LANGSMITH_API_KEY` |

For example: `pip install "10xgraph[openai,otel]"`. The `observability` extra installs OpenTelemetry, Logfire and LangSmith support together, and `all_publishers` installs Redis, Kafka and RabbitMQ.

Local brokers for `brokers.py`:

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7
    ports: ["6379:6379"]
  rabbitmq:
    image: rabbitmq:3-management
    ports: ["5672:5672", "15672:15672"]
  kafka:
    image: apache/kafka:3.8.0
    ports: ["9092:9092"]
```

Start the one you need, for example `docker compose up redis`.

## Run it

```bash
python examples/observability/console_publisher.py
python examples/observability/composite_publisher.py
python examples/observability/otel_tracing.py
BROKER=redis python examples/observability/brokers.py      # or kafka, rabbitmq
TRACE_BACKEND=logfire python examples/observability/logfire_langsmith.py   # or langsmith
```

## How it works

### Console

`ReactAgent` passes `publisher` to the graph it builds:

```python
agent = ReactAgent(
    model="gpt-5-mini",
    tools=[get_time],
    publisher=ConsolePublisher({"use_logger": True}),
)
```

With `use_logger=True` the events go to the `tenxgraph.publisher` logger, so standard logging configuration decides what you see.

### Composite

A list of publishers becomes a `CompositePublisher`. `EventCounter` is a custom `BasePublisher` that tallies `event/event_type` pairs:

```python
publishers: list[BasePublisher] = [ConsolePublisher({"use_logger": True}), counter]
agent = ReactAgent(model="gpt-5-mini", tools=[get_stock_price], publisher=publishers)
```

### OpenTelemetry

Configure a `TracerProvider` as you would for any OTEL application, then pass an `OtelPublisher`:

```python
publisher = OtelPublisher(level=ObservabilityLevel.STANDARD)
agent = ReactAgent(model="gpt-5-mini", tools=[get_inventory], publisher=publisher)
```

Pass the publisher when the graph is created. The library also exports `setup_tracing(graph)`, but in the current release it sets the publisher on a `StateGraph` after the graph's dependency container was already built, and no events reach it. Constructor injection works.

`OtelPublisher` also accepts `user_id_mode="hash"` or `"mask"` to keep raw user ids off spans.

### Brokers

`brokers.py` builds one publisher from a config dict, selected with `BROKER`. Redis supports `mode` `pubsub` or `stream`, Kafka takes `bootstrap_servers` and `topic`, and RabbitMQ takes `url`, `exchange` and `routing_key`. Connections open on first use, so a missing broker shows up as a publish error in the logs, not at import.

### Logfire and LangSmith

`LogfirePublisher(service_name=...)` and `LangsmithPublisher(project=...)` call the backend setup in their constructors and then behave like `OtelPublisher`. `setup_observability(graph, config)` builds the same setup from a `10xgraph.json`-style `observability` block for Logfire and LangSmith.

## Expected output

Illustrative output of `composite_publisher.py` (console lines trimmed):

```text
1730123456.1 -> Source: MAIN.start:-> Payload: {...} -> {...}
...
assistant: ACME is trading at 187.20 USD.

events seen by EventCounter:
  graph_execution/end              1
  graph_execution/start            1
  llm_call/end                     2
  llm_call/start                   2
  node_execution/end               3
  tool_execution/progress          1
```

## Using a different model

Change the model string and install the matching extra. For Gemini use `model="gemini-2.5-flash"` with `pip install "10xgraph[google-genai]"` and `GEMINI_API_KEY`. For Anthropic use `model="claude-sonnet-5-5"` with `pip install "10xgraph[anthropic]"` and `ANTHROPIC_API_KEY`. Publishers do not depend on the model.

## Next steps

- [Callbacks](../callbacks/) hooks into individual model and tool calls instead of the event stream.
- [Graceful shutdown](../graceful_shutdown/) covers flushing publishers and checkpointers on SIGTERM.
- [Streaming](../streaming/) streams chunks to the caller of a single run.
