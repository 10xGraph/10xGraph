---
title: Dependency Injection in 10xGraph Agents with InjectQ
description: Register your own services in an InjectQ container, receive them in tools and nodes with Inject[T], and swap them for tests with rebinding or override_node.
section: Examples
group: "Tools and integrations"
order: 70
label: Dependency injection
updated: "2026-10-28"
faq:
  - question: How do I give a tool access to a database client in 10xGraph?
    answer: Bind the client in an InjectQ container, pass the container to StateGraph or ReactAgent with container=..., and declare the tool parameter as client: Client = Inject[Client]. The parameter is filled at call time and does not appear in the tool schema the model sees.
  - question: Why call fresh() on an Inject default?
    answer: An Inject[T] default caches the first object it resolves for the whole process. fresh() resolves it from the active container on each call, so a rebound service or a graph with its own container is picked up.
  - question: How do I replace a service in tests?
    answer: Build a container that binds a fake implementation, or replace a whole node with graph.override_node("MAIN", fake_fn) before compile().
---

## What you will build

A small order-lookup agent. An `OrderRepository` (a fake database) and a `Settings` object live in an InjectQ container. A tool reads them through `Inject[...]` defaults, and a hand-written node does the same. A second script swaps the repository for a test double and replaces the model node, without calling an LLM.

## Key concepts

- `InjectQ()` and `bind_instance(Type, obj)`: your own container and the services in it.
- `StateGraph(container=...)` and `ReactAgent(container=...)`: make the graph use that container.
- `Inject[T]`: a parameter default that is filled from the container. Inject parameters are not part of the tool schema.
- `fresh(value)` from `tenxgraph.utils.injection`: resolve an `Inject[...]` default from the active container on every call.
- `graph.override_node(name, fn)`: replace a node before `compile()`, for tests.

## Prerequisites

```bash
pip install "10xgraph[openai]" python-dotenv
export OPENAI_API_KEY=...    # only for di.py
```

`nodes.py` needs no key.

## Run it

```bash
python examples/dependency_injection/di.py
python examples/dependency_injection/nodes.py
```

## How it works

`di.py` builds the container and registers the services:

```python
container = InjectQ()
container.bind_instance(Settings, Settings())
container.bind_instance(OrderRepository, repo or OrderRepository())
```

The tool declares what it needs. The model supplies only `customer`:

```python
def list_orders(
    customer: str,
    repo: OrderRepository = Inject[OrderRepository],
    settings: Settings = Inject[Settings],
) -> str:
    repo, settings = fresh(repo), fresh(settings)
```

The agent receives the container with `ReactAgent(model="gpt-5-mini", tools=[list_orders], container=container)`.

`nodes.py` shows the same for a graph node (`planner(state, config, settings=Inject[Settings])`) with a scripted node standing in for the model. It runs three times:

1. With the real `OrderRepository`.
2. With `build_container(repo=FakeRepository())`, which binds a test double. The tool code is unchanged.
3. With `graph.override_node("MAIN", ...)` replacing the model node by a stub.

Each graph gets its own container, so the runs do not affect each other. That is also why `fresh()` is needed in functions that use `Inject` defaults.

## Expected output

Illustrative, from `nodes.py`:

```text
real tool result: A-100: 42.00 USD; A-101: 17.50 USD
real assistant: Done. Amounts are in USD.
fake-repo tool result: TEST-1: 1.00 USD
fake-repo assistant: Done. Amounts are in USD.
override_node assistant: stub
```

## Using a different model

`di.py` uses `gpt-5-mini`. Change the string to `gemini-2.5-flash` (`pip install "10xgraph[google-genai]"`) or `claude-sonnet-5-5` (`pip install "10xgraph[anthropic]"`) and set that key. The container code does not change.

## Next steps

- [`../testing/`](../testing/) for test helpers that avoid live LLM calls.
- [`../tools/`](../tools/) for tool definitions.
- [`../mcp/`](../mcp/) for tools that live in another process.
