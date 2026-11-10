# 10xGraph

**Write your AI logic. 10xGraph runs the infrastructure.**

[![CI](https://github.com/10xGraph/10xGraph/actions/workflows/ci.yml/badge.svg)](https://github.com/10xGraph/10xGraph/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/10xGraph/10xGraph)](https://github.com/10xGraph/10xGraph/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/10xgraph?color=blue)](https://pypi.org/project/10xgraph/)
[![Python](https://img.shields.io/pypi/pyversions/10xgraph)](https://pypi.org/project/10xgraph/)
[![License](https://img.shields.io/github/license/10xGraph/10xGraph)](https://github.com/10xGraph/10xGraph/blob/main/LICENSE)

10xGraph is open-source, self-hosted infrastructure for AI applications. You write the agents, tools and tasks; 10xGraph runs them behind an API server with authentication, role-based access, tenant isolation, durable state in Postgres and Redis, background jobs, metrics and tracing, and generates Docker and Kubernetes deploy files. MIT licensed, by 10xScale. Formerly Agentflow.

This repository is the core library of that infrastructure (PyPI `10xgraph`, import `tenxgraph`). The API server is [`10xgraph-api`](https://github.com/10xGraph/10xgraph-api) and the TypeScript client is [`10xgraph-client`](https://github.com/10xGraph/10xgraph-client). Together they are one stack. Docs: [10xgraph.com](https://10xgraph.com).

---

## What this package gives you

- **Graph engine.** Build workflows as a `StateGraph` of nodes, tools and conditional edges, or start from prebuilt agents. Multi-agent patterns (swarm, supervisor team, handoff) are included as features. [State graph](https://10xgraph.com/docs/concepts/state-graph), [prebuilt agents](https://10xgraph.com/docs/guides/prebuilt-agents)
- **Durable state and checkpointing.** `PgCheckpointer` keeps threads in Postgres with Redis as a hot cache and versioned compare-and-swap writes, so two runs on one thread cannot overwrite each other. `SqliteCheckpointer` and `InMemoryCheckpointer` cover local work. [Checkpointing and threads](https://10xgraph.com/docs/concepts/checkpointing-and-threads)
- **Tenant columns and row-level security.** Declare `tenant_columns` and `extra_columns` on the checkpointer and store. They become real indexed columns, filled from the verified identity in `config["user"]`, never from client input. Native Postgres row-level security is opt-in. [Multi-tenant data](https://10xgraph.com/docs/guides/multi-tenant-data), [tenancy and access](https://10xgraph.com/docs/concepts/tenancy-and-access)
- **Long-term memory.** `PgVectorStore` keeps memory and RAG in the same Postgres (similarity, full-text or hybrid search). `QdrantStore` is there if you want a dedicated vector service. [Use PgVectorStore](https://10xgraph.com/docs/guides/use-pgvector-store), [memory and store](https://10xgraph.com/docs/concepts/memory-and-store)
- **Replay-safe tools.** The run loop persists the current node before running it. Before a tool runs, the checkpointer's tool ledger is checked, and each completed call is recorded as soon as it returns, so a tool that already ran is not executed again on resume. Requires a checkpointer. Limit: a crash before a tool returns, or in the short gap before its record is written, can repeat the tool, so send an idempotency key derived from stable inputs for side effects like payments. [Replay-safe tools](https://10xgraph.com/docs/concepts/replay-safe-tools), [limits](https://10xgraph.com/docs/production/limits)
- **Tools and MCP.** Plain Python functions as tools, run in parallel, with dependency injection through InjectQ and native Model Context Protocol support. Remote tools run in the user's browser or client. `interrupt()` pauses a run for human approval. [Use MCP](https://10xgraph.com/docs/guides/use-mcp), [remote tools](https://10xgraph.com/docs/concepts/remote-tools), [human approval](https://10xgraph.com/docs/guides/add-human-approval)
- **Timeouts and graceful shutdown.** Set `node_timeout` and `tool_timeout` in the run config (no limit unless you set them). `aclose()` refuses new runs, lets active runs finish or saves them as interrupted. [Graceful shutdown](https://10xgraph.com/docs/guides/graceful-shutdown)
- **Publishers and tracing.** Events to Kafka, Redis Pub/Sub, RabbitMQ and OpenTelemetry; tracing to Logfire, LangSmith, Phoenix or any OTLP endpoint. [Publishers](https://10xgraph.com/docs/guides/use-publishers), [Phoenix](https://10xgraph.com/docs/guides/send-traces-to-phoenix), [Logfire and LangSmith](https://10xgraph.com/docs/guides/send-traces-to-logfire-langsmith), [OTLP](https://10xgraph.com/docs/guides/send-traces-to-otlp)
- **Evaluation and testing.** An eval framework (LLM-as-judge, trajectory matching, rubrics, user simulation, JSON, HTML and JUnit reports) and `TestAgent` with mocks for unit tests without live LLM calls. [Testing and evaluation](https://10xgraph.com/docs/testing)
- **Any model.** OpenAI and OpenAI-compatible endpoints, Google Gemini (including Vertex AI), Anthropic (direct, Vertex AI or Bedrock). Changing the model string does not change the graph or the tools. No LangChain dependency.
- **Also included:** structured output, skills (Agent Skills spec), streaming, [realtime audio](https://10xgraph.com/docs/guides/use-realtime-audio) over Gemini Live, multimodal media handling.

---

## Install

```bash
pip install 10xgraph
```

Provider SDKs and infrastructure integrations are optional extras. Install only what you use:

| Extra | Adds |
|---|---|
| `google-genai`, `openai`, `anthropic` | Provider SDK adapters |
| `anthropic-vertex`, `anthropic-bedrock` | Claude on Vertex AI or Amazon Bedrock |
| `realtime` | Audio-to-audio agents over Gemini Live |
| `mcp` | Model Context Protocol client and tools |
| `pg_checkpoint`, `sqlite_checkpoint` | Durable checkpointing (Postgres + Redis, or SQLite) |
| `pgvector`, `qdrant` | Long-term memory store (Postgres pgvector, or Qdrant) |
| `redis`, `kafka`, `rabbitmq`, `all_publishers` | Event publishers |
| `otel`, `logfire`, `langsmith`, `phoenix`, `otlp`, `observability` | Tracing and OpenTelemetry export |
| `images`, `cloud-storage` | Multimodal media handling and offload |
| `all` | Every extra at once, for development and CI |

```bash
pip install "10xgraph[openai,pg_checkpoint,pgvector]"
```

To serve your graph over HTTP with auth, jobs, metrics and deploy files, add the server package:

```bash
pip install 10xgraph-api
```

Set your provider key; a `.env` file in the working directory is loaded automatically:

```bash
export GEMINI_API_KEY=...        # Google Gemini
export OPENAI_API_KEY=sk-...     # OpenAI, or any OpenAI-compatible endpoint
export ANTHROPIC_API_KEY=sk-...  # Anthropic Claude
```

Requires Python 3.12 or newer. Tested on 3.12, 3.13, 3.14 and 3.15. Full guide: [installation](https://10xgraph.com/docs/get-started/installation).

---

## Quick start: a support agent with durable state

`lookup_order` reads data. `refund_order` moves money, so it must run once per request. Compiled with a checkpointer, a completed tool call is recorded and not re-run after a crash or retry.

```python
# graph/agent.py
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import PgCheckpointer

def lookup_order(order_id: str) -> dict:
    """Look up an order by id and return its status and total."""
    return {"order_id": order_id, "status": "delivered", "total": 59.0}

def refund_order(order_id: str, amount: float) -> str:
    """Refund an order. This moves money, so it must run once per request."""
    return f"Refunded {amount:.2f} for order {order_id}"

checkpointer = PgCheckpointer(postgres_dsn="postgresql://...", redis_url="redis://...")
app = ReactAgent(
    model="claude-sonnet-5-5",
    tools=[lookup_order, refund_order],
).compile(checkpointer=checkpointer)
```

Call it directly from Python:

```python
from tenxgraph.core.state import Message

result = app.invoke(
    {"messages": [Message.text_message("Order A-1001 arrived broken, please refund it.")]},
    config={"thread_id": "1"},
)
```

Swap `ReactAgent` for `RAGAgent`, `SwarmAgent`, `SupervisorTeamAgent` or `PlanActReflectAgent` and the shape stays the same. For local work use `InMemoryCheckpointer` or `SqliteCheckpointer` instead of Postgres.

### Serve it with `10xgraph dev`

Point a config file at the compiled graph:

```json
{
  "agent": "graph.agent:app",
  "env": ".env",
  "auth": "jwt"
}
```

Save it as `10xgraph.json`, then run the server with the playground:

```bash
pip install 10xgraph-api
10xgraph dev
```

From those two files you get REST and streaming endpoints, WebSocket, thread, memory and file APIs, health probes and the playground. `10xgraph build --docker-compose --k8s` writes the Dockerfile, Compose file and Kubernetes manifest. For a scaffold with auth on, a Redis cache store, rate limits and tests, run `10xgraph init --template production`. See [Why 10xGraph](https://10xgraph.com/docs/get-started/why-10xgraph) and the [server docs](https://10xgraph.com/docs/server).

---

## Building your own graph

Prebuilt agents are graphs. When you need custom control flow, build one directly:

```python
from tenxgraph.core.graph import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState
from tenxgraph.utils.constants import END

graph = StateGraph()
graph.add_node("MAIN", Agent(
    model="gemini/gemini-2.5-flash",
    system_prompt=[{"role": "system", "content": "You are a support agent."}],
    tool_node="TOOL",
))
graph.add_node("TOOL", ToolNode([lookup_order, refund_order]))


def route(state: AgentState) -> str:
    if state.context and state.context[-1].tools_calls:
        return "TOOL"
    return END


graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile()
```

Nodes can be plain functions too, so you can call a provider SDK directly. See [Build a graph](https://10xgraph.com/docs/guides/build-a-graph).

---

## One stack

| Package | Role | Install | Source |
|---|---|---|---|
| `10xgraph` (this repository) | Core library: graph engine, durable state and checkpointing, long-term memory, replay-safe tools, MCP, publishers and tracing, evaluation | `pip install 10xgraph` | [10xGraph/10xGraph](https://github.com/10xGraph/10xGraph) |
| `10xgraph-api` (formerly `10xscale-agentflow-cli`) | API server and CLI: authentication, roles with reach, tenant isolation, cache store, background jobs, metrics, audit log, Docker and Kubernetes files | `pip install 10xgraph-api` | [10xGraph/10xgraph-api](https://github.com/10xGraph/10xgraph-api) |
| `10xgraph-client` (formerly `@10xscale/agentflow-client`) | Typed TypeScript client: invoke, stream, threads, memory, files, client-side tools | `npm install 10xgraph-client` | [10xGraph/10xgraph-client](https://github.com/10xGraph/10xgraph-client) |
| Docs | Tutorials, guides, concepts, reference | [10xgraph.com](https://10xgraph.com) | [10xGraph/10xgraph-docs](https://github.com/10xGraph/10xgraph-docs) |

The playground is a React UI that `10xgraph dev` opens against your local server ([10xHub/agentflow-playground](https://github.com/10xHub/agentflow-playground)).

Built for teams going through SOC 2 and ISO 27001 audits: see the [control mapping](https://10xgraph.com/docs/production/control-mapping) for which controls the stack supports and which stay with you.

---

## Examples

Runnable scripts in [`examples/`](https://github.com/10xGraph/10xGraph/tree/main/examples) (index in [`examples/README.md`](https://github.com/10xGraph/10xGraph/blob/main/examples/README.md)):

| Topic | Directory |
|---|---|
| Quickstart, graph basics, ReAct agents | [`quickstart/`](https://github.com/10xGraph/10xGraph/tree/main/examples/quickstart), [`graph_basics/`](https://github.com/10xGraph/10xGraph/tree/main/examples/graph_basics), [`react/`](https://github.com/10xGraph/10xGraph/tree/main/examples/react) |
| Streaming | [`streaming/`](https://github.com/10xGraph/10xGraph/tree/main/examples/streaming), [`react_stream/`](https://github.com/10xGraph/10xGraph/tree/main/examples/react_stream) |
| Tools, injection, remote tools | [`tools/`](https://github.com/10xGraph/10xGraph/tree/main/examples/tools), [`dependency_injection/`](https://github.com/10xGraph/10xGraph/tree/main/examples/dependency_injection) |
| MCP servers and tools | [`mcp/`](https://github.com/10xGraph/10xGraph/tree/main/examples/mcp), [`react-mcp/`](https://github.com/10xGraph/10xGraph/tree/main/examples/react-mcp) |
| Persistence and checkpointing | [`persistence/`](https://github.com/10xGraph/10xGraph/tree/main/examples/persistence), [`checkpointer/`](https://github.com/10xGraph/10xGraph/tree/main/examples/checkpointer) |
| Long-term memory, RAG, vector stores | [`long_term_memory/`](https://github.com/10xGraph/10xGraph/tree/main/examples/long_term_memory), [`rag/`](https://github.com/10xGraph/10xGraph/tree/main/examples/rag) |
| Multi-agent: swarm, supervisor, handoff, plan-act-reflect | [`multi_agent/`](https://github.com/10xGraph/10xGraph/tree/main/examples/multi_agent) |
| Human approval | [`human_in_the_loop/`](https://github.com/10xGraph/10xGraph/tree/main/examples/human_in_the_loop) |
| Resilience, graceful shutdown | [`resilience/`](https://github.com/10xGraph/10xGraph/tree/main/examples/resilience), [`graceful_shutdown/`](https://github.com/10xGraph/10xGraph/tree/main/examples/graceful_shutdown) |
| Publishers and tracing | [`observability/`](https://github.com/10xGraph/10xGraph/tree/main/examples/observability) |
| Serving over the API | [`serving/`](https://github.com/10xGraph/10xGraph/tree/main/examples/serving) |
| Realtime audio, multimodal | [`realtime/`](https://github.com/10xGraph/10xGraph/tree/main/examples/realtime), [`multimodal/`](https://github.com/10xGraph/10xGraph/tree/main/examples/multimodal) |
| Structured output, skills | [`structured_output/`](https://github.com/10xGraph/10xGraph/tree/main/examples/structured_output), [`skills/`](https://github.com/10xGraph/10xGraph/tree/main/examples/skills) |
| Providers | [`providers/`](https://github.com/10xGraph/10xGraph/tree/main/examples/providers) |
| Evaluation and testing | [`evaluation/`](https://github.com/10xGraph/10xGraph/tree/main/examples/evaluation), [`testing/`](https://github.com/10xGraph/10xGraph/tree/main/examples/testing) |

Run one:

```bash
export GEMINI_API_KEY=...   # or OPENAI_API_KEY
python examples/quickstart/quickstart.py
```

---

## Limits

- No hosted service. You run it, or nobody does.
- Authentication in the API server is off until you configure the `auth` key. The production template (`10xgraph init --template production`) turns it on.
- Tenant isolation needs columns you declare on the checkpointer and store. It is not automatic.
- A crash before a tool returns, or just before its record is written, can repeat that tool. Use idempotency keys for side effects.
- Automatic per-tool permissions (user A may call `refund`, user B may not) are not built in. Tools can check the caller with `tenxgraph.core.authz.can(config, "resource:action")`.
- Smaller community and fewer integrations than LangGraph or CrewAI. LangGraph has stronger visual tooling.
- Requires Python 3.12 or newer. Code-first, not a no-code builder.
- 10xGraph supports your SOC 2 and ISO 27001 controls; it does not hold an audit report of its own.

More: [production limits](https://10xgraph.com/docs/production/limits).

---

## Formerly Agentflow

10xGraph is the new name of Agentflow. The license and maintainers are the same. `10xscale-agentflow` 0.10.1 is the last release under the old name. See [coming from Agentflow](https://10xgraph.com/docs/get-started/coming-from-agentflow).

```bash
pip uninstall 10xscale-agentflow
pip install 10xgraph
```

Uninstall first: both distributions provide the `agentflow` module and must not be installed side by side.

```python
# before
from agentflow.core.graph import StateGraph
# after
from tenxgraph import StateGraph
```

The import name is `tenxgraph` because a Python identifier cannot start with a digit. Canonical paths are the old ones with `agentflow` replaced by `tenxgraph` (for example `tenxgraph.core.graph`, `tenxgraph.storage.checkpointer`, `tenxgraph.prebuilt.agent`). `import agentflow` keeps working as a deprecated alias until 2.0, with one `DeprecationWarning`.

| Item | Old | New |
|---|---|---|
| PyPI core | `10xscale-agentflow` | `10xgraph` |
| PyPI API and CLI | `10xscale-agentflow-cli` | `10xgraph-api` |
| npm client | `@10xscale/agentflow-client` | `10xgraph-client` |
| OpenTelemetry tracer and meter name, `GEN_AI_SYSTEM` | `agentflow` | `10xgraph` |
| Logger names | `agentflow.*` | `tenxgraph.*` |
| Media URI scheme | `agentflow://media/` | `graph://media/` (old URIs still read) |
| Default home directory | `~/.agentflow` | `~/.10xgraph` (falls back to `~/.agentflow` if only that exists) |
| Cloud media prefix | `agentflow-media` | `10xgraph-media` (old objects still read) |
| Server config file | `agentflow.json` | `10xgraph.json` (the CLI falls back to `agentflow.json`) |
| CLI command | `agentflow` | `10xgraph` (`agentflow` stays as a deprecated alias until 2.0) |

---

## Contributing

**Your avatar belongs on this wall.**

<a href="https://github.com/10xGraph/10xGraph/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=10xGraph/10xGraph" alt="People who have contributed to 10xGraph" />
</a>

Merge one pull request and you join them, here and on the contributor page at [10xgraph.com/maintainers](https://10xgraph.com/maintainers).

Your first pull request can be small:

- **Type one module.** `pyproject.toml` lists the modules `mypy` still skips. Pick one, fix its types, delete its line.
- **Add the example you wish had existed.** A runnable script in [`examples/`](https://github.com/10xGraph/10xGraph/tree/main/examples) for a real use case: a support agent, a research pipeline, an approval flow.
- **Turn a bug into a failing test.** Open it as a draft pull request; the fix can come later.
- **Fix the docs where you got stuck.** Every page on [10xgraph.com](https://10xgraph.com) has an "Edit this page" link.

From clone to a passing check:

```bash
git clone https://github.com/10xGraph/10xGraph.git
cd 10xGraph
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy tenxgraph/
```

For bigger changes, start a thread in [Discussions](https://github.com/10xGraph/10xGraph/discussions) first. [CONTRIBUTING.md](https://github.com/10xGraph/10xGraph/blob/main/CONTRIBUTING.md) has the full workflow, and the [Code of Conduct](https://github.com/10xGraph/10xGraph/blob/main/CODE_OF_CONDUCT.md) applies everywhere.

---

## Security

Found a vulnerability? Do not open a public issue. Follow the process in [SECURITY.md](https://github.com/10xGraph/10xGraph/blob/main/SECURITY.md).

---

## License

10xGraph is [MIT licensed](https://github.com/10xGraph/10xGraph/blob/main/LICENSE) and made by [10xScale](https://10xscale.ai). Copyright 10xScale. Contributions are accepted under the same license.

---

## Links

- Documentation: [10xgraph.com](https://10xgraph.com)
- PyPI: [`10xgraph`](https://pypi.org/project/10xgraph/), [`10xgraph-api`](https://pypi.org/project/10xgraph-api/)
- npm: [`10xgraph-client`](https://www.npmjs.com/package/10xgraph-client)
- [Issues](https://github.com/10xGraph/10xGraph/issues) and [Discussions](https://github.com/10xGraph/10xGraph/discussions)
- [Changelog](https://github.com/10xGraph/10xGraph/blob/main/CHANGELOG.md)
- [Examples](https://github.com/10xGraph/10xGraph/tree/main/examples)
