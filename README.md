# 10xGraph

*Formerly Agentflow.* 10xGraph by 10xScale: graph engineering for production AI agents.

[![CI](https://github.com/10xGraph/10xGraph/actions/workflows/ci.yml/badge.svg)](https://github.com/10xGraph/10xGraph/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/10xGraph/10xGraph)](https://github.com/10xGraph/10xGraph/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/10xgraph?color=blue)](https://pypi.org/project/10xgraph/)
[![Python](https://img.shields.io/pypi/pyversions/10xgraph)](https://pypi.org/project/10xgraph/)
[![License](https://img.shields.io/github/license/10xGraph/10xGraph)](https://github.com/10xGraph/10xGraph/blob/main/LICENSE)

10xGraph is an open-source Python framework for building multi-agent AI systems and running them in production. You write the agent as a graph of nodes and tools. 10xGraph keeps tool calls from running twice after a crash, guards state writes, enforces timeouts, and, with the `10xgraph-api` package, generates the API server around the graph.

This repository is the core engine (PyPI `10xgraph`, import `tenxgraph`). The docs live at [10xgraph.com](https://10xgraph.com).

---

## What it gives you

**1. Correct under failure (this package)**

- **Replay-safe tools.** The run loop persists the current node before running it, so a process killed mid-node re-runs that node on resume. Before calling a tool, 10xGraph checks the checkpointer's tool ledger and records each completed call as soon as it returns. A tool that already ran is not executed again: no double charge, no duplicate email. Requires a checkpointer.
- **Versioned state writes.** Durable writes use optimistic compare-and-swap, so two runs on one thread cannot overwrite each other. The Redis cache write is version-guarded too.
- **Real node and tool timeouts.** Set `node_timeout` and `tool_timeout` in the run config (defaults 900 s and 300 s), so a hung tool cannot hold a worker forever.
- **Human approval inside a tool.** `interrupt()` pauses the run and saves the thread; resume with the decision.

**2. The production server ships in the box (`10xgraph-api`, MIT)**

`10xgraph-api` generates the production server around your compiled graph: REST, SSE streaming, WebSocket and realtime-audio endpoints; JWT or custom auth; scoped authorization on every endpoint; thread ownership isolation; rate limiting; and Docker Compose and Kubernetes files. You drive it with the `10xgraph` command (`10xgraph init`, `10xgraph api`, `10xgraph build`). See [the ecosystem table](#ecosystem).

**3. Built to scale**

- **Two-tier persistence.** `PgCheckpointer` caches active thread state in Redis and reads it first (default TTL 24 hours); PostgreSQL holds the durable history with versioned writes. Threads survive restarts and cache expiry. `SqliteCheckpointer` covers local work.
- **Event publishing** to Kafka, Redis Pub/Sub, RabbitMQ and OpenTelemetry.
- **Long-term memory** in Qdrant or Mem0.

**4. One stack, backend to frontend**

- **Remote tools.** The model can call tools that run in the user's browser or client. Declare them on the graph or pass them per run in `config["remote_tools"]`; the run pauses until the client returns the result.
- **Typed TypeScript client** (`10xgraph-client`) for invoke, stream, threads, memory and files, plus a React playground (`10xgraph play`).

**5. You own it**

- MIT licensed and self-hosted. The server layer is part of the same open-source project, not a paid platform.
- No LangChain dependency. Core requires InjectQ, Pydantic, Pillow, PyYAML and python-dotenv; everything else is an optional extra.
- Any model: OpenAI and OpenAI-compatible endpoints, Google Gemini (including Vertex AI), Anthropic (direct, Vertex AI or Bedrock). Changing the model string does not change the graph or the tools.
- Built and run in production by 10xScale for its own AI products.

**Also included** (standard for agent frameworks, listed as facts): graph orchestration and the ReAct tool-calling loop, parallel tool execution, OpenAI, Google Gemini and Anthropic support, MCP tools, streaming, and checkpointing to a database.

---

## When it fits

| You are | The problem | What 10xGraph does |
|---|---|---|
| A Python team taking an agent to production | Server, auth, persistence and deployment all have to be built around the agent | `10xgraph-api` generates them from the graph |
| Running agents with side effects (payments, email, tickets) | A retry or crash repeats an action | Tool ledger: a completed tool call is replayed from the checkpointer, not re-run |
| Building a multi-user product | Users must not see each other's threads | Thread ownership isolation and scoped authorization on every endpoint |
| Required to self-host | Paid platforms, data residency, lock-in | MIT, self-hosted, any model |
| A Python backend with a TypeScript frontend | Hand-written SSE and client glue | Typed client and remote tools |

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
| `qdrant`, `mem0` | Long-term vector memory stores |
| `redis`, `kafka`, `rabbitmq`, `otel` | Event publishers and tracing |
| `images`, `cloud-storage` | Multimodal media handling and offload |
| `all` | Every extra above at once, for development and CI |

```bash
pip install "10xgraph[google-genai,openai,anthropic,mcp,pg_checkpoint]"
```

Then set your provider key. A `.env` file in the working directory is loaded automatically.

```bash
export GEMINI_API_KEY=...        # Google Gemini
export OPENAI_API_KEY=sk-...     # OpenAI, or any OpenAI-compatible endpoint
export ANTHROPIC_API_KEY=sk-...  # Anthropic Claude
```

Requires Python 3.12 or newer.

---

## Quick start: a support agent with an approval step

`lookup_order` reads data. `refund_order` moves money, so it pauses for a human decision with `interrupt()` before it acts. With a checkpointer, a crash or retry after the refund does not issue it twice.

```python
from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import interrupt


def lookup_order(order_id: str) -> dict:
    """Look up an order by id."""
    return {"order_id": order_id, "status": "delivered", "total": 42.00}


def refund_order(order_id: str, amount: float) -> str:
    """Refund an order. Requires approval."""
    decision = interrupt({"amount": amount}, message=f"Refund ${amount}?")
    if not decision.get("approved"):
        return "Refund declined by reviewer."
    return f"Refunded ${amount} for order {order_id}."


app = ReactAgent(
    model="gemini/gemini-2.5-flash",
    system_prompt=[{"role": "system", "content": "You are a support agent for an online store."}],
    tools=[lookup_order, refund_order],
).compile(checkpointer=InMemoryCheckpointer())

config = {"thread_id": "1"}
result = app.invoke(
    {"messages": [Message.text_message("Order A-1001 arrived broken, please refund it.")]},
    config=config,
)

# The run pauses inside refund_order. After a reviewer approves:
result = app.invoke({"resume": {"approved": True}}, config)
```

Swap `ReactAgent` for `RAGAgent`, `SwarmAgent`, `SupervisorTeamAgent` or `PlanActReflectAgent` and the shape stays the same. Use `PgCheckpointer` (Postgres plus Redis) in production, or `SqliteCheckpointer` for local work.

**Stream it:**

```python
async for chunk in app.astream(
    {"messages": [Message.text_message("Where is order A-1001?")]},
    config={"thread_id": "2"},
):
    print(chunk.model_dump())
```

**Add MCP tools** by passing a `fastmcp` client; remote tools join your local ones:

```python
from fastmcp import Client

mcp_client = Client({
    "mcpServers": {
        "orders": {"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"},
    }
})

app = ReactAgent(
    model="gemini/gemini-2.5-flash",
    tools=[lookup_order],
    client=mcp_client,
).compile()
```

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

Nodes can be plain functions too, so you can call a provider SDK directly. See [`examples/react/`](https://github.com/10xGraph/10xGraph/tree/main/examples/react).

---

## Realtime Audio Agents

Live audio-to-audio sessions over Gemini Live. The provider owns the turn loop, so the session runs
through `arealtime`: you push into an input queue and consume normalized events.

```python
from tenxgraph.prebuilt.agent import AudioAgent
from tenxgraph.core.realtime import LiveInputQueue, RealtimeConfig

app = AudioAgent(
    "gemini-live-2.5-flash-preview",
    realtime_config=RealtimeConfig(model="gemini-live-2.5-flash-preview", voice="Puck"),
    tools=[lookup_order],
).compile()

queue = LiveInputQueue()
queue.send_audio(pcm16_bytes)   # non-blocking, safe to call from an audio callback

async for event in app.arealtime(queue, {"thread_id": "t1"}):
    ...                         # AudioDeltaEvent / transcripts / ToolCallEvent / ...

queue.close()
```

Barge-in, persisted transcripts (raw audio is never stored), automatic reconnect with session
resumption, and image/video frame input are handled for you. `system_prompt`, `skills`, and `memory`
work as they do on any other agent. Needs ``pip install "10xgraph[realtime]"``.

---

## Prebuilt agents and patterns

- **Agents:** React, RAG, Guarded, Plan-Act-Reflect, Audio, Swarm, SupervisorTeam, StructuredOutput
- **Orchestration:** Router, MapReduce, Sequential, Branch-Join
- **Other:** dependency injection through InjectQ, skills (Agent Skills spec), 3-layer memory (working state, checkpointer, vector stores such as Qdrant and Mem0), publishers (Console, Redis, Kafka, RabbitMQ, OpenTelemetry), evaluation and testing helpers

---

## Moving from Agentflow

10xGraph is the new name of Agentflow. The framework, license and maintainers are the same. `10xscale-agentflow` 0.10.1 is its last release.

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

The import name is `tenxgraph` because a Python identifier cannot start with a digit. All canonical paths are the old ones with `agentflow` replaced by `tenxgraph` (for example `tenxgraph.core.graph`, `tenxgraph.storage.checkpointer`, `tenxgraph.prebuilt.agent`). `import agentflow` keeps working as a deprecated alias until 2.0, with one `DeprecationWarning`.

Other renamed identifiers (old values still work where noted):

| Item | Old | New |
|---|---|---|
| OpenTelemetry tracer and meter name, `GEN_AI_SYSTEM` | `agentflow` | `10xgraph` |
| Logger names | `agentflow.*` | `tenxgraph.*` |
| Media URI scheme | `agentflow://media/` | `graph://media/` (old URIs still read) |
| Default home directory | `~/.agentflow` | `~/.10xgraph` (falls back to `~/.agentflow` if only that exists) |
| Cloud media prefix | `agentflow-media` | `10xgraph-media` (old objects still read) |
| Prebuilt tools user-agent | `agentflow-prebuilt-tools` | `10xgraph-prebuilt-tools/1.0` |
| Server config file | `agentflow.json` | `10xgraph.json` (the CLI falls back to `agentflow.json`) |
| CLI command | `agentflow` | `10xgraph` (`agentflow` stays as a deprecated alias until 2.0) |

---

## Ecosystem

| Package | What it does | Install | Source |
|---|---|---|---|
| Core framework, `10xgraph` | Graph engine, state and checkpointing, memory, tools, MCP, publishers, evaluation | `pip install 10xgraph` | this repository |
| API server, `10xgraph-api` (formerly `10xscale-agentflow-cli`) | Generates the production server around your graph: REST, SSE, WebSocket, JWT auth, scoped authorization, rate limiting, Docker and Kubernetes files | `pip install 10xgraph-api` | [10xGraph/10xgraph-api](https://github.com/10xGraph/10xgraph-api) |
| TypeScript client, `10xgraph-client` (formerly `@10xscale/agentflow-client`) | Typed client for every endpoint, React streaming hooks, client-side tools | `npm install 10xgraph-client` | [10xGraph/10xgraph-client](https://github.com/10xGraph/10xgraph-client) |
| Playground | React UI to chat with agents and inspect graphs, threads and state | `10xgraph play` | [10xHub/agentflow-playground](https://github.com/10xHub/agentflow-playground) |
| Documentation | Tutorials, guides, concepts, reference | [10xgraph.com](https://10xgraph.com) | [10xGraph/10xgraph-docs](https://github.com/10xGraph/10xgraph-docs) |

From install to a running service:

```bash
pip install 10xgraph-api         # pulls in 10xgraph
10xgraph init --path my-agent && cd my-agent
10xgraph api                     # REST and WebSocket API on :8000
10xgraph play                    # server plus playground
10xgraph build --docker-compose --k8s
```

A production scaffold with JWT auth and Redis rate limiting:

```bash
10xgraph init --path my-agent --yes --template production --auth jwt --rate-limit redis
```

---

## Examples

Runnable scripts in [`examples/`](https://github.com/10xGraph/10xGraph/tree/main/examples):

| Topic | Directory |
|---|---|
| React agents, sync and class-based | [`react/`](https://github.com/10xGraph/10xGraph/tree/main/examples/react), [`react-injection/`](https://github.com/10xGraph/10xGraph/tree/main/examples/react-injection), [`agent-class/`](https://github.com/10xGraph/10xGraph/tree/main/examples/agent-class), [`tool-decorator/`](https://github.com/10xGraph/10xGraph/tree/main/examples/tool-decorator) |
| Streaming and stop/resume | [`react_stream/`](https://github.com/10xGraph/10xGraph/tree/main/examples/react_stream) |
| MCP servers and tools | [`react-mcp/`](https://github.com/10xGraph/10xGraph/tree/main/examples/react-mcp), [`github-mcp/`](https://github.com/10xGraph/10xGraph/tree/main/examples/github-mcp), [`xquik-mcp/`](https://github.com/10xGraph/10xGraph/tree/main/examples/xquik-mcp) |
| RAG, memory, and vector stores | [`rag/`](https://github.com/10xGraph/10xGraph/tree/main/examples/rag), [`memory/`](https://github.com/10xGraph/10xGraph/tree/main/examples/memory), [`store/`](https://github.com/10xGraph/10xGraph/tree/main/examples/store) |
| Multi-agent: swarm, supervisor, handoff, plan-act-reflect | [`swarm/`](https://github.com/10xGraph/10xGraph/tree/main/examples/swarm), [`supervisor_team/`](https://github.com/10xGraph/10xGraph/tree/main/examples/supervisor_team), [`handoff/`](https://github.com/10xGraph/10xGraph/tree/main/examples/handoff), [`multiagent/`](https://github.com/10xGraph/10xGraph/tree/main/examples/multiagent), [`plan_act_reflect/`](https://github.com/10xGraph/10xGraph/tree/main/examples/plan_act_reflect) |
| Realtime audio, multimodal | [`realtime/`](https://github.com/10xGraph/10xGraph/tree/main/examples/realtime), [`multimodal/`](https://github.com/10xGraph/10xGraph/tree/main/examples/multimodal) |
| Structured output, skills, custom state | [`structured_output/`](https://github.com/10xGraph/10xGraph/tree/main/examples/structured_output), [`skills/`](https://github.com/10xGraph/10xGraph/tree/main/examples/skills), [`custom-state/`](https://github.com/10xGraph/10xGraph/tree/main/examples/custom-state) |
| Providers and A2A | [`providers/`](https://github.com/10xGraph/10xGraph/tree/main/examples/providers), [`a2a_sdk/`](https://github.com/10xGraph/10xGraph/tree/main/examples/a2a_sdk) |
| Checkpointing, graceful shutdown | [`checkpointer/`](https://github.com/10xGraph/10xGraph/tree/main/examples/checkpointer), [`graceful_shutdown/`](https://github.com/10xGraph/10xGraph/tree/main/examples/graceful_shutdown) |
| Evaluation and testing | [`evaluation/`](https://github.com/10xGraph/10xGraph/tree/main/examples/evaluation), [`testing/`](https://github.com/10xGraph/10xGraph/tree/main/examples/testing) |

Run one:

```bash
export GEMINI_API_KEY=...   # or OPENAI_API_KEY
python examples/react/react_single_class.py
```

Some examples still use pre-rename import paths; the canonical paths are listed in [CLAUDE.md](https://github.com/10xGraph/10xGraph/blob/main/CLAUDE.md).

---

## Limitations

- Smaller community and fewer integrations than LangGraph or CrewAI.
- Pre-1.0 (current release line 0.10.x): pin versions and read the changelog.
- The rename from Agentflow resets brand recognition; "10xGraph" has no search history yet.
- LangGraph has stronger visual tooling (Studio, LangSmith observability).
- Requires Python 3.12 or newer. Code-first, not a no-code builder.
- Automatic per-tool permissions (user A may call `refund`, user B may not) are not built in. Tools can check the caller's verified scopes with `tenxgraph.core.authz.has_scope`.

---

## Roadmap

- Done: Core graph engine with nodes and edges
- Done: State management and checkpointing
- Done: Tool integration (MCP, custom tools, parallel execution)
- Done: Streaming and event publishing
- Done: Human-in-the-loop support
- Done: Prebuilt agent patterns
- Done: Agent-to-Agent (A2A) communication protocols
- Done: Observability and tracing (OpenTelemetry)
- Done: Realtime audio-to-audio agents (Gemini Live)
- Planned: Remote node execution for distributed processing
- Planned: More persistence backends (Redis, DynamoDB)
- Planned: Parallel/branching strategies
- Planned: Visual graph editor

---

## Contributing

10xGraph is built in the open and contributions are welcome: bug reports with a clean reproduction, docs and examples, provider coverage, persistence backends, and typing (removing a module from the `mypy` ignore list is a welcome pull request).

```bash
git clone https://github.com/10xGraph/10xGraph.git
cd 10xGraph
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy tenxgraph/
```

Read [CONTRIBUTING.md](https://github.com/10xGraph/10xGraph/blob/main/CONTRIBUTING.md) for the full workflow and the [Code of Conduct](https://github.com/10xGraph/10xGraph/blob/main/CODE_OF_CONDUCT.md). Questions and ideas go to [Discussions](https://github.com/10xGraph/10xGraph/discussions).

---

## Security

Found a vulnerability? Do not open a public issue. Follow the process in [SECURITY.md](https://github.com/10xGraph/10xGraph/blob/main/SECURITY.md).

---

## License

10xGraph is [MIT licensed](https://github.com/10xGraph/10xGraph/blob/main/LICENSE) and made by [10xScale](https://10xscale.ai). Copyright 10xScale. Contributions are accepted under the same license.

---

## Links

- Documentation: [10xgraph.com](https://10xgraph.com)
- PyPI: [`10xgraph`](https://pypi.org/project/10xgraph/)
- [Issues](https://github.com/10xGraph/10xGraph/issues) and [Discussions](https://github.com/10xGraph/10xGraph/discussions)
- [Changelog](https://github.com/10xGraph/10xGraph/blob/main/CHANGELOG.md)
- [Examples](https://github.com/10xGraph/10xGraph/tree/main/examples)
