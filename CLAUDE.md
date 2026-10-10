# 10xGraph core (Python library) — Engineering Guide

This file documents the **core Python framework** only (`10xgraph`, formerly `10xscale-agentflow`, the
package that lives in this repo). For the API/CLI, TS client, docs, or playground, see the CLAUDE.md in
their respective folders and the workspace-root `CLAUDE.md` for the monorepo overview.

- Package name (PyPI): `10xgraph` (old name `10xscale-agentflow`, last release 0.10.1)
- Repo: https://github.com/10xGraph/10xGraph (this folder is the repo root)
- Version: see `pyproject.toml` (0.10.0 is the first `10xgraph` release) (single source of truth: `pyproject.toml`; `tenxgraph.__version__`
  resolves from installed distribution metadata, so it cannot drift from this file)
- Requires: Python >= 3.12
- Importable top-level package is `tenxgraph/` at the repo root. `agentflow/` is a deprecated
  alias shim (same module objects, one `DeprecationWarning`) kept until 2.0; do not add code there.
  Uninstall `10xscale-agentflow` before installing `10xgraph`: both provide the `agentflow` module.

## What this package is

A graph-based orchestration engine for multi-agent LLM systems. It is **LLM-agnostic**: you bring
the provider SDK (OpenAI / Google GenAI), and 10xGraph provides the workflow engine, state,
persistence, tools, memory, evaluation, and event publishing. Inspired by LangGraph but simpler.

## Working principles for this codebase

- **Read before writing.** The public API is large and re-exported through many `__init__.py`
  files. Confirm the real export path before referencing a symbol (see Import Map below).
- **Examples are the source of truth**, not the README. `examples/` uses current import paths;
  the README and several docstrings still show pre-refactor paths (see Known Doc Drift).
- **Surgical edits.** This is `Development Status :: 5 - Production/Stable`. Don't refactor
  module boundaries or rename exports without checking every `__init__.py` that re-exports them.
- **Keep coverage green.** `pytest` enforces `--cov-fail-under=80`. New code needs tests.
- **Optional deps are optional.** Provider SDKs, MCP, Postgres, Redis, Qdrant, Mem0, Kafka,
  RabbitMQ, OTEL, a2a are all extras. Guard imports; never make core import a hard optional dep.

## Package layout (real, current)

The importable package is `tenxgraph/`. Top-level subpackages:

| Subpackage | What lives there |
|---|---|
| `core/` | The engine. `graph/` (StateGraph, Agent, ToolNode, CompiledGraph, Node, Edge), `state/` (AgentState, Message, content blocks, reducers, context managers), `llm/` (provider detection + client factory + `call_llm`), `skills/` (dynamic skill injection), `exceptions/` |
| `storage/` | `checkpointer/` (InMemory, Pg), `store/` (vector/long-term memory: Qdrant, Mem0, embeddings), `media/` (multimodal media processing, offload, resolvers, stores) |
| `runtime/` | `adapters/llm/` (OpenAI / OpenAI-Responses / Google GenAI / Anthropic response converters), `publisher/` (Console, Redis, Kafka, RabbitMQ, OTEL, Composite), `protocols/` (a2a, acp) |
| `prebuilt/` | `agent/` (React, RAG, PlanActReflect, SupervisorTeam, Swarm, StructuredOutput), `tools/` (calculator, fetch, files, handoff, memory, search) |
| `qa/` | `evaluation/` (criteria, datasets, evaluator, reporters, simulators) and `testing/` (TestAgent, mocks, quick tests) |
| `utils/` | constants (START/END/ResponseGranularity), `tool` decorator, `convert_messages`, callbacks, validators, id generators, background tasks, graceful shutdown |

## Import Map (verified) — this is the part that bites people

The package was restructured into `core/`, `storage/`, `runtime/`, `qa/`. **There are no
top-level `tenxgraph.graph`, `tenxgraph.state`, `tenxgraph.checkpointer`, `tenxgraph.skills`,
`tenxgraph.evaluation`, `tenxgraph.testing`, `tenxgraph.adapters`, or `tenxgraph.publisher`
shims.** Those paths raise `ModuleNotFoundError`. Use the canonical paths:

```python
# Graph engine
from tenxgraph.core.graph import Agent, StateGraph, ToolNode, CompiledGraph, Node, Edge, RetryConfig
# or the aggregate: from tenxgraph.core import StateGraph, Agent, ToolNode, AgentState, Message, ...

# State and messages
from tenxgraph.core.state import AgentState, Message, TextBlock, ToolResultBlock, add_messages

# LLM client/provider helpers
from tenxgraph.core.llm import call_llm, create_llm_client, detect_provider

# Skills
from tenxgraph.core.skills import SkillConfig, SkillMeta, SkillsRegistry

# Persistence
from tenxgraph.storage.checkpointer import InMemoryCheckpointer, PgCheckpointer, BaseCheckpointer
# Vector / long-term memory
from tenxgraph.storage.store import QdrantStore, Mem0Store, MemoryConfig, AgentMemoryConfig

# Publishers / converters
from tenxgraph.runtime.publisher import ConsolePublisher, RedisPublisher, KafkaPublisher, RabbitMQPublisher
from tenxgraph.runtime.adapters.llm import OpenAIConverter, GoogleGenAIConverter, OpenAIResponsesConverter

# Prebuilt
from tenxgraph.prebuilt.agent import ReactAgent, RAGAgent, SwarmAgent, SupervisorTeamAgent
from tenxgraph.prebuilt.tools import safe_calculator, fetch_url, create_handoff_tool, memory_tool

# QA
from tenxgraph.qa.evaluation import AgentEvaluator, EvalConfig, EvalCase, EvalSet
from tenxgraph.qa.testing import TestAgent, MockMCPClient, MockToolRegistry

# Utils
from tenxgraph.utils import tool, convert_messages, Command
from tenxgraph.utils.constants import START, END, ResponseGranularity
```

Note: the root `tenxgraph/__init__.py` exports `StateGraph`, `Agent`, `ToolNode`, `AgentState`,
`Message`, `START`, `END` lazily and does not eagerly import submodules. `import agentflow` and
`agentflow.*` still work as a deprecated alias until 2.0.

## Core concepts

**StateGraph -> CompiledGraph.** Build with `StateGraph()`, `add_node`, `add_edge`,
`add_conditional_edges`, `set_entry_point`; then `.compile(...)` returns a `CompiledGraph`.
`compile()` accepts: `checkpointer`, `store`, `media_store`, `interrupt_before`,
`interrupt_after`, `callback_manager`, `shutdown_timeout` (default 30.0).

**CompiledGraph execution API:** `invoke` / `ainvoke` (run), `stream` / `astream` (incremental),
`stop` / `astop` (interrupt), `override_node`, `attach_remote_tools`, `generate_graph`, `aclose`.
- Pause from inside a node or tool with `tenxgraph.utils.interrupt(value, message=..., ...)`; the
  run saves the thread paused before that node. Resume with `invoke({"resume": value}, config)`:
  the node re-runs and `interrupt()` returns `value`. `GraphInterrupt` is a `BaseException`.
- Per-run client tools: `config["remote_tools"]` (flat or OpenAI schemas) are offered by the
  `ToolNode` for that run only and handed to the client like `attach_remote_tools` tools. A
  client tool call pauses the graph after the tool node; the client's `ToolResultBlock`
  resumes it after that node.
- Core functions take DI deps as `Inject[...]` defaults and call `fresh()` (in
  `tenxgraph.utils.injection`) at the top: the proxy caches its first resolution for the
  process otherwise. New code taking an `Inject[...]` default must do the same.
- Input shape: `{"messages": [Message...]}`.
- Config keys: `user_id`, `thread_id`, `run_id`, `recursion_limit` (default 25).
- `response_granularity`: `LOW` (messages only, default), `PARTIAL` (context+summary+messages),
  `FULL` (full state).

**Agent class** (`tenxgraph.core.graph.Agent`) — the high-level node that wraps LLM calls,
message conversion, and tool integration. Key constructor params:
`model` (required), `output_type="text"`, `system_prompt`, `tool_node` (name or ToolNode),
`extra_messages`, `trim_context`, `tools_tags`, `reasoning_config`, `skills`, `memory`,
`retry_config` (default True), `fallback_models`, `multimodal_config`, `output_schema`.

**Model strings and providers.** `detect_provider(model)` infers the provider from a
`"provider/model"` prefix or the model name, and resolves to `"google"`, `"openai"`, or
`"anthropic"`. Examples: `"gemini/gemini-2.5-flash"`, `"openai/gpt-4o"`, `"gpt-4o-mini"`,
`"claude-opus-5"`, `"anthropic/claude-sonnet-5"`. Google's Vertex AI is selected via
`use_vertex_ai=True`.

Anthropic has three backends, so a boolean flag cannot express them; the selector is the
string `anthropic_backend`: `None` (direct Claude API, the default), `"vertex"`
(`AsyncAnthropicVertex`), or `"bedrock"` (`AsyncAnthropicBedrockMantle`, the Messages-API
endpoint, not the legacy `InvokeModel` client). Bedrock model ids keep their `anthropic.`
prefix (`"anthropic.claude-opus-5"`); it is preserved, not stripped. Install with the
`[anthropic]`, `[anthropic-vertex]`, or `[anthropic-bedrock]` extra.

Anthropic-specific request behaviour, all handled by the provider and not the caller:
`max_tokens` is required and defaulted (16000 non-streaming, 64000 streaming);
`temperature`/`top_p`/`top_k` are stripped for models that reject them with a 400;
`reasoning_config={"effort": ...}` maps to `thinking={"type": "adaptive"}` plus
`output_config.effort`, and `budget_tokens` is never emitted; a trailing assistant turn is
dropped because prefill 400s on current models. `output_type` is limited to `text` and
`json`: the Messages API has no image/audio/video generation endpoint.

Note the SDK is capped at `anthropic>=1.0.0,<2`. The 1.0 release moved to **httpx2**, so an
`http_client` passed through `llm_kwargs` on the Anthropic path must be an httpx2 client,
while the OpenAI path still takes httpx.

**ToolNode.** `ToolNode(tools, client=None, pass_user_info_to_mcp=False)`. First positional arg
is `tools` (an iterable of callables). `client` is an MCP client (fastmcp/mcp). Tools run in
**parallel** when the LLM requests several at once. Define tools as plain functions; injectable
params (`tool_call_id`, `state`, `config`, plus InjectQ-provided deps) are filled automatically.

**State and Message.** `AgentState` is a Pydantic model; subclass it for custom fields.
`Message.text_message(content, role="user")` is the text factory. `Message.tool_message(...)`,
`Message.image_message(...)` exist. There is **no `Message.from_text`** (README shows it; it is
wrong). Content is a list of typed blocks (TextBlock, ImageBlock, ToolCallBlock, ToolResultBlock,
ReasoningBlock, etc.). Reducers (`add_messages`, `replace_messages`, `append_items`) control how
state lists merge.

**Persistence.** `InMemoryCheckpointer` for dev/tests. `PgCheckpointer` (Postgres + Redis dual
layer) for production; requires `[pg_checkpoint]`.

**Memory / store.** 3-layer model: working state -> checkpointer (hot/durable) -> vector store
(Qdrant/Mem0) for long-term. `MemoryConfig` / `AgentMemoryConfig` drive it; `memory_tool` and
`create_memory_preload_node` wire it into a graph.

**Skills.** Implements the Agent Skills spec (agentskills.io): `SkillConfig(skills_dir=...)`
discovers `<dir>/<name>/SKILL.md` skills. Two modes: `on-demand` (an `<available_skills>` catalog
in the system prompt; the LLM calls `activate_skill()` and `read_skill_resource()` for bundled
files) and `session` (preload a fixed skill from a state field via `preload_from`). Activations
are recorded in `execution_meta.internal_data["active_skills"]` and re-injected after trimming.
`validate_skill()` / `10xgraph skills --validate` check skills against the spec.

**Publishers.** Emit execution events to Console, Redis Pub/Sub, Kafka, RabbitMQ, or OTEL.
`CompositePublisher` fans out to several. OTEL publisher provides tracing (`setup_tracing`).

**QA.** `tenxgraph.qa.evaluation` is a full eval framework (criteria incl. LLM-as-judge,
trajectory matching, rubric, safety, hallucination; datasets; console/JSON/HTML/JUnit reporters;
user simulators). `tenxgraph.qa.testing` provides `TestAgent`, `MockMCPClient`, `MockToolRegistry`,
`TestContext` for unit-testing graphs without live LLMs.

## Development workflow

This repo root is the 10xGraph repo; the importable package is `tenxgraph/`. A `.venv` is
already present.

```bash
# from the repo root
.venv/bin/python -m pytest               # full suite (enforces coverage >= 80%)
.venv/bin/python -m pytest tests/graph   # one area
ruff check . && ruff format .            # lint + format (line-length 100, py312)
uv run mypy tenxgraph/                   # type check
# editable install with extras for local dev:
pip install -e ".[google-genai,openai,anthropic,mcp,pg_checkpoint]"
```

- Tests live in `tests/` (mirrors package layout: `graph/`, `state/`, `storage/`, `store/`,
  `checkpointer/`, `publisher/`, `prebuilt/`, `evaluation/`, `testing/`, plus `chaos/`,
  `benchmarks/`, `integration/`). Markers: `asyncio`, `integration` (needs real DBs), `slow`.
- Lint config is in `pyproject.toml` `[tool.ruff]` (broad rule set; per-file ignores for a few
  large modules). `mypy` and `bandit` are also configured there.
- `examples/` is organized by feature (react, rag, swarm, supervisor_team, memory, skills, mcp,
  a2a_sdk, evaluation, testing, multimodal, structured_output, ...). Use these as canonical usage.

## Known doc drift (do not copy from these without checking)

- **`Message.from_text` does not exist.** Use `Message.text_message`.
- **`ToolNode(functions=...)`** is wrong. The param is `tools`.
- Some `examples/` files still use pre-rename or dead paths (`agentflow.*`, `agentflow.state.message`,
  `agentflow.graph.tool_node`, `agentflow.evaluation.*`). `agentflow.*` works through the alias with a
  `DeprecationWarning`; the dead ones are broken. Use `tenxgraph.*` in new code.

## Renamed identifiers (fallbacks kept)

- OTEL tracer/meter name and `GEN_AI_SYSTEM`: `10xgraph`. Logger names: `tenxgraph.*`.
- Media URI scheme `graph://media/` (old `agentflow://media/` still read).
- Default home dir `~/.10xgraph` (falls back to `~/.agentflow` if only that exists); used by the
  `SqliteCheckpointer` default path.
- Cloud media prefix `10xgraph-media` (old `agentflow-media` objects still read).
- Prebuilt tools user-agent `10xgraph-prebuilt-tools/1.0`.
- Server config file is `10xgraph.json` (the CLI package falls back to `agentflow.json`). The CLI
  is `10xgraph-api` (command `10xgraph`, `agentflow` deprecated alias) and the TS client is
  `10xgraph-client`.

When you touch any of the above, prefer fixing the doc/example to match the code rather than the
reverse, unless the export path itself is the bug.
