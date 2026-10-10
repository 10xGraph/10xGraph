# Changelog

All notable changes to `10xgraph` (formerly `10xscale-agentflow`) are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/). From `1.0.0` on, the
public API is stable: breaking changes bump the **major** version and follow the
deprecation policy below.

## Deprecation policy

Starting from this release:

- A public API is never removed without first being deprecated for at least one
  minor release. Deprecated APIs emit a `DeprecationWarning` naming the
  replacement, and keep working until removal.
- Modules that move keep a back-compat shim for at least one minor release rather
  than being deleted outright. (The `agentflow.graph` / `agentflow.state` /
  `agentflow.checkpointer` moves in an earlier release shipped with no shims and
  simply started raising `ModuleNotFoundError`; that will not happen again.)
- Breaking changes are listed under a `### Breaking` heading, with the migration.

---

## [Unreleased]

## [0.10.0] - 2026-10-10

First release of `10xgraph`. Its version numbering starts fresh; the old
`10xscale-agentflow` releases below keep their own numbers.

### Breaking

- **The project is renamed from Agentflow to 10xGraph.** The PyPI distribution is now
  `10xgraph` (was `10xscale-agentflow`, whose last release is 0.10.1). The importable
  package is now `tenxgraph` (a Python identifier cannot start with a digit). All
  canonical paths are the old ones with `agentflow` replaced by `tenxgraph`, for example
  `tenxgraph.core.graph`, `tenxgraph.core.state`, `tenxgraph.storage.checkpointer`,
  `tenxgraph.prebuilt.agent` and `tenxgraph.utils`. The top level also exports
  `StateGraph`, `Agent`, `ToolNode`, `AgentState`, `Message`, `START` and `END` lazily.
- **Observable identifiers changed.** If dashboards, alerts or log filters match the old
  values, update them:
  - OpenTelemetry tracer name, meter name and `GEN_AI_SYSTEM`: `agentflow` -> `10xgraph`.
  - Logger names: `agentflow.*` -> `tenxgraph.*`.
  - Prebuilt tools HTTP user-agent: now `10xgraph-prebuilt-tools/1.0`.

### Changed

- **`agentflow` stays importable as a deprecated alias until 2.0.** `import agentflow`
  and `agentflow.*` resolve to the same module objects as `tenxgraph.*` and emit one
  `DeprecationWarning`. Existing code runs unchanged while you migrate imports.
- **Media URI scheme** `agentflow://media/` -> `graph://media/`. Old URIs are still read.
- **Default home directory** `~/.agentflow` -> `~/.10xgraph`. If only `~/.agentflow`
  exists it is used instead. This affects the default `SqliteCheckpointer` path.
- **Cloud media prefix** `agentflow-media` -> `10xgraph-media`. Old objects are still read.
- The server config file is now `10xgraph.json`. The CLI falls back to `agentflow.json`.
  The CLI package is renamed alongside this release: `10xgraph-api` (was
  `10xscale-agentflow-cli`) with the `10xgraph` command (`agentflow` stays as a
  deprecated alias). The TypeScript client becomes `10xgraph-client` on npm.

### Migration

```bash
pip uninstall 10xscale-agentflow
pip install 10xgraph
```

Uninstall first: both distributions provide the `agentflow` module and must not be
installed side by side. Then replace `agentflow` with `tenxgraph` in imports:

```python
# before
from agentflow.core.graph import StateGraph
# after
from tenxgraph import StateGraph
```

---

## `10xscale-agentflow` [0.10.0] - 2024-06-05

### Added

- **`interrupt()`: pause a graph from inside a node or tool.** `from agentflow.utils import
  interrupt`. The first call stops the run and saves the thread paused before that node
  (`agentflow.utils.pending_interrupt(state)` returns the request; streaming emits
  an `UPDATES` chunk with `status="interrupted"`). Resume with `ainvoke({"resume": value},
  config)`: the node runs again and `interrupt()` returns `value`. Supports `message`, `reason`
  and `response_schema`, several calls per node (answered in order), and tools, including
  parallel tool calls (finished siblings are served from the tool-result ledger under
  `invoke`). `GraphInterrupt` derives from `BaseException`, so tool and node error handling
  cannot swallow it. A paused thread rejects input without `resume` with `ValueError`.
- **Per-run client tools (`config["remote_tools"]`).** A run can bring its own client-executed
  tool schemas (flat `{name, description, parameters}` or OpenAI shape) without mutating the
  graph. `ToolNode.all_tools(config=...)` lists them for that run and routes calls to them to
  the client like configured remote tools; a name the node already has is ignored.
  `Agent` passes its run config when resolving tools.
- `agentflow.utils.injection.fresh()` resolves `Inject[...]` defaults per call (see Fixed).

- **Native Anthropic provider.** `model="claude-opus-5"` (or `"anthropic/..."`,
  `"claude/..."`) now builds a real Anthropic client instead of silently
  constructing an `AsyncOpenAI` that failed at request time. Covers non-streaming
  and streaming, tool calling, multimodal input, reasoning, usage accounting, and
  retry/fallback. Install with the `anthropic`, `anthropic-vertex`, or
  `anthropic-bedrock` extra.
  - Three backends, selected with `anthropic_backend`: `None` (direct Claude API),
    `"vertex"` (`AsyncAnthropicVertex`), `"bedrock"`
    (`AsyncAnthropicBedrockMantle`, the Messages-API endpoint). Bedrock model ids
    keep their `anthropic.` prefix.
  - `max_tokens` is required by the API and is defaulted automatically (16000
    non-streaming, 64000 streaming).
  - `temperature`/`top_p`/`top_k` are stripped for models that reject them with a
    400, per model rather than as a blanket strip.
  - `reasoning_config={"effort": ...}` maps to `thinking={"type": "adaptive"}`
    plus `output_config.effort`. `budget_tokens` is never emitted: it returns a
    400 on current Claude models.
  - A trailing assistant turn is dropped, since prefill returns a 400 on current
    models. This specifically protects the injected `context_summary`.
  - A policy `refusal` is a successful HTTP 200 with a possibly-empty `content`;
    it is surfaced as a message with `metadata["refusal"]` rather than being
    treated as a transient failure that burns the model fallback list.
- **`Agent.count_tokens(messages, tools)`.** Counts a request's input tokens
  before sending it, using the provider's own endpoint and the exact payload the
  real call would send (system prompt, tool schemas, merged tool results). Only
  Anthropic exposes a server-side counting endpoint today; other providers raise
  `NotImplementedError` rather than returning a guess.
- **Anthropic prompt caching.** `anthropic_cache=True` (or a dict such as
  `{"type": "ephemeral", "ttl": "1h"}`) places `cache_control` breakpoints at the
  end of the stable request prefix: the last tool and the last system block,
  since render order is tools -> system -> messages. Skipped when the caller
  placed their own breakpoints. Verify hits with
  `usage.cache_read_input_tokens`; a prefix under ~1024 tokens silently does not
  cache.
- **Anthropic server tools.** `web_search_tool()`, `web_fetch_tool()`, and
  `code_execution_tool()` build correctly-dated definitions, picking the
  dynamic-filtering variant on models that support it and the basic variant
  otherwise. Server-tool result blocks are captured into
  `metadata["server_tool_results"]`, and a `server_tool_use` block is recorded
  without being added to `tools_calls` so the graph does not re-run work
  Anthropic already did. A server-tool error arrives as a normal HTTP 200 with an
  error object rather than raising, and is surfaced as `error_code`.
  `stop_reason: "pause_turn"` is flagged as `metadata["pause_turn"]`.
- **`AnthropicBatch`** (`agentflow.core.llm.AnthropicBatch`) for the Message
  Batches API: build, submit, poll, and collect. Results are keyed by
  `custom_id` throughout, because batch results arrive in any order.
- **`call_llm` supports Anthropic**, so `SummaryContextManager`, the evaluation
  judge, and `UserSimulator` all work with Claude models.
- **Skills follow the Agent Skills specification (agentskills.io).** Skills
  written for Claude Code, Codex or GitHub Copilot load unchanged:
  - All spec frontmatter is parsed (`license`, `compatibility`, `allowed-tools`,
    `metadata`) and exposed on `SkillMeta`.
  - The system prompt gets an `<available_skills>` catalog with each skill's name
    and description.
  - `activate_skill(skill_name)` returns the body in `<skill_content>` tags, with
    a `<skill_resources>` list of bundled files.
  - `read_skill_resource(skill_name, path)` reads any file in the skill directory
    as text: references, `.py` / `.sh` scripts, extension-less executables, data.
    Binary files are described, large files truncated, and paths outside the
    skill directory rejected.
  - `skill_name` is an enum of the discovered names.
  - Loading is lenient: a spec violation is recorded as a `SkillDiagnostic`
    instead of dropping the skill, and a value containing `: ` that breaks the
    YAML is recovered.
  - `SkillConfig.skills_dir` accepts a list of directories (the earlier one wins
    on name clashes), and a directory that is itself a skill.
  - New options: `max_resource_bytes`, `include_skill_path`.
  - New `validate_skill()` checks a skill against the specification.
- **Activated skills survive context trimming.** Activations are recorded in
  `execution_meta.internal_data["active_skills"]`. If a context manager drops the
  tool result that carried a skill's instructions, the agent re-injects them as a
  system message, and a repeated activation of a skill still in context is not
  duplicated.
- **Skill tool calls fire `InvocationType.SKILL` callbacks.** The enum value
  existed but was never fired. `activate_skill` and `read_skill_resource` now
  report as `SKILL` instead of `TOOL`, so callbacks registered for `TOOL` no
  longer see them.

### Fixed

- **Client-side (remote) tool calls now pause the graph.** The check was
  `RemoteToolCallBlock in message.content`, a class compared against block instances, so it
  was never true: a graph whose tool node returned a remote call kept running, and an `Agent`
  saw the empty placeholder as the tool's result. The graph now pauses after the tool node
  (after saving any server tool results from the same step) and, when the client sends the
  results on the same thread, resumes after that node with the next node chosen from the
  updated context. The placeholder is kept out of the context; `invoke` still returns it in
  `messages` so clients can read the calls.
- **Messages a node appends to `state.context` are streamed.** A node that appended to
  `state.context` and returned the same state object had those messages saved but never
  streamed or returned in `messages`. `Command(state=...)` also stopped re-streaming messages
  from earlier steps.
- **`Inject[...]` defaults no longer pin the first resolved dependency for the whole
  process.** The core took checkpointer, publisher, store, context manager and callback
  manager as `Inject[...]` defaults without `@inject`; the proxy caches its first resolution,
  so the first graph that ran decided which checkpointer every later graph used. They now
  resolve from the active container on each call. `StateGraph` also checks whether a
  publisher is bound (`has`) instead of resolving one, which could try to build the abstract
  `BasePublisher` when a `dict` happened to be bound.

- **`use_vertex_ai=True` hijacked Claude models.** The flag short-circuited
  provider detection to `"google"` before the model name was examined, so
  `Agent(model="claude-opus-5", use_vertex_ai=True)` built a Google GenAI client
  with a Claude model name and failed at request time. Claude runs on Vertex too,
  and `use_vertex_ai` is the flag a caller naturally reaches for, so it now acts
  as a backend selector for Anthropic (equivalent to
  `anthropic_backend="vertex"`) instead of a provider override. An explicit
  `anthropic_backend` still wins. Behaviour for every non-Claude model is
  unchanged.
- **`call_llm` sent unrecognised model prefixes to the SDK verbatim.** It used
  `detect_provider`, which selects a provider but does not strip the prefix, so
  `call_llm("gemini/gemini-2.5-flash", ...)` passed the full string as the model
  name. It now uses `resolve_provider_and_model`.
- **Parallel tool calls lost `execution_meta.internal_data` writes.** Each parallel
  branch runs on its own state copy and `execution_meta` was never merged back, so
  anything a tool recorded there disappeared. Changed keys are now merged, and
  lists that several branches appended to keep every branch's items.
- **Reading a non-UTF-8 skill resource crashed the tool** with an uncaught
  `UnicodeDecodeError`.
- **Checkpointers no longer import a class named by stored data.** `PgCheckpointer` and
  `SqliteCheckpointer` saved the state's module path (`__class_path__`) in every row and
  imported it on load. A row could choose which module got imported, renaming or moving a
  state class broke every stored thread, and `PgCheckpointer` fell back to `AgentState` on a
  failed import, silently dropping custom fields. Rows now hold
  `model_dump(mode="json")` plus a `{"__state_meta__": {"format": 1, "class": "<Name>"}}`
  header, and are rebuilt into the state class bound by `StateGraph.compile()`
  (`BaseCheckpointer.bind_state_type()`, `state_type`). Old rows still load: the class path
  is ignored, never imported. A stored class name that differs from the bound class logs a
  warning.

### Breaking

- **`anthropic/` and `claude/` are now recognised model prefixes.** Previously
  they were unknown prefixes that fell through to the OpenAI provider, which was
  how Claude-behind-an-OpenAI-compatible-proxy worked. `Agent(model=
  "anthropic/claude-3")` now selects the native Anthropic provider and strips the
  prefix. **Migration:** pass `provider="openai"` explicitly to keep routing
  through an OpenAI-compatible endpoint:
  `Agent(model="anthropic/claude-3", provider="openai", base_url=...)`.
- **Skills API reworked around the Agent Skills specification.** **Migration:**
  - The `set_skill(skill_name, resource)` tool is replaced by
    `activate_skill(skill_name)` and `read_skill_resource(skill_name, path)`.
    Prompts or code that mention `set_skill`, or parse its `## SKILL:` header,
    must use the new names and the `<skill_content name="...">` wrapper.
  - `SkillConfig.inject_trigger_table` is renamed `inject_catalog`.
  - `SkillsRegistry.build_trigger_table()` is replaced by `build_catalog()`.
  - `build_set_skill_tool()` and `load_resources()` are removed. Use
    `activation.make_activate_skill_tool()` and `SkillsRegistry.read_file()`.
  - The `resources:` frontmatter list and `SkillMeta.resources` are removed; every
    file in the skill directory is readable.
  - `triggers`, `tags` and `priority` are read only from `metadata`, as strings
    (`triggers: "a; b"`, `tags: "x, y"`, `priority: "10"`). YAML lists inside
    `metadata` still load, with a diagnostic. Top-level fields are ignored, with a
    diagnostic.
  - Skill names are no longer lowercased, and underscores, 128-character names and
    2000-character descriptions produce diagnostics instead of being accepted
    silently.
  - Discovering the same name from two directories no longer raises; the first
    wins and the second is reported as shadowed. `register()` still raises unless
    `replace=True`.
  - With no skills discovered, no tool or catalog is registered and a `ToolNode`
    is no longer required.
- **One checkpointer instance serves one state class.** `compile()` binds the graph's state
  class to the checkpointer, and binding a different class raises `ValueError`. A checkpointer
  created with an explicit type (`PgCheckpointer[MyState](...)`) also raises if the graph's
  state class does not subclass that type. A checkpointer used without `compile()` restores
  plain `AgentState`. **Migration:** give each graph with its own state class its own
  checkpointer instance; call `checkpointer.bind_state_type(MyState)` when reading threads
  outside a compiled graph.

## [1.0.0] - 2026-07-19

First stable release. The public API is now covered by the deprecation policy above,
and the package is classified `Development Status :: 5 - Production/Stable`. This
release is the production-hardening round on top of `0.8.0`: durable state gets
optimistic concurrency and an idempotent tool ledger, execution is bounded by real
timeouts and cancellation, and per-user isolation is enforced across the storage
layer.

### Breaking

- **`injectq` is pinned to `>=0.4.0,<0.5`.** It is pre-1.0, so a `0.5` release may
  break the API; without an upper bound it would have been picked up automatically
  and broken fresh installs.
- **Default `user_id` is now `"anonymous"`** (was `"test-user-id"`). A run with no
  `user_id` previously filed itself under a placeholder that looks like a real
  account, which, with per-user isolation enabled, silently pooled every
  unauthenticated run into one identity.
- **A conditional edge whose condition raises now fails the run** (`GraphError`,
  `GRAPH_ROUTING_001`). Previously the exception was swallowed and the graph fell
  through to the first static edge or `END` — silently taking a path nobody chose.
- **Production refuses to start with wildcard CORS *and* credentials enabled**
  (API package). Set explicit `ORIGINS`, or `CORS_ALLOW_CREDENTIALS=false`.

### Added

- **Optimistic concurrency control on durable state.** `states` now carries a
  `version` column with `UNIQUE (thread_id, version)`; writes take a per-thread row
  lock and compare-and-swap. A write based on a stale version raises the new
  `StaleStateError` (HTTP 409 at the API) instead of silently discarding another
  run's update.
- **Durable tool-execution ledger** (`tool_executions`, schema v3). A node replayed
  after a crash no longer re-fires tool calls that already completed — the
  double-charge scenario. Keyed by
  `(thread_id, origin_message_id:tool_call_id)`, because a `tool_call_id` alone is
  not unique across turns.
- **Per-step durable checkpointing** (`durable_checkpoint_every_step`, default on),
  so a crash replays one node rather than the whole run.
- **Node and tool timeouts** (`node_timeout`, `tool_timeout`) that actually cancel
  the work, plus **stop-cancels-a-running-node** — previously stop was only polled
  *between* nodes, so a hang inside one was unreachable.
- **Real schema migrations** with a stepwise, idempotent runner and a
  `pg_advisory_xact_lock`, so concurrent workers cannot race the DDL.
- **Per-user isolation in the checkpointer** (`enforce_user_isolation`, default on)
  across state, messages, and threads.
- **File ownership.** Uploads record an owner; reads by another user 404.
- **Backpressure on background tasks** (`max_pending_tasks`, default 1000). A slow
  or dead publisher sink previously grew an unbounded task set until OOM.
- **OpenTelemetry metrics** via `metrics.setup_otel_metrics()`; counters and
  histograms on node/tool execution, with outcome dimensions.
- **Structured, correlated logging** via `logging.setup_structured_logging()`; every
  record carries `run_id` / `thread_id` / `node`, so one run can be grepped out of a
  busy server.
- `agentflow build --k8s` generates a Kubernetes manifest whose termination grace
  period is long enough that a rolling deploy does not kill in-flight runs.

### Fixed

- **Lost updates on concurrent writes to one thread** (see CAS above). Reads were
  also non-deterministic: `ORDER BY created_at DESC` with no tiebreak.
- **The realtime cache could be moved backwards**, wedging a thread until its TTL
  expired. Cache writes are now an atomic version-guarded compare-and-set, and a
  lost version check invalidates the cache so the thread self-heals.
- **Parallel tools clobbering each other's state.** Each tool now runs on its own
  branch copy, merged back field-by-field against a baseline, using a field's
  reducer when it has one.
- **One failing tool orphaned its siblings** (`gather` without `return_exceptions`),
  and malformed tool arguments raised `JSONDecodeError` through the whole node.
- **Retries on non-retryable errors.** Status classification matched `"500"` as a
  substring, so `max_tokens must be <= 500` was treated as a server error.
- **Cross-tenant reads/deletes** of state, messages, threads, and files.
- **Rate limit bypass.** The bucket key came from the leftmost `X-Forwarded-For`
  entry, which the caller controls — a new value per request meant a new bucket and
  no limit at all. Proxy hops are now counted from the right.
- Blocking `urllib.urlopen` inside `async def` (cloud media store) stalled the event
  loop for every concurrent run in the process.
- Connection-pool and Qdrant-collection cold-start races (double creation).
- Schema-version failures were swallowed instead of raised.
