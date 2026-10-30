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

### Versioning

From `1.0.0` on, `10xgraph`, `10xgraph-api` and `10xgraph-client` are versioned together:

- **major.minor is identical across the three packages.** `10xgraph` 1.1.x works with
  `10xgraph-api` 1.1.x and `10xgraph-client` 1.1.x.
- **Patch numbers move independently.** A patch release contains only bug or security fixes.
- **Any new feature means all three packages release the next `X.Y.0` together.** A package with
  no changes of its own ships that release as "version alignment, no changes".
- **Breaking changes need a new major version (2.0)**, and the feature they change is deprecated
  for at least one minor release first.
- The deprecated `agentflow` import alias is removed in 2.0.

---

## [1.0.0] - Unreleased

The stable 1.0 release of `10xgraph`, aligned with `10xgraph-api` 1.0 and `10xgraph-client` 1.0.
It is the `1.0.0rc1` code plus the changes below. If you upgrade from 0.10.x, the `1.0.0rc1`
breaking list further down applies to you as well.


### Breaking

- **Behaviour change: default reasoning is the provider's lowest effort.** `reasoning_config`
  left unset or set to `True` (the `ReactAgent` default) used to mean `{"effort": "medium"}`
  everywhere (Google `thinking_budget=8192`). It now means reasoning on at each provider's lowest
  effort that still reasons, resolved per request so a fallback model on another provider gets
  its own value: OpenAI `"minimal"`, Anthropic `"low"`, Google `"low"` (`thinking_budget=512`).
  The values live in `DEFAULT_REASONING_EFFORT_BY_PROVIDER`
  (`tenxgraph/core/graph/agent_internal/constants.py`). Pass a dict such as
  `{"effort": "medium"}` to keep the old depth; `None` or `False` still turns reasoning off.
- **OpenAI: reasoning text comes back by default, and models without reasoning still work.** On
  the Responses API, when reasoning is on and no `summary` is set, the request now asks for
  `reasoning.summary: "auto"` (before, no reasoning text came back). When the reasoning came from
  the default and the model answers with a 400 that names the reasoning parameter (for example
  gpt-4o), the call is retried once without it and the model is remembered for the process, so
  later calls skip it. An explicit `reasoning_config` dict is never dropped and still raises.
- **OpenAI: a custom `base_url` switches to Chat Completions only when Responses is missing.**
  The switch used to happen after any error, so one 429 or timeout moved the agent to Chat
  Completions for good. Now only 404, 405 and 501 switch it; other errors are raised.
- **Behaviour change: `Agent` uses the OpenAI Responses API by default.** `api_style` now
  defaults to `"responses"` (it was `"chat"`), except when `base_url` is set and `api_style` is
  not passed: then it stays `"chat"`, because most OpenAI-compatible servers only speak Chat
  Completions. An explicit `api_style` always wins. Pass `api_style="chat"` to keep the old
  behaviour. This fixes `Agent(model="gpt-6-luna", tools=...)` failing with "Function tools with
  reasoning_effort are not supported ... in /v1/chat/completions". Related changes:
  - With `api_style="responses"` and a `base_url`, a server that rejects the Responses API is
    probed once; the agent then stays on Chat Completions and logs one warning (it used to retry
    Responses and fall back on every call).
  - `output_schema` now uses `responses.parse(text_format=...)` on the Responses style instead
    of switching to Chat Completions. The result is in `Message.parsed_content` as before. Streaming
    is still turned off for these calls.
  - On Chat Completions, `reasoning_effort` (and the `extra_body` reasoning for a `base_url`) is
    sent only when you pass `reasoning_config` yourself, not for the implicit
    `{"effort": "medium"}` default.
  - The chat structured-output path calls `client.chat.completions.parse`; the
    `client.beta.chat.completions.parse` it used does not exist in openai 3.x.
  - `output_type="image"` and `"audio"` always use the images and audio endpoints, whatever
    `api_style` is.
  - The `openai` extra is now `openai>=1.93.1,<4`. openai 3.x uses httpx2, so an `http_client`
    passed through `llm_kwargs` must be an `httpx2.AsyncClient` on 3.x.
- **Tool and node timeouts are off by default.** `DEFAULT_TOOL_TIMEOUT_SECONDS` and
  `DEFAULT_NODE_TIMEOUT_SECONDS` in `tenxgraph.utils.constants` are now `None` (they were 300 s
  and 900 s). A slow tool or node runs until it finishes. Set `tool_timeout` or `node_timeout`
  in the run config to get a limit; a value you set now applies to both `invoke` and `stream`
  (the stream path used to ignore it). A run timeout is a server setting: see `10xgraph-api`.
- **Tool arguments are checked against the types the tool declares.** A value that converts
  cleanly is converted (`"5"` for an `int`, a number for a `str` parameter); a value that cannot
  convert (a list for an `int`) is refused and the model gets a failed tool result. Unknown
  extra keys are still ignored. Before, `add(a: int)` ran with whatever the model sent.
- **Published events are snapshots.** Each event is copied before it is queued, so a Kafka,
  Redis, RabbitMQ or OTEL consumer sees the values at the moment of the event. Before, the run
  loop edited the one event object after it was queued, and consumers saw the last values for
  every event. Code that relied on seeing later edits no longer does.
- **`pass_user_info_to_mcp=True` sends the user info as request metadata.** It arrives in the
  MCP call's `meta` (`{"user": {...}}`), not as a `user` tool argument. A strict-schema MCP
  server no longer rejects the call, and a server that read `user` from the arguments must
  read it from the request meta. It needs a fastmcp client whose `call_tool` accepts `meta`;
  otherwise the user info is not sent and a warning is logged.
- **`LiveAgent.arun()` raises `RuntimeError` when the same instance is already running a
  session.** `CompiledGraph.arealtime()` now runs each connection on its own copy
  (`LiveAgent.new_session()`), so concurrent connections no longer cross-wire. If you call
  `arun()` yourself, give each concurrent session its own `new_session()`.
- **A stop request is saved as `StopRequestStatus.STOPPED`.** The state of a stopped thread
  used to keep `STOP_REQUESTED`, which the next run read back from the cache and stopped at
  once. Code that checks `execution_meta.stop_current_execution` for `STOP_REQUESTED` after a
  stop must also accept `STOPPED`.
- **`mem0ai` is bounded to `>=2.0,<3`.** mem0ai 1.x is no longer supported.
- **Sync entry points raise inside a running event loop.** Sync `invoke()`, `stream()` and
  `stop()`, and sync checkpointer methods such as `get_state()`, now raise `RuntimeError` naming
  the async method to use (`ainvoke`, `astream`, `astop`, `aget_state`) when called from a
  running loop. Before, they hung the loop. Call the async method from async code.
- **Removed: `Message.reasoning`.** Reasoning lives only as `ReasoningBlock`s in
  `Message.content`; read `block.summary` (plus `signature` / `redacted_data` for Anthropic) from
  the blocks with `type == "reasoning"`. A saved state that still holds a top-level `reasoning`
  string on a message loads without error, and the string is ignored (it is not turned into a
  block). The Postgres `messages.reasoning` column is dropped (see Upgrade notes), and the
  store's `has_reasoning` metadata is now true when a message has a reasoning block with text,
  details or redacted data.

### Fixed

- **A handoff no longer reads as an unfinished transfer.** The result of the handoff call now
  tells the receiving agent it is in charge: "Transferred to WEATHER. You are now WEATHER:
  answer the user's latest request yourself, and call your tools if you need them." Before, the
  bare "Transferred to WEATHER." made Gemini reply that it would transfer the user to its tool
  and stop without calling it. Applies to the ToolNode handoff and the `SwarmAgent` handoff nodes.
- **`PlanActReflectAgent`: the user sees the planner's reply, not the reflector's review.** Every
  reflector message now carries `metadata["internal"] = True` (a step message clients may show
  collapsed; the turn's answer is the last assistant message without it). The reflector prompt
  asks for a one-line verdict ending in `[DONE]` instead of a second answer. If the planner's last
  reply has no text (tool calls only), the done verdict is not marked internal because it is the
  only text answer. Routing is unchanged.
- **Streaming: clients saw the final message instead of the deltas.** The agent, function-node
  and converter paths reused one `StreamChunk` for every item and only changed its `message`.
  Because the stream is queued before it is read, every queued chunk pointed at the last item:
  readers got the complete message once per converter item (duplicate reasoning rows, usage
  counted per copy). Each item now gets its own chunk with its own metadata dict.
- **Cancel and shutdown cleanup edge cases.** `aclose()` now waits (within its drain budget,
  reported as `cleanup_tasks` in its stats) for cancel saves and run-marker releases that
  outlived their run, so the checkpointer is not released under them. The run-marker heartbeat
  re-creates a missing marker with the new `aput_cache_value_if_absent` (Redis `SET NX`, one
  SQLite statement, in-memory under its lock; the base class falls back to get-then-put), so it
  cannot overwrite a newer run's marker. The stop, interrupt and pause saves go through the same
  ordering as the final save, so a late one cannot overwrite it. `astream` keeps the run handle
  pointed at the task of the current step, so a hard cancel reaches the step that is running.
- **Eval: a failing judge model is an error, not a 0.5 score.** The LLM judge and
  `SimulationGoalsCriterion` turned any LLM failure (bad key, unsupported provider, empty or
  invalid JSON reply) into a believable score of 0.5 with the reason "No LLM provider
  available". The criterion now reports an error with the real exception message (when every
  sample of a multi-sample judge failed; a sample that succeeds still counts).
- **Gemini memory search ranks better.** `GoogleEmbedding` now sends the Gemini task type
  `RETRIEVAL_DOCUMENT` when embedding text to store and `RETRIEVAL_QUERY` when embedding a search
  query (new `BaseEmbedding.aembed_query`, which defaults to `aembed` for other providers;
  `QdrantStore.asearch` uses it). Set `GoogleEmbedding(document_task_type=...,
  query_task_type=...)` to other Gemini task types, or `None` to send none (the old behaviour).
  Vectors stored before this change stay searchable.
- **A Redis outage no longer slows every run step.** `PgCheckpointer` skips Redis for a short
  cool-down after a connection failure and takes its usual fallback (Postgres, or no cache) at
  once, so a run no longer waits for the connect timeout on each of its Redis calls. New kwarg
  `redis_breaker_cooldown` (seconds, default 5.0; 0 disables). A cache entry whose invalidation
  was skipped is deleted when Redis is reachable again, and a cached state with no version is
  no longer served once Postgres holds a newer state.

### Upgrade notes

#### Postgres schema migration (read before the rollout)

`PgCheckpointer` now creates and upgrades its schema on first use. The upgrade from 0.10.x
includes migration v4, which changes the key of the `messages` table and builds its index
under an `ACCESS EXCLUSIVE` lock. Old and new code cannot run against the same database at the
same time: once v4 has run, every message insert from a 0.10.x pod fails.

Do not roll the upgrade pod by pod. Instead:

1. Stop all old (0.10.x) pods.
2. Run the migration (start one new pod, or call `await checkpointer.asetup()`), and wait for
   it to finish.
3. Start the remaining new pods.

A rollback to 0.10.x needs a restore of the database from before the migration; the migration
is not reversible.

#### Postgres schema version 5 (drops `messages.reasoning`)

Schema version 5 drops the unused `reasoning` column from the `messages` table. The migration
runs automatically at startup (`ALTER TABLE ... DROP COLUMN IF EXISTS reasoning`). A rolling
deploy that mixes old and new versions on one database is not supported across this migration:
old pods still write the column, so once it is dropped their message inserts fail. Use the same
stop-migrate-start steps as above.

#### Dependency minimums

The extras raise these floors. A fresh install resolves them; an existing environment must be
upgraded.

- `images`: `Pillow>=13.0.0` (was 12.2.0).
- `mcp`: `mcp>=1.30.0` (was 1.13.0).
- `otel`, `langsmith`: `opentelemetry-api>=1.45.1` and `opentelemetry-sdk>=1.45.1` (were 1.20.0).
- `mem0`: `mem0ai>=2.0,<3` (was `>=0.1.117`).

Related notes:

- `PgCheckpointer` creates its schema on first use, and `10xgraph-api` calls `asetup()` at
  boot. No separate setup step is needed.
- If you ran two graphs on one checkpointer, `compile()` now logs a WARNING for the second
  one. Use one graph per container or give each graph its own checkpointer.

### Added

- **User simulator progress events.** `UserSimulator.run(..., on_event=callback)` and
  `BatchSimulator.run_batch(..., on_event=callback)` call `callback(event, data)` with
  `turn_started`, `agent_replied`, `goals_checked`, `evaluating` and `agent_error`. A callback
  that raises is logged at debug and ignored.
- **Graceful shutdown.** `await graph.aclose(grace=None, stop_timeout=None, timeout=None)` ends
  active runs in phases: new runs are refused, runs get `grace` seconds to finish, the rest
  get a stop request and save their thread as INTERRUPTED, then anything still running is
  cancelled and background tasks are drained. Defaults come from `compile(shutdown_timeout=...)`
  (50% grace, 25% stop, 25% cancel and drain). It returns the counts of runs finished, stopped
  and cancelled, and of background tasks drained or dropped.
- `GraphClosedError` (`GRAPH_CLOSED_001`): raised by `invoke`, `stream`, `ainvoke`, `astream`
  and `arealtime` once the graph is closing or closed.
- `install_shutdown_signals(graph, ...)` in `tenxgraph.utils.shutdown`: an opt-in helper that
  connects SIGTERM and SIGINT to `graph.aclose()`. Nothing installs signal handlers unless you
  call it.
- `compile(..., owns_resources=True)`: the default is unchanged. Resources (checkpointer,
  store, publisher) are now counted per holder, so `aclose()` releases one only when the last
  graph using it closes. Pass `owns_resources=False` to keep `aclose()` from closing resources
  you manage yourself.
- `ExecutionState.run_id` records which run wrote the state. A state saved by an older version
  loads with `run_id` as `None`.
- `NothingToResumeError` (`INTERRUPT_003`, a resume sent to a thread that is not paused) and
  `InterruptPendingError` (`INTERRUPT_004`, a plain new message sent to a thread paused at
  `interrupt()`). Both are also `ValueError`, so existing handlers keep working.
- `ToolNode(max_result_chars=...)`: cuts a local tool's text result and appends a marker.
  Off by default.
- `ToolNode(mcp_include=..., mcp_exclude=...)`: allow-list and deny-list of MCP tool names. A
  tool that is hidden is also refused when called.
- `create_fetch_url(allowed_hosts)`, `create_google_web_search(model)` and
  `create_vertex_ai_search(datastore, model)`: tool factories that fix the host list, model or
  datastore in code, so the model cannot choose them.
- `RetryConfig.jitter` (default `0.25`) and `RetryConfig.max_concurrent_calls` (default `None`,
  no cap): spread retries out, and optionally cap in-flight LLM requests per provider for one
  Agent.
- `TENXGRAPH_LLM_TIMEOUT` (seconds) sets the default LLM client timeout. The old
  `AGENTFLOW_LLM_TIMEOUT` still works. A `timeout=` in an Agent's `llm_kwargs` sets it per
  agent, and Google clients now honour it.
- Message metadata `stop_issue` is set to `truncated`, `context_window_exceeded`,
  `content_filter` or `malformed_tool_call` when a provider (Anthropic, Google, OpenAI) stopped
  for one of those reasons, and a warning is logged.
- `StructuredOutputAgent` marks a reply that still fails validation after the last attempt with
  `metadata["structured_output_valid"] = False` and `metadata["structured_output_error"]`.
- Opt-in publisher setting `publish_timeout` (seconds) bounds one send; a send that times out is
  logged and dropped.
- `redis_key_prefix` option on `PgCheckpointer`: a prefix for its Redis keys, so deployments
  that share one Redis do not read each other's cache. Ids in keys are percent-encoded.
- Safe default timeouts on `PgCheckpointer`: 30 s to get a pooled connection
  (`pg_acquire_timeout`), 60 s per query, 30 s per new connection, and Redis socket timeouts
  (5 s connect, 10 s read) with a 30 s health check. Override them key by key with
  `pool_config` and `redis_pool_config`.
- Public `aclear_state_cache(config)` (and sync `clear_state_cache`) on the checkpointers.
- Storage limits: `MediaConfig.max_image_pixels` (default 50,000,000) refuses oversized images
  before decoding, and the media stores accept an opt-in size cap (`MediaTooLargeError`).
- The SQLite and in-memory checkpointers now keep the tool ledger, so a run killed mid-node does
  not run finished tools again on resume. A custom checkpointer without ledger support logs one
  warning per class.
- `compile(node_aliases={old: new})`: an opt-in map for threads stored at a node that a deploy
  renamed or removed. A thread stored at `old` resumes at `new`, in `invoke` and `stream`.
- `StoredNodeMissingError` (`GRAPH_NODE_MISSING_001`): a thread stored at a node the graph no
  longer has (and no alias covers) fails with this error, naming the node and the thread. Before,
  every run on it raised a bare `KeyError`. `10xgraph-api` answers 409.
- `StateFormatVersionError` (`STATE_FORMAT_001`): raised when stored state was written by a
  newer version than the code reads, for example after a rollback. The state is left untouched.
  `10xgraph-api` answers 500 with a message that says to upgrade.
- `ReasoningBlock.signature` and `ReasoningBlock.redacted_data` (both optional): hold the
  signature of an Anthropic `thinking` block and the payload of a `redacted_thinking` block.
- `BaseMediaStore.modify_metadata(key, fn)`: reads the metadata, applies `fn` and writes the
  result as one step. The local and in-memory stores make it atomic. The cloud stores re-read
  just before the write and retry up to 5 times, which leaves a small window.
  `BaseMediaStore.list_keys_for_owner(owner_id)` lists the keys one owner stored. A custom store
  inherits a non-atomic `modify_metadata` and a `list_keys_for_owner` built on `list_keys`.
- Opt-in `stream_idle_timeout` (per agent in `llm_kwargs`, `set_default_stream_idle_timeout()`
  or `TENXGRAPH_LLM_STREAM_IDLE_TIMEOUT`; no default) and `LLMStreamIdleTimeoutError`. A stall
  before the first chunk is retried by `RetryConfig`; a stall after any chunk is not retried.
- Cloud signed-URL downloads in `CloudMediaStore` have a 10 s connect timeout and a 60 s idle
  timeout.
- More `ToolNode` MCP options, all off or unchanged by default: `mcp_prefix` (a string put in
  front of every MCP tool name the model sees, removed again before the server call, so two
  servers with the same tool name can share a graph), `mcp_max_result_chars` (cuts one MCP tool
  result and appends a marker) and `mcp_max_description_chars` (cuts a tool description). Next to
  `mcp_include` and `mcp_exclude`, which match the server's own names without the prefix.
- `await tool_node.refresh_mcp_tools()` fetches the MCP tool list again (see `mcp_cache_tools`
  under Changed).
- Connect timeouts, each bounding only the connect and never a call: `ToolNode(mcp_connect_timeout=30.0)`
  and `GeminiLiveClient(connect_timeout=30.0)`; pass `None` to wait without a limit. The
  RabbitMQ publisher's `connection_timeout` (default 10 s) now also bounds opening the channel
  and exchange, and a failed open closes the half-open connection, so the next publish retries
  on a clean connection.
- Publisher setting `state_snapshot_limit` (see Changed).
- Opt-in cross-process LLM concurrency cap on `RetryConfig`: `distributed_max_concurrent_calls`
  (default `None`, off) limits in-flight requests per provider and model across every process
  that shares one Redis, on top of `max_concurrent_calls`. It needs `distributed_redis` (an async
  client) or `distributed_redis_url`. Also `distributed_lease_seconds` (default `300.0`) and
  `distributed_key_prefix` (default `""`). It fails open: when Redis cannot be reached a WARNING
  is logged (at most every 30 s) and the call runs without a slot, so a Redis outage never blocks
  runs. A slot is released on cancel and expires with its lease after a crash.

### Changed

- **The HTML eval report is redesigned as a scorecard.** `HTMLReporter` (and
  `ReporterConfig(html=True)`) now writes one self-contained file that opens offline: a verdict
  against the pass-rate gate, headline numbers (pass rate, average score, latency, tokens) with
  deltas against the previous run, a regression line, a cases-by-criteria matrix with pass, near
  and fail marks and changed-cell markers, filters, search and sort, a detail drawer per case
  (previous and next, keyboard, `#eval_id` deep link), run-over-run bars, a cost and latency strip,
  collapsed run details, light and dark themes, a phone layout and print styles. The previous run
  is the most recent earlier JSON report of the same `eval_set_id` in the same folder (excluding
  the current run's file); without one the page says so. The gate is read from
  `report.metadata["pass_rate_threshold"]` (set by `10xgraph eval`) and is 100% ("all cases must
  pass") when absent. Report data is embedded as JSON and rendered with text escaping, and a
  Content-Security-Policy meta tag blocks network requests and any script injected through
  report content. `HTMLReporter.to_html` takes optional `baseline_dir` and `exclude`; the private
  `_render_case` helper and the `_html_render` module are removed.

- **Every converter orders blocks the same way.** A message from the OpenAI chat, OpenAI
  Responses, Google GenAI and Anthropic converters has its blocks in the order reasoning, text,
  media (image, audio, video), tool calls. This holds for a non-stream response, each streaming
  delta and the final streamed message. Before, some stream finals and OpenAI deltas put text
  ahead of reasoning. Each delta carries only its new reasoning piece as a `ReasoningBlock`, and
  the OpenAI Responses stream now emits reasoning deltas too (it used to hold them back until
  the final message).
- **A cancelled, crashed or shut-down run leaves a usable thread.** On cancel or shutdown the
  run closes unanswered tool calls (the recorded result if there is one, else a failed result
  saying the run was interrupted), marks the thread INTERRUPTED (reason `cancelled`,
  `shutdown` or `crashed`), saves it, and re-raises the cancel. A new user message then starts
  a new turn on the repaired history; a call with no new input resumes at the saved node.
- **Crash recovery on load.** If the saved state says RUNNING but no live run owns it (the
  active-run marker now expires within 60 s and is refreshed every 20 s), the next load repairs
  the thread from the cached state, marks it INTERRUPTED with reason `crashed`, saves it, logs a
  warning and counts `tenxgraph.run.recovered`.
- The stream path (`astream`, the API path) now matches `invoke`: the tool ledger (a resumed
  run does not run finished tools again), per-tool state isolation for parallel tools,
  cancel and stop handling inside a node, before and after callbacks, node events and metrics,
  and log context ids on every line. Parallel tool results are appended in call order on both
  paths.
- `stop` works without a checkpointer or Redis: the stop check also reads an in-process flag.
- Background tasks wait up to the timeout first and are cancelled only for what is left.
- A circuit breaker in the half-open state lets one trial call through at a time.
- Retries honour `Retry-After` when it is longer than the computed delay (the next fallback is
  tried if it is longer than `max_delay`), never retry a 429 that reports a spent quota or
  spend cap, and the SDK's own retries no longer multiply the attempts.
- OpenAI streaming asks for token usage (`stream_options.include_usage`), so streamed runs
  report usage for OpenAI chat and Google.
- A provider fallback uses the fallback provider's converter and credentials. It no longer
  sends one provider's `api_key` or `base_url` to another, and it no longer rewrites the
  shared Agent while other runs use it.
- `SummaryContextManager` never starts the kept context with a tool result, counts tool
  results and images in its token estimate, and bounds the summary size.
- `PgCheckpointer` uses timezone-aware timestamps everywhere (local-time stamps are no longer
  stored), and a message that is saved again updates its tool calls, role and metadata.
- Thread delete is best effort on the cache: a Redis failure after the Postgres delete no
  longer raises, the tool ledger and cache keys are removed, and a stale cache entry cannot
  bring a deleted thread back.
- Deleting a message rewrites the saved state versions and the cache too, so the message is
  gone from the next run and from the state.
- After a Redis restore from an older snapshot, the cached state is checked against Postgres
  at run start and dropped if older. Before, every active thread failed its next turn once
  with a state conflict.
- `Mem0Store` applies the owner `user_id` last and drops the reserved filter keys `user_id`,
  `agent_id`, `app_id` and `run_id` from caller `filters`, so a `filters` value cannot replace
  the owner. `Mem0Store` works with mem0ai 2.x.
- `max_tokens` for the memory preload is now applied by the Qdrant and Mem0 stores.
- `compile()` logs a WARNING when it falls back to the in-memory checkpointer, and the shared
  anonymous user (no `user_id`) is warned once per graph. Both were debug lines.
- Importing `tenxgraph` no longer needs a home directory. The home path is resolved when first
  used.
- The module-level file tools (`file_read`, `file_write`, `file_search`) still work until 2.0
  and now log a deprecation warning that names `create_file_tools`. File writes are atomic,
  and `.git-credentials` and `terraform.tfstate` are on the default deny list.
- A duplicate tool name on one `ToolNode` logs a warning when the earlier tool is replaced.
- A tool that returns a `ToolResult` keeps its message on `invoke` as it does on `stream`.
- The `message` stream chunk can carry `data` beside the message for tool results (`status`,
  `tool_name`, `args`, `tool_call_id`), at every granularity. A consumer should not assume the
  two are exclusive.
- Pause chunks (`interrupt()`, `interrupt_before`, `interrupt_after`, client tool) are passed at
  every stream granularity, and chunk timestamps are timezone-aware.
- **Anthropic thinking is stored first in the message and replayed unchanged.** Signed
  `thinking` blocks and `redacted_thinking` blocks are kept on `ReasoningBlock` (with
  `signature` or `redacted_data`) and sent back, before the text and `tool_use` blocks, on the
  next Anthropic call. Anthropic rejects a tool turn whose signed thinking was dropped or
  edited. Other providers never receive these blocks. A message saved by an earlier version has
  no signature, so its thinking is not replayed to Anthropic.
- The core checkpointer's Redis pool is capped at 50 connections, and a caller past the cap
  waits up to 5 s for a free one (`BlockingConnectionPool`, `max_connections` and `timeout` in
  the default `redis_pool_config`). Override either key through `redis_pool_config`.
- A media folder shared by several workers now contains a `.meta.lock` file, used to serialize
  sidecar updates across processes (POSIX only).
- `CloudMediaStore` keeps a per-owner index object (`{prefix}/owners/<hash>.index.json`), so
  `list_keys_for_owner` works on S3 and GCS. Files stored before this release, or stored with no
  owner, are not in the index. `10xgraph-api` uses it for `GET /v1/files`, the per-user
  quotas and release on thread delete.
- **MCP tool lists are cached per `ToolNode`** (`mcp_cache_tools=True`, the default). The
  server's tool list is fetched once and reused, which saves a connect, ping and list round trip
  on every model call. A tool added on the server later appears only after
  `await tool_node.refresh_mcp_tools()`. A failed listing is never cached. Pass
  `mcp_cache_tools=False` to list on every call as before.
- **Large `data["state"]` in published events is replaced by a summary.** When it is over the
  publisher's `state_snapshot_limit` (JSON characters, default `65536`; `0` or `None` disables),
  `data["state"]` becomes `{truncated, bytes, fields, context_count, context, context_summary,
  execution_meta}`: `context` lists the `message_id` and `role` of the last 50 messages (system
  messages kept up to 8 KiB), `context_summary` is cut to 1000 characters and `execution_meta` is
  kept if under 8 KiB. `data["messages"]` is not bounded. A queued event no longer holds the
  full state during a broker stall.
- **Realtime tool calls run once per thread.** A tool call from a live session is recorded in the
  tool ledger under the thread id, a hash of the tool name and arguments, and the provider call
  id. A new session on the same thread that re-sends an identical call gets the recorded result
  instead of running the tool again. With no thread id the key falls back to the session id, and
  a call with no provider id cannot replay.

### Fixed

- **Multi-agent handoffs no longer leave an unanswered tool call.** `SwarmAgent` routed a
  `transfer_to_<name>` call straight to the target without a tool result, so the target's
  request ended on the caller's assistant tool call: Gemini answered with an empty reply
  (`NODE_004`), OpenAI and Anthropic reject it. Each member with handoff targets now has a
  `<NAME>_HANDOFF` node (`TRIAGE -> TRIAGE_HANDOFF -> PLANNER` in the graph) that answers every
  tool call of the handoff message before the target runs: "Transferred to <target>." for the
  handoff and "Not run: control was transferred to <target>." for a second handoff call in
  the same message. The results are saved to state and the
  messages table like any node output. A handoff now takes one more graph step, which counts
  toward `recursion_limit`. `SupervisorTeamAgent` requests no longer end on an assistant turn:
  the supervisor's request (after a worker's reply) and a later worker's request (after the
  previous worker's reply) get one transient user message that is never saved. Worker nodes now
  wrap the agent you pass in (`graph.nodes[name].func.agent`).

- **Realtime (AudioAgent) sessions now save their transcripts and state.** A live
  `gemini-3.8-live` session left its thread with no messages and no state: transcripts were
  saved only on a transcription chunk marked `finished`, which Gemini Live does not send (a
  turn just ends with `turn_complete`), so the text stayed in a buffer and was dropped when the
  session ended. Now the end of each turn saves the user's and the model's text (one message
  each, plus one complete transcript event per side for the client), a barge-in keeps what the
  user said and drops only the model's cut-off reply, and the end of a session (also a client
  that disconnects mid-turn) saves what is left. The session also saves its state as the
  thread's checkpoint after each turn. A session now starts from the thread's saved state (a
  later session or text turn continues the conversation instead of overwriting it) or from a
  deep copy of the graph's state: sessions used to append to one shared state object, so
  concurrent users' transcripts could mix. Audio is still never stored.

- **A handoff no longer drops the other tool calls of its message.** OpenAI often calls a tool
  and a handoff together (`get_weather` + `transfer_to_planner`). The handoff used to answer
  the tool call "Not run", so the target had no weather and handed back: a live `gpt-6-luna`
  swarm bounced WEATHER, PLANNER, WEATHER, PLANNER. The other calls now run first (in the
  ToolNode handoff path and, through the member's `<NAME>_TOOL` node, in `SwarmAgent`), then
  control moves on and the target sees their results. A second handoff call in the same
  message still gets "Not run: control was transferred to <target>.", and that tool message
  carries `metadata["not_run"] = True` (the playground shows it as "Skipped").

- **`SupervisorTeamAgent` no longer re-dispatches workers that already answered.** The
  supervisor's request held the workers' replies as unlabelled assistant turns, followed by its
  past decisions replayed as bare `WEATHER` / `PLANNER` assistant turns; it could not tell who
  said what or that the user was answered (a live Gemini run ran WEATHER, PLANNER, WEATHER,
  PLANNER, PLANNER). Its own call now sees earlier turns as plain text and one user message
  naming each worker's reply in order ("WEATHER replied: ...") and the workers already run.
  Naming the worker that just ran again counts as `FINISH`. When the supervisor finishes before
  any worker ran (a greeting), a new `RESPOND` node answers the user, so the turn is never
  blank. `RESPOND` and `PRE_SUPERVISOR` are now reserved worker names.

- **`StructuredOutputAgent` with tools now works on Google.** Gemini cannot combine function
  calling with a JSON output schema in one request, so the prebuilt agent failed on the first
  message. On Google with tools it now runs GENERATE (tools, plain text) then a FORMAT step
  (schema, no tools) that turns the conversation into the schema; validation and REPAIR loop
  over FORMAT. OpenAI, Anthropic and Google without tools are unchanged. The `Agent` guard for
  tools plus `output_schema` on Google is unchanged.

- **`PlanActReflectAgent` no longer loops on empty replies.** The reflector's request used to end
  with the planner's assistant message (and a later planner request with the reflector's), which
  makes Gemini return an empty reply and Anthropic drop the turn. Both now get one transient user
  message at the end of the request (never saved to state), so a greeting such as "HI" ends after
  2 LLM calls. `[DONE]` is recorded in a per-turn flag and removed from the saved reply, and an
  empty reflector reply ends the turn instead of looping to `max_iterations`.
- **Empty model replies are retried once, then fail the node.** An `Agent` reply with no text, no
  tool calls and no media (reasoning alone counts as empty) is no longer saved. The call is
  repeated once, with a warning that names the node, the model and the tokens the empty call
  used; a second empty reply raises a `NodeError` (`NODE_004`). Streaming holds back the empty
  final message, so no half-built reply reaches state. Saved empty assistant replies in old
  threads are skipped when the request is built (a reasoning block with an Anthropic signature is
  kept).
- **Tool result messages carry `metadata["node_name"]`**, and so do assistant replies on the
  `ainvoke` path (they only had it when streamed), so clients can group a turn by node.
- **A cancelled run releases its thread only after its cancel save.** Under a level-triggered
  cancel (anyio, Starlette on a client disconnect) the INTERRUPTED save used to go on in the
  background while the active-run marker was already released, so a quick next run on the
  thread could race it. The waiter now waits for the save (up to 10 s) while cancelled, inside a
  shielded anyio scope when anyio is loaded, then re-raises the cancel; the marker release is
  waited for the same way. A cancel during the run's own save waits for that save instead of
  starting a second one beside it, and a run whose COMPLETED state was saved stays completed.
  The INTERRUPTED event is published after the save and carries the saved state. Detached
  cleanup is cancelled (ERROR log) after 20 s. `aend_run` clears the marker and stop request
  with a compare-and-delete in one step (new `BaseCheckpointer.aclear_cache_value_if`, atomic in
  the in-memory, SQLite and Redis caches), and the run heartbeat restores its own missing marker.
- **Evals: tool arguments in a trajectory are the ones the model sent.** The recorded args used
  to include injected parameters (`emit`, `config`, `store`, an `Inject[...]` service, ...), so
  an exact `check_args=True` trajectory failed for any tool that declared one. The collector now
  records the call's own arguments and drops every name in `INJECTABLE_PARAMS`.
- **Evals: `EvalCaseResult.expected`.** Each case result now carries what the case expected
  (`response`, `tool_calls`, `node_order`, `input`), so a JSON report shows the expectations next
  to the outcome, failures included. Older reports without the field still load.
- **Evals: `SimulationResult.duration_seconds`.** `UserSimulator.run` now times the whole
  simulation, so callers that build case results from simulations no longer report `0.0s`.
- The user simulator no longer duplicates history on a graph with a checkpointer. It sent the
  whole conversation every turn, and each resend got new message ids, so the thread grew by
  1, 1+2, 1+2+3 messages. With a checkpointer only the new user message is sent; without one
  the full history is still sent.
- The Gemini call used by evaluation (`tenxgraph.core.llm.caller`) joins the text parts itself
  instead of reading `response.text`, so the "non-text parts in the response" warning is gone.
- Async id factories (`AsyncIDGenerator`) work for `Message()` and for run and thread ids.
- Sync `invoke()` and `stream()` wait up to `shutdown_timeout` (default 30 s) for background
  events, such as publisher events, before they return. Before, that work was dropped.
- A stopped thread resumes, and later messages are no longer dropped. A stop no longer loses
  the output of the node that was running.
- Streamed runs that pause (`interrupt()`, `interrupt_before`, `interrupt_after`, client tool)
  keep the run's messages in the messages table. Before, a pause saved none of them.
- Resuming a thread paused at `interrupt()` works on `PgCheckpointer`. It failed with
  `user_id must be provided in config` on the HTTP and WebSocket routes.
- A handoff gives the target agent a tool result for the handoff call. Before, the context ended
  with an unanswered tool call, which OpenAI and Anthropic reject.
- `PlanActReflectAgent` with tools no longer sends an unanswered tool call to the reflector.
- The recursion limit is exact (it was off by one), the turn after a recursion error is no
  longer swallowed, and `astream` raises `GraphRecursionError` instead of a pydantic
  `ValidationError`.
- `invoke` no longer sends the messages of a resume twice, and a cancel that arrives while the
  run is already cancelling is no longer absorbed.
- Stream runs log the run, thread and user ids, and log context ids no longer leak from one run
  to the next.
- A stop request that fails because Redis is down is no longer lost.
- OpenAI chat streaming no longer saves a half answer as complete when the connection drops or
  a 429 arrives mid-reply; the error is raised.
- Anthropic: a `max_tokens` stop that ends in a `tool_use` block no longer runs the tool.
- Deeply nested tool arguments return a tool error instead of failing the run with
  `RecursionError`.
- Google: `output_type="audio"` works, and a bad `data:` URL no longer fails the run.
- OpenAI batch `results()` includes failed requests from the error file.
- A reranker error in `RAGAgent` no longer fails the run.
- Swarm and supervisor member names no longer have to be upper case, and a supervisor `FINISH`
  is matched as a whole word, not as a substring of a worker name.
- `@tool(name)` works as documented, bound methods are supported, and the converter handles
  `mpeg` and `x-wav` audio, `file_id` image parts and `resolve_media_refs` correctly.
- `QdrantStore` filters on `user_id` and `thread_id` equal to `0`. The in-memory tool ledger is
  scoped to the thread owner.
- `OpenAI` embeddings: `text-embedding-3-large` maps to 3072 dimensions, and
  `text-embedding-ada-002` and other common models are known.
- `SqliteCheckpointer` rolls back a failed write, and claiming a new thread and writing it are
  atomic.
- Redis `alist_cache_keys` no longer matches keys that only share a prefix, and a retry after a
  lost commit acknowledgement is no longer a `StaleStateError`.
- The cloud media store no longer reports a delete as done when only the sidecar was removed,
  and logs errors it used to hide.
- `LocalFileMediaStore` writes data and sidecar files through a temp file and a rename, so a
  crash no longer leaves a truncated file. Deleting a thread's media also removes the Google
  File API uploads recorded for it.
- Realtime: transcript saves no longer overwrite the thread's metadata and name after a
  transient read error, setup failures and cancels no longer leak a billed Gemini session, and
  tool calls with non-object arguments return an error result.
- Kafka and Redis publishers retry after a failed first start and stay closed once closed. A
  late event after close no longer reopens a connection.
- An OTEL LLM span ends on failed and streaming calls, so the span registry no longer grows.
  A streaming span ends with the stream and carries the token usage, finish reason and model
  of the final message.
- README: dropped the pre-1.0 line and corrected the stream format.
- A running LLM call renews its slot in the distributed concurrency cap
  (`distributed_max_concurrent_calls`) every third of `distributed_lease_seconds`, so a call that
  outlives the lease keeps counting against the cap. Before, a slow call lost its slot. A renewal
  error is logged and does not fail the call.
- A stalled stream is retried before its first chunk and never after it (see `stream_idle_timeout`
  under Added), so streamed output is not duplicated.
- The `END` event of an MCP tool call carries the result that `after_invoke` returned, like a
  local tool. Before, it carried the raw MCP result.
- `CloudMediaStore.get_direct_url` takes the signed URL's file extension from the stored
  metadata. The caller's `mime_type` is ignored.
- Link updates on media (`modify_metadata`) are atomic on the local and in-memory stores, so
  concurrent workers no longer lose thread links.

### Security

- Exceptions redact secret-looking keys (`api_key`, tokens, passwords) and URLs in their
  context before logging it.
- Node error events no longer carry the text of provider or other foreign exceptions. The event
  shows the exception type and says the details are in the server log; the full text is logged.
- Log lines neutralise CR and LF in model-controlled values such as tool names.
- `Mem0Store` cannot be pointed at another user by a client `filters` value (see Changed).
- Dependency floors now exclude versions with known advisories: Pillow 13.0.0, mcp 1.30.0,
  OpenTelemetry 1.45.1. See Upgrade notes.
- `PromptInjectionValidator` strict mode no longer blocks ordinary text.

## [1.0.0rc1] - 2026-10-28

First 1.0 release candidate. The public API is now stable under the versioning policy above,
and the version is aligned with `10xgraph-api` and `10xgraph-client` 1.0. This candidate is
the 0.10.3 code plus the hardening round listed below, which includes the breaking changes
here. Each bullet gives the migration; the detailed entry further down has the full text.

### Breaking

- **The graph-level memory API is removed** (`MemoryIntegration`, `create_memory_preload_node`,
  `get_memory_system_prompt`, `memory_tool`). Pass `Agent(memory=MemoryConfig(...))` instead.
  See Removed.
- **`fetch_and_cache` is removed** from `tenxgraph.storage.media.temp_cache`. Download through
  `fetch_media_url` with a `MediaFetchPolicy`. See Removed.
- **Store operations with a missing or empty `user_id` now raise `ValueError`**, on `QdrantStore`
  and `Mem0Store`. Pass a concrete `user_id`, or `system_user_id(name)` for shared data. Re-index
  records written with an empty owner. `RAGAgent` now searches as
  `system_user_id("knowledge_base")` unless `store_config` sets `user_id`, and
  `store_config_builder` output must include `user_id`. See Changed and the Q5 entry in Security.
- **Agent memory is owned by `system_user_id(agent_id)` (or `app_id`), not the current user.**
  `Agent(memory=...)` raises `ValueError` if agent memory is enabled with neither id. Memories
  written under a user by the old behaviour are not found; re-store them. See Security.
- **A `collection` in the run config no longer overrides the memory scope's collection.** Set
  `collection` on `UserMemoryConfig` / `AgentMemoryConfig`. See Security (Q6).
- **`PRELOAD` memories are no longer a system message.** They arrive as a fenced block in the
  latest user message of each request and are never saved to `state.context`.
  `Agent._build_memory_prompts` is replaced by `Agent._build_memory_block`; prompts or tests that
  look for the `[Long-term Memory Context]` system message need updating. See Security.
- **`RAGAgent` changes its context handling.** Retrieved documents go into a request-only block,
  `state.context` is no longer rewritten, and `execution_meta.internal_data["rag_docs"]` and
  `_RAG_DOCS_KEY` are gone. A custom agent passed to `RAGAgent` must accept the keyword-only
  `request_blocks` argument on `execute`, or `RAGAgent` raises `TypeError`. See Security.
- **Skill and memory text is no longer run through `str.format`.** Only your own `system_prompt`
  is interpolated, and a bad placeholder is logged instead of failing the run. Remove any
  reliance on `{placeholder}` expansion inside skill or memory text. See Security.
- **Handoffs need a registered handoff tool.** A `transfer_to_<X>` call only jumps if a tool made
  with `create_handoff_tool` is on that ToolNode; a `Command.goto` to an unknown node raises
  `GraphError` (`GRAPH_007`). `SwarmAgent` now enforces `can_handoff_to`. See Security and
  Fixed.
- **Tools hidden by `tools_tags` can no longer be called.** Calls to a tool the Agent did not
  advertise return "Tool not found". Add the tag or the tool if the call was intended. See
  Security.
- **An MCP tool with the same name as a local or remote tool raises `ValueError`.** Rename one of
  them. See Security.
- **Parallel tool calls are capped at 10 per turn by default.** Set `max_parallel_tool_calls` in
  the run config, or `TENXGRAPH_MAX_PARALLEL_TOOL_CALLS`, to change it. See Security.
- **Unexpected tool errors no longer reach the model verbatim.** The model sees
  `Tool '<name>' failed with an internal error (reference <ref>)` and the full error is logged.
  A tool that wants the model to see a message must return it as a string; a raised
  `ValueError` is no longer shown. See Security.
- **A tool call with invalid JSON arguments is answered with a tool error** instead of running
  with empty arguments. See Fixed.
- **Multi-agent loop limits are per turn.** `max_rounds`, `max_iterations` and `max_attempts`
  reset on each new turn instead of being a lifetime budget per thread. See Fixed.
- **Resume is checked.** A `resume_interrupt_id` that is not the pending interrupt's raises
  `InterruptMismatchError`, and a value that fails the interrupt's `response_schema` raises
  `ResumeValidationError`. See Security.
- **File tools refuse secret files and `file_search` stays inside the root.** `.env`, keys,
  `.ssh`, `.aws`, `.git` and similar paths are refused or skipped for any root; use
  `create_file_tools(root, deny=..., use_default_deny=False)` to change that. Prefer
  `create_file_tools(root=...)` to the run-config root, which is deprecated. See Security and
  Deprecated.
- **`fetch_url` no longer follows redirects to internal addresses or uses `HTTP_PROXY` /
  `HTTPS_PROXY`.** Fetches that relied on a proxy or on an internal redirect now fail. See
  Security.
- **Client-supplied media URLs are fetched under a `MediaFetchPolicy`.** Only `http` and `https`
  to public hosts, 20 MB, 10 s, and `image/*` and `application/pdf` by default. Audio and video
  links on Google graphs stop downloading until you allow those content types; pass
  `compile(media_policy=MediaFetchPolicy(...))`. The private `_convert_to_google_format` no
  longer downloads. See Security and Changed.
- **`CloudMediaStore` raises `ValueError` for storage keys that are not 32 lowercase hex
  characters.** Objects stored under other keys by custom code are no longer reachable through
  the store. See Security.
- **Local media and the SQLite checkpointer create owner-only files** (`0o700` folders, `0o600`
  files). Other OS users or services that read them lose access; widen the mode yourself if
  that is intended. See Security.
- **Checkpointer list methods validate paging.** `alist_messages` and `alist_threads` raise
  `ValueError` for a negative `limit` or `offset`, and an explicit `limit` is lowered to
  `MAX_LIST_LIMIT` (1000). See Security.
- **`PgCheckpointer` schema version 4 changes the messages primary key to
  `(thread_id, message_id)`**, and `aget_message` / `adelete_message` require `thread_id` in the
  config. The migration runs automatically on first setup. `aget_tool_result` is scoped to the
  thread owner. See Security and Changed.
- **Checkpointers apply the owner rule to state-cache writes**, and the in-memory checkpointer
  returns copies of state, so mutating a loaded state no longer changes the store until you save
  it. See Security and Fixed.
- **Tool metrics label unknown tools as `"unknown"`** instead of the model-invented name. See
  Security.
- **`PromptInjectionValidator` behaves differently.** Non-strict mode scans only the first
  `max_length` characters, strict mode now raises on bad base64, and text in other languages is
  no longer rejected. See Security and Fixed.
- **`QdrantStore.aforget_memory` applies `memory_type`, `category` and `filters`**, and
  `Mem0Store.aforget_memory` raises `ValueError` if any of them is passed. The `qdrant` extra
  needs `qdrant-client>=1.10.0` and a Qdrant server 1.10 or newer. See Fixed.
- **The old example folders and the `a2a_sdk` example are removed**; use the reorganised
  `examples/` folders. See Changed.

### Security

- **Tool metrics no longer label by model-invented tool names.** `tenxgraph.tool.calls`,
  `.duration`, `.timeouts` and `.errors` carried the `tool` attribute from the name the model
  wrote, before any check that such a tool exists, so a model emitting many made-up names created
  an unbounded number of series. The label is now the real name only for a tool the ToolNode can
  run (local, MCP, configured or per-run client, handoff), and `"unknown"` otherwise. The call
  still returns the usual "Tool not found" result.
- **`RedisPublisher` connection errors no longer include the Redis password.** The error text
  shows the URL through `redact_url` (userinfo, query and fragment removed), so
  `redis://:s3cret@host:6379/0` appears as `redis://host:6379/0`. `redact_url` now also keeps
  IPv6 hosts in brackets and returns `<invalid url>` for a string with no scheme or host.
- **Checkpointer list methods validate and bound paging.** `alist_messages` and `alist_threads` on
  the SQLite, in-memory and Postgres checkpointers now raise `ValueError` for a negative `limit`
  or `offset` (a negative `limit` used to drop the last row) and lower an explicit `limit` to
  `MAX_LIST_LIMIT` (1000, in `base_checkpointer`, the same cap as the API). `limit=None` or `0`
  still returns everything. SQLite applies `LIMIT`/`OFFSET` and the owner filter in SQL instead of
  loading every row (with `search`, rows are streamed and reading stops once the page is full).
- **`PgCheckpointer` tool-ledger reads are scoped to the thread owner.** `aget_tool_result`
  now applies the same owner filter as the message reads: with `enforce_user_isolation=True`
  (the default) or an `authz` scope of `owner`, a caller naming another user's thread gets no
  recorded result. With isolation off the query skips the ownership join. Writes were already
  checked.
- **`PromptInjectionValidator` no longer backtracks for minutes or hours on crafted input.**
  The `ignore` / `disregard` / `forget` / `override` patterns took polynomial time on long
  whitespace runs (CWE-1333), and `validate()` runs them on the event loop, so one message
  could block a server. They and the delimiter, `<system>` and template patterns now match in
  linear time. `max_length` is enforced before any pattern runs: non-strict mode, which used to
  scan input of any length, now scans only the first `max_length` characters.
- The `<think>`, `<reasoning>` and `<thought>` tag parsers used on model output now run in
  linear time instead of quadratic time on unclosed tags.
- **`fetch_url` no longer follows redirects to internal addresses.** It checked only the first
  host, then let urllib follow redirects anywhere, so a public page that redirected to
  `127.0.0.1` or `169.254.169.254` was fetched and returned to the model. It also resolved the
  name twice, which DNS rebinding could exploit. Every redirect hop is now checked (5 at most),
  the connection goes to the exact address that was checked, and an address must be globally
  routable, which also rejects 100.64.0.0/10 (shared address space, including Alibaba Cloud
  metadata) and the NAT64 and IPv4-compatible forms of internal addresses. `fetch_url` no
  longer uses `HTTP_PROXY` / `HTTPS_PROXY`.
- **New `create_file_tools(root)`** returns `file_read`, `file_write` and `file_search` locked to
  a directory set in code. The module-level file tools read their root from the run config
  (`file_tool_root` / `workspace_root`), so anyone who can set the run config could point them
  at any directory on the server.

- **Handoffs only go where a registered handoff tool says.** Any tool call named
  `transfer_to_<X>` used to jump the run to node `<X>`, so a model steered by prompt injection
  could skip edges and approval steps. A handoff now needs a tool made with
  `create_handoff_tool` registered on that ToolNode; other `transfer_to_*` calls are normal tool
  calls. A `Command.goto` to a name that is not a node or END raises `GraphError`
  (`GRAPH_007`) instead of a raw `KeyError`.
- **Tools hidden by `tools_tags` can no longer be called.** The Agent advertised only the tagged
  tools, but the ToolNode ran any registered name the model guessed. The Agent now records the
  advertised names on its message (`metadata["allowed_tools"]`) and other calls get the usual
  "Tool not found" error.
- **An MCP tool with the same name as a local or remote tool raises `ValueError`.** The MCP tool
  used to win silently, so calls meant for a local tool went to the MCP server.
- **Parallel tool calls are capped.** One model turn with hundreds of tool calls started them
  all at once, each with its own copy of the state (300 calls took 2.4 GB). At most
  `max_parallel_tool_calls` run at once: set it in the run config, or with the
  `TENXGRAPH_MAX_PARALLEL_TOOL_CALLS` env var; the default is 10.
- **`file_search` no longer follows symlinks out of the workspace root.** A symlink inside the
  root that pointed outside it was searched, and matching lines from the outside file were
  returned as previews. Each file is now resolved and must stay under the root, the same rule
  `file_read` applies.
- **Tool failures no longer send the raw exception text to the model.** An unexpected error in
  a local or MCP tool put its message (database URLs, file paths, SQL errors) into the tool
  result and events. The model now gets `Tool '<name>' failed with an internal error
  (reference <ref>)`, and the full error with the same reference is logged server-side.
  Argument errors, "Tool not found" and timeouts still reach the model as before. A tool that
  wants the model to see a message should return it as a string; a raised `ValueError` is no
  longer shown verbatim.
- **Memory and skill text is no longer run through `str.format`.** Skill prompts and long-term
  memory were appended to the system prompts before `{placeholder}` interpolation, so a stored
  memory such as `{execution_meta[current_node]:>200000000}` built a 200 MB prompt, `{context.x}`
  failed every turn, and `{execution_meta}` pasted internal state into the prompt. Only your own
  `system_prompt` is interpolated now; skill and memory text is added after, unchanged (new
  `convert_messages(extra_system_prompts=...)`). A bad placeholder in your own prompt is logged
  and left as is instead of failing the run.

- **The in-memory checkpointer now stores and returns copies of the state.** `aget_state` handed
  out the stored object itself, so a request that was rejected afterwards (for example a new
  message sent to a paused thread) had already merged its input into the stored state. Mutating
  a loaded state no longer changes the store until you save it. This costs one extra copy per
  state read and write.
- **Message ids are unique per thread in Postgres.** `messages.message_id` was the primary key on
  its own, so a user could claim an id that another thread needed later and make that thread's
  checkpoint fail. The key is now `(thread_id, message_id)`. The same id in two threads is
  allowed, and a re-save inside one thread still updates the message.

- **Client-supplied media URLs are no longer downloaded blindly.** On Google graphs and the
  Google File API / inline-bytes paths the server downloaded the URL a client sent, with
  `urllib` or aiohttp: `file:///app/.env`, `http://169.254.169.254/` and `http://localhost:6379`
  all worked, there was no size cap, and the Google conversion blocked the event loop. All four
  download points (`agent_internal/google.py` for images and for documents and video,
  `MediaRefResolver._fetch_external_url` and `MediaResolver._fetch_external_url`) now go through
  `fetch_media_url` and a `MediaFetchPolicy`. Only `http` and `https` are downloaded, the host
  must resolve to public addresses only (cloud metadata addresses are never reachable, not even
  with the private-host opt-in), the connection is pinned to the checked address, every
  redirect is checked again (5 at most), and a size cap (20 MB) and a timeout (10 s) always
  apply. The response Content-Type must match the policy (`image/*` and `application/pdf` by
  default). The download runs off the event loop. A refused or failed URL still becomes a
  `[Failed to load image]` / `[Failed to load media]` text part, and the warning logs the URL
  without its query string. `gs://` URLs are still passed through to Vertex AI unchanged by
  the converters; `MediaFetchPolicy.check_url` lets an API reject them unless the bucket is in
  `allowed_gcs_buckets`, because Vertex AI reads `gs://` with the server's own credentials.
  Behaviour change: audio and video links on Google graphs stop downloading until you allow
  those content types in the policy.
- **`CloudMediaStore` validates storage keys.** It now applies the rule `LocalFileMediaStore`
  already used (32 lowercase hex characters) before building any object path, including in
  `get_direct_url`, which signed whatever path a key gave it when `mime_type` was passed. A
  non-hex key raises `ValueError`. The legacy `agentflow-media` prefix keeps working for valid
  keys. An object that was stored under a non-hex key by custom code would no longer be
  reachable through the store.
- **Local media and the SQLite checkpointer create owner-only files.** `LocalFileMediaStore`
  creates directories with mode `0o700` and data and sidecar files with `0o600`. The sidecar
  holds the owner id and the original filename. `SqliteCheckpointer` creates the database
  folder with `0o700` and the database file with `0o600` before connecting; SQLite gives the
  `-wal` and `-shm` files the mode of the database file. Existing files and folders keep their
  mode. The process umask can only make the modes stricter. Other OS users or services that
  read these files (for example a web server user serving uploads) lose access. No effect on
  Windows.
- **File tools hide secret files.** `file_read`, `file_write` and `file_search`, and the tools
  from `create_file_tools(...)`, now refuse or skip `.env`, `.env.*`, `*.pem`, `*.key`,
  `*.p12`, `*.pfx`, `*key*.json`, `service_account*.json`, `id_rsa*`, `id_ed25519*`, `.netrc`,
  `.npmrc`, `.pypirc` and anything under `.ssh`, `.aws`, `.gnupg` or `.git`, for any root
  including the working-directory default. Matching uses the resolved path and ignores case,
  so symlinks, `..` and case tricks cannot reach them. Read and write return the same "path is
  not allowed" error whether or not the file exists, and search never shows hidden names or
  contents. `create_file_tools` gains `deny=` to add glob patterns and `use_default_deny=False`
  to opt out of the defaults.
- **A resume is now tied to the interrupt it answers and checked against its schema.** A
  retried or stale `{"resume": value}` used to answer whatever prompt was pending, so a double
  click could approve the second prompt of a node that calls `interrupt()` twice, and `{"x":
  "no"}` passed for a boolean schema. Resume input can now carry `resume_interrupt_id`; an id
  that is not the pending interrupt's raises `InterruptMismatchError` and records nothing. The
  value is validated against the interrupt's `response_schema` before it is recorded, and a
  failure raises `ResumeValidationError`. A `None` resume (cancel) is never validated. Both
  errors are also `ValueError`s.
- **A new run no longer erases a stop request meant for a run that is still going.** Starting a
  run cleared the thread's stop flag, so stopping run 1 and starting run 2 before run 1 noticed
  left run 1 running (a runaway tool loop could not be stopped). A stop request now records the
  run it targets, and a new run clears only a stale request.
- **`PRELOAD` memories are no longer a system message.** They were added as a
  `[Long-term Memory Context]` system message, so one poisoned memory (text a tool wrote after
  reading a web page) ran with system rank in every later thread of that user, and because it
  changed each turn it also broke provider prompt caching of the whole history. They now go
  into the latest user message of the request as a separate text part before the user's own
  text, inside a `<user_memories_xxxxxxxx>` block with a random suffix per request, a line
  saying the notes are background information and not instructions, and any `<user_memories`
  text inside a memory escaped. The block is added to the request only: it is never saved to
  `state.context`, so the next turn moves it to the newest user message. Image, video and audio
  output agents get no block. In `LiveAgent` the same fenced block stays in the session
  instruction. `Agent._build_memory_prompts` is replaced by `Agent._build_memory_block`, and
  the memory system prompts describe the new placement. The search runs once per graph
  invocation instead of once per model call: the finished block (or an empty result) is cached
  per Agent under the run's `run_id` and the latest user message, so every model call of a tool
  loop sends the identical block, and the next `invoke`/`stream` call, including a resume,
  searches again. The cache is bounded (256 entries per Agent) and is never saved to state or
  the checkpointer. Without a `run_id` (an Agent called outside a compiled graph) nothing is
  cached.
- **`RAGAgent` no longer breaks out of its context wrapper or saves retrieved documents into
  the thread.** Chunks went into an unescaped `<context>` tag, so a chunk containing
  `</context>` ended the block, and `SYNTHESIZE` replaced the user's message in
  `state.context` with the augmented text: the documents were saved, the thread grew every
  turn, and the original message's images and files, id and metadata were lost. The documents
  now go into a `<retrieved_documents_xxxxxxxx>` block (random suffix per request, a line
  saying they are reference material and not instructions, any `<retrieved_documents` text in a
  chunk escaped) that is added to the latest user message of the outgoing request only, after
  any memory block and before the user's text. `state.context` is not modified. The same helpers
  (`wrap_request_block`, `inject_request_blocks` in `utils/converter.py`) now serve memories and
  documents. `Agent.execute` and `BaseAgent.execute` take a keyword-only
  `request_blocks: list[str] | None`; a custom agent passed to `RAGAgent` must accept it
  (`RAGAgent` raises `TypeError` otherwise). Image, video and audio output agents get no block.
  The document list is no longer written to `execution_meta.internal_data["rag_docs"]` (the
  `_RAG_DOCS_KEY` constant is gone): `RAGAgent` keeps it in a bounded per-run cache (256
  entries, keyed by `run_id` and the user text), written by `RETRIEVE`, updated by `RERANK`,
  read by `SYNTHESIZE`. If the entry is missing (a restart between nodes, eviction, or no
  `run_id`), `SYNTHESIZE` retrieves and reranks again instead of answering without documents.
- **Long-term memory stores no longer fail open on an empty `user_id` (Q5).** `QdrantStore`
  dropped the user filter when `user_id` was empty, so `asearch` and `aget_all` returned every
  user's memories and `aforget_memory` with only a `thread_id` deleted across users. Every
  store operation (search, get, get_all, store, update, delete, forget) now raises `ValueError`
  ("user_id is required for memory operations; use system_user_id(name) for shared data") when the
  `user_id` is missing, `None`, empty or blank. Only the authz policy `scope="none"` still
  skips the user filter on reads. Writes and `aforget_memory` need a concrete `user_id` even
  then, so a record never has an empty owner and a forget never wipes the whole collection.
  `Mem0Store` raises the same message.
- **Shared agent memory was searched as the current user.** `agent_memory_tool` and the preload
  block merged the run's `user_id` into the agent-scope config, so shared agent memory was only
  found under the user who wrote it. Agent scope now uses a reserved shared owner,
  `system_user_id(agent_id)` (or `system_user_id(app_id)`), so one agent's shared memories are
  not mixed with another's. `Agent(memory=...)` raises `ValueError` at setup if agent memory is
  enabled with neither `agent_id` nor `app_id`.
- **A run-config `collection` can no longer override the memory scope's collection (Q6).**
  `QdrantStore` took `config["collection"]` and `_memory_scope_config` merged the run config
  last, so a `collection` in the run config replaced the developer's and was created if missing.
  The scope's own `collection` (new, see Added) now always wins. Then comes a `collection` in
  the merged config from the developer's own code, then the store default.

### Added

- `user_id_mode` (`"raw"` default, `"hash"`, `"mask"`) on `OtelPublisher`, `setup_tracing`,
  `setup_logfire`, `setup_langsmith`, and as a top-level key of the `setup_observability` config
  dict. It controls how `user_id` appears on OTEL spans (and so in Logfire and LangSmith).
  `"hash"` emits 16 hex chars of SHA-256, or of HMAC-SHA256 when the
  `TENXGRAPH_TRACE_USER_ID_KEY` env var is set (plain SHA-256 can be guessed for short ids).
  `"mask"` gives `al***42`, or `***` for ids of 4 characters or fewer. An invalid value raises
  `ValueError`. Event payloads of the Redis, RabbitMQ, Kafka and Console publishers are unchanged.
- Opt-in TLS and SASL for `KafkaPublisher`: `security_protocol`, `ssl_context` (or
  `ssl_cafile` / `ssl_certfile` / `ssl_keyfile`), `sasl_mechanism`, `sasl_plain_username`,
  `sasl_plain_password`. Each is passed to `AIOKafkaProducer` only when set; plaintext stays the
  default. Redis and RabbitMQ already encrypt with `rediss://` and `amqps://` URLs (now in their
  docstrings).

- `collection` on `UserMemoryConfig` and `AgentMemoryConfig`: a name or a function
  `(run_config) -> name`, resolved for every memory tool call and preload search. A function
  must read the authenticated identity (`cfg["user"]`, `cfg["user_id"]`), never state, for
  example `UserMemoryConfig(collection=lambda cfg: f"company_{cfg['user']['tenant_id']}")`.
  An empty name, or a function that returns anything but a non-empty string, raises `ValueError`.
- `QdrantStore(collection=...)` (and the `create_*_qdrant_store` factories) also accept a function
  `(config) -> name`, used when the config has no `collection`. With a function default,
  `asetup()` creates no collection (there is no fixed name); collections are created on first use.
- `SYSTEM_USER_PREFIX` (`"__system__"`), `system_user_id(name)` (returns `"__system__:<name>"`)
  and `is_system_user_id(user_id)`, exported from `tenxgraph.storage` and
  `tenxgraph.storage.store`: the reserved owners of shared, ownerless data (agent/app memory,
  RAG knowledge bases). Any id starting with `__system__` is reserved; the API refuses it as a
  real user's id.
- `RAGAgent(store_config_builder=...)`: an optional `(run_config) -> dict` used to build the
  `store.asearch` config for each request, so the knowledge-base search can be scoped per user
  or tenant (`lambda cfg: {"user_id": cfg["user_id"]}` with `QdrantStore`). Without it the fixed
  `store_config` is used for everyone, as before.
- `BaseCheckpointer.aupdate_message_metadata(config, message_id, metadata)` (and the sync
  `update_message_metadata`) merges keys into a stored message's metadata, replacing keys that
  already exist, and returns the updated message or `None` if the message is not in that thread.
  It applies the same owner isolation as `aget_message`. Implemented for Postgres, SQLite and
  in-memory; a custom checkpointer that does not override it raises `NotImplementedError`.
- **`MediaFetchPolicy`, `MediaURLError` and `fetch_media_url`**, exported from
  `tenxgraph.storage.media`. `MediaFetchPolicy` (frozen dataclass: `enabled`, `allowed_schemes`,
  `allowed_hosts` with `*.example.com` wildcards, `allow_private_for_allowed_hosts`,
  `allowed_content_types`, `max_bytes`, `timeout`, `allowed_gcs_buckets`) has `check_url(url)`,
  which needs no network and raises `MediaURLError` for a disabled policy, a scheme or host that
  is not allowed, an IP literal that is not public, or a `gs://` bucket that is not listed.
  `await fetch_media_url(url, policy=None)` runs that check, downloads safely and returns
  `(bytes, mime)`.
- `StateGraph.compile(media_policy=...)` (and the same argument on the `compile()` of the
  prebuilt agents) sets the policy; it is bound in the dependency container next to the media
  store. `CompiledGraph.media_policy` is a read-only property that returns it, or the default
  `MediaFetchPolicy()` when none was set, so an API can call `compiled.media_policy.check_url(url)`
  on client input.
- `BaseMediaStore.update_metadata(storage_key, updates)` shallow-merges `updates` into the
  stored metadata (a value of `None` removes the key) and returns `False` when the object does
  not exist. Implemented for `LocalFileMediaStore` (atomic sidecar rewrite, mode `0o600`),
  `InMemoryMediaStore` and `CloudMediaStore`; a custom store that does not override it raises
  `NotImplementedError`.
- `tenxgraph.utils.safe_http`: the checked HTTP client behind `fetch_url` and `fetch_media_url`
  (public-address check including NAT64 and IPv4-compatible forms, address pinning, manual
  redirect loop with a check on every hop, body cap, optional deadline, no proxy variables).
- New scope constant `CHECKPOINT_IMPORT` (`"checkpointer:import"`), included in `ALL_SCOPES`.
- `InterruptMismatchError` (`INTERRUPT_001`) and `ResumeValidationError` (`INTERRUPT_002`) in
  `tenxgraph.core.exceptions`, and the optional `resume_interrupt_id` resume input key
  (`RESUME_INTERRUPT_ID_KEY` in `tenxgraph.utils.interrupt`). `response_schema` is a JSON Schema
  dict; the resume check supports `type`, `enum`, `properties`, `required` and `items` and
  ignores other keywords (`validate_against_schema`).
- `BaseCheckpointer.abegin_run`, `aend_run` and `aget_active_run`: the checkpointer tracks the
  active run id per thread (generic cache namespace `active_run`, same TTL as the stop flag), so
  it works on every backend with no schema change. `arequest_stop` stores the targeted run id and
  `ais_stop_requested` only reports a request that targets the config's `run_id` (a config with
  no `run_id`, or an old untargeted flag, still matches). Existing method signatures are
  unchanged.

### Changed

- **Behaviour change:** store calls with an empty `user_id` used to work and now raise.
  `RAGAgent` searches as `system_user_id("knowledge_base")` unless `store_config` sets
  `user_id` (index the knowledge base with
  `config={"user_id": system_user_id("knowledge_base")}`); `store_config_builder` output is
  used as is and must include `user_id`. Records written with an empty owner before this
  change are not found by any user-scoped read; re-index them under the intended owner.
- Postgres schema version 4 changes the messages primary key. The migration runs automatically
  the first time the checkpointer sets up its schema, under the existing advisory lock.
  `aget_message` and `adelete_message` on `PgCheckpointer` now require `thread_id` in the config
  (they used to match on `message_id` alone when it was missing).
- The Google message conversion (`_convert_to_google_format`) no longer downloads anything
  itself. The async callers fetch external media first (`_prefetch_google_media`) and pass it
  in; an external URL that was not fetched becomes a `[Failed to load ...]` text part.
- `fetch_url` now runs on `tenxgraph.utils.safe_http`; its behaviour and messages are unchanged.
- **Examples rewritten.** `examples/` is reorganised by feature, uses `tenxgraph` imports and
  10xGraph naming throughout, and builds on the prebuilt agents instead of repeating the
  hand-wired ReAct loop. Every folder has a README with docs-site frontmatter, and
  `examples/README.md` is the index. Model strings are written in each file and rotate across
  OpenAI, Google Gemini, Anthropic and DeepSeek (OpenAI-compatible `base_url`); multi-agent
  examples use a different company per agent. Folders:
  - Getting started: `quickstart`, `graph_basics`, `streaming`.
  - Tools and integrations: `tools`, `mcp` (local server, GitHub, Xquik), `providers`,
    `dependency_injection`.
  - Production: `graceful_shutdown`, `resilience`, `human_in_the_loop`, `persistence`,
    `context_management`, `callbacks`, `observability`.
  - Memory and knowledge: `long_term_memory`, `rag`.
  - Agent patterns: `structured_output`, `multi_agent`, `skills`.
  - Multimodal and realtime: `multimodal`, `realtime_audio`.
  - Quality: `testing`, `evaluation`.
  - Deployment: `serving`.
  - Use cases (SDK only): `use_cases/customer_support`, `use_cases/research_assistant`.
  - Full-stack apps: `apps/rag_assistant` (ReAct + RAG, API server, web UI, Docker Compose) and
    `apps/copilotkit_agui` (CopilotKit frontend over AG-UI).

  The old `a2a_sdk` example is removed (A2A is disabled in this release), and the old per-topic
  folders (`react`, `react_stream`, `react-mcp`, `github-mcp`, `xquik-mcp`, `swarm`,
  `supervisor_team`, `handoff`, `plan_act_reflect`, `memory`, `checkpointer`, ...) are merged
  into the folders above.

### Deprecated

- Setting the file tool root through the run config (`file_tool_root` / `workspace_root`)
  now emits a `DeprecationWarning` and will be removed in 2.0. Use `create_file_tools(root=...)`.

### Removed

- **The graph-level memory API.** Memory is now configured only on the Agent
  (`Agent(memory=MemoryConfig(...))`, `UserMemoryConfig`, `AgentMemoryConfig`,
  `user_memory_tool`, `agent_memory_tool`). Removed with no deprecation period:
  `MemoryIntegration` (`wire`, `preload_node`, `tools`, `system_prompt`),
  `create_memory_preload_node`, `get_memory_system_prompt` and the `memory_tool` function, from
  `tenxgraph.storage`, `tenxgraph.storage.store`, `tenxgraph.prebuilt` and
  `tenxgraph.prebuilt.tools`. Migration: pass `memory=MemoryConfig(retrieval_mode="preload" |
  "postload", user_memory=UserMemoryConfig(...))` to `Agent` instead of wiring a preload node
  and a `ToolNode([memory_tool])`.
- `tenxgraph.storage.media.temp_cache.fetch_and_cache` and its private helpers. Nothing called
  it, and it downloaded any URL with aiohttp and no checks. `TemporaryMediaCache` and
  `TempCacheEntry` stay.

### Fixed

- **`QdrantStore` search and list work on current qdrant-client (B15).** `asearch` and `aget_all`
  called `AsyncQdrantClient.search()`, which qdrant-client 1.18 no longer has, so both raised
  `AttributeError` (and `aget_all` passed an empty vector). `asearch` now uses `query_points` and
  `aget_all` uses `scroll` with the same user filter. The `qdrant` extra now requires
  `qdrant-client>=1.10.0`, and a Qdrant server must be 1.10 or newer.
- **`aforget_memory` no longer widens a narrow forget.** `QdrantStore` ignored `memory_type`,
  `category` and `filters` and deleted every memory of the user; it now applies them (AND,
  with the user and thread), and raises `ValueError` for a filter value it cannot match
  exactly instead of skipping it. `Mem0Store` can only delete by user, so it raises
  `ValueError` when any of the three is passed. A forget without them still deletes all of
  the user's memories.
- **`aput_state_cache` follows the owner rule in `SqliteCheckpointer` and
  `InMemoryCheckpointer`.** With owner isolation on, a cache write from another user was
  accepted, and a write to a thread nobody owned yet planted state the first real user then
  read. Now a non-owner's write is dropped and the first writer of a thread becomes its
  owner, as `aput_state` already did. Postgres was not affected (its cache key includes the
  user).
- **A tool call with cut-off, empty or non-object arguments no longer fails the node.** The
  OpenAI converter called `json.loads` on the model's arguments with no guard, so a call truncated
  at `max_tokens` (`{"a": `) raised and killed the run. All converters (OpenAI chat, OpenAI
  Responses, Anthropic, Google) now share `parse_tool_arguments`
  (`tenxgraph.runtime.adapters.llm.tool_arguments`): empty means no arguments, a JSON object is
  used, anything else is invalid. An invalid call keeps its raw argument string in `tools_calls`,
  and the ToolNode (invoke and stream) answers it with the tool error "Tool call arguments were
  not valid JSON (they may have been cut off). Call the tool again with a complete JSON object."
  instead of running the tool with empty arguments, so the model can retry. Previously the Anthropic
  converter replaced bad streamed input with `{}` and ran the tool, and the stream handler did not
  catch bad arguments at all. History sent back to Anthropic and Google uses `{}` for such a call.
- **Multi-agent loop limits are per turn, not per thread.** `SupervisorTeamAgent(max_rounds=...)`,
  `PlanActReflectAgent(max_iterations=...)` and `StructuredOutputAgent(max_attempts=...)` kept
  their counters in the checkpointed `execution_meta.internal_data` and never reset them, so
  with `max_rounds=2` turn 3 of a thread ended at once. The counters now clear when a new turn
  starts on a finished thread or with a new message after a run that ended in an error (a
  failed turn does not use up the next one's budget); resuming a paused or stopped thread
  (interrupt, client tool, stop, `resume`) keeps them. Prebuilt agents register such keys with `register_per_turn_key`. Behaviour
  change: an app that relied on a lifetime budget now gets the full limit on every turn.
- **The `graceful_shutdown` example no longer runs `eval()` on model input.** Its `calculate`
  tool passed the model's expression to `eval()` with empty builtins and called it safe; an
  expression can climb from `().__class__` to interpreter internals, and `9**9**9` hangs the
  process. It now uses the prebuilt `safe_calculator`, as do the docs pages that copied it.
- **`SwarmAgent` handoffs now fire with real model output.** The route read the tool name from
  the top level of each tool call, but every provider stores it at `function.name`, so no
  handoff was ever seen. The name is now read from `function.name` (the flat `name` key is still
  accepted). `can_handoff_to` is enforced: a call to a target that is not allowed does not jump
  and gets a tool-not-found result.
- `import tenxgraph.core.exceptions.media_exceptions` as the first import no longer fails with a
  circular import (`MediaTransportMode` is now a type-only import).
- `ainvoke` and `astream` release the run's stop scope when the run ends (`aend_run`), so a stop
  request that targeted a finished run is dropped instead of lingering until its TTL.
- `ainvoke` now follows a `Command` returned by a ToolNode (update and goto), as `astream`
  already did. It used to drop it, so handoffs worked in stream but not in invoke.
- `ToolNode.mcp_tools` no longer grows on every tool listing; it is rebuilt from the server's
  list each time.
- The validator's base64 check never raised: the `ValidationError` was swallowed by the
  `suppress(Exception)` around the decode. It now raises in strict mode, matches the `DAN`
  keyword case-insensitively, and decodes URL-safe and unpadded base64.
- The validator also scans `DocumentBlock.excerpt` and the arguments of assistant
  `tools_calls`, which are sent to the model but were not checked.
- Input is NFKC-normalized and stripped of zero-width characters before matching, so
  fullwidth letters and zero-width spaces no longer bypass the patterns.
- The injection patterns require their keywords to start a word, so "ignore the overall
  instructions" no longer matches through "over*all*".
- **The validator no longer rejects text in other languages.** It blocked any message that
  was 30% or more non-ASCII, which covered almost any message in Bengali, Hindi, Arabic,
  Chinese, Japanese, Russian and others, and any Chinese, Japanese or Korean clause of 30 or
  more characters. It now flags only characters used to hide or disguise text (Unicode tag
  characters, emoji variation-selector smuggling, bidi overrides, and Latin and Cyrillic
  letters mixed inside one word), and the long-word check applies to ASCII words only.
- Docstrings no longer claim that `strict_mode=False` sanitizes input. It logs a warning and
  passes the input through unchanged.

## [0.10.3] - 2026-10-22

Python 3.14 and 3.15 support. No API changes.

### Added

- **Python 3.14 and 3.15 are officially supported.** The full test suite runs on 3.12, 3.13,
  3.14 and 3.15 in CI, and the package classifiers list all four.

### Changed

- **`injectq` 0.5 is now allowed** (`injectq>=0.4.0,<0.6`). 0.5.0 fixes a crash under
  Python 3.15 global lazy imports (`PYTHON_LAZY_IMPORTS=all`); 0.4.x keeps working.
- Replaced `asyncio.iscoroutinefunction` and the event-loop policy API, both removed in
  Python 3.16. The sync `CompiledGraph.stream()` no longer changes the thread's current
  event loop. No behaviour change otherwise.

## [0.10.2] - 2026-10-22

Smaller installs: Pillow is now optional. No API changes.

### Changed

- **Pillow is no longer a required dependency.** It moved to the `images` extra, which cuts
  about 20 MB from every install and Docker image. Image resizing, EXIF orientation and format
  conversion in the media processor still work when Pillow is present; without it, images pass
  through unchanged. To keep the old behaviour, install `10xgraph[images]` (also included in
  `10xgraph[all]`).

## [0.10.1] - 2026-10-16

Bug fixes for the checkpointers' state cache. No API changes.

### Fixed

- **`aput_state` now refreshes the state cache (write-through) on every checkpointer.** A run
  loads its state from the cache first (`aget_state_cache`), but `aput_state` wrote only the
  durable store, so a state written between runs was not what the next run started from.
  `InMemoryCheckpointer` and `SqliteCheckpointer` silently lost the update (the next run wrote
  the old state back over it). `PgCheckpointer` also left the old checkpoint version in Redis,
  so the next run's first write raised `StaleStateError` (`STORAGE_CONFLICT_001`) and the run
  failed. `PgCheckpointer` refreshes Redis after the durable write succeeds, at the new
  version, version-guarded and best-effort like every cache write (a Redis outage never fails
  the write); `SqliteCheckpointer` updates its cache row in the same transaction.
- **Clearing a thread's state now also drops its cached state.** `aclear_state` on
  `InMemoryCheckpointer` and `SqliteCheckpointer` deleted only the durable state, so the next
  run read the cleared state back from the cache and resumed from it. Both now delete the
  cache entry too (SQLite in the same transaction), as `PgCheckpointer` already did.
  `InMemoryCheckpointer.aclean_thread` deleted only the thread record; it now also deletes
  the thread's state, cached state and messages, like the SQLite and Postgres checkpointers.

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

## [0.9.0] - 2026-07-21

This release was prepared as `1.0.0` but published to PyPI as `0.9.0`. It is the
production-hardening round on top of `0.8.0`, and the package is classified
`Development Status :: 5 - Production/Stable`: durable state gets
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
