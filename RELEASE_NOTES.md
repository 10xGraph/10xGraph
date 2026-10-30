# 10xgraph 1.0.0rc3

The third 1.0 release candidate: the `1.0.0rc2` code plus a required connectivity check on memory stores, so `10xgraph-api` can verify the store at startup and in `/health/ready`.

```bash
pip install -U --pre "10xgraph==1.0.0rc3"
```

Use it with `10xgraph-api` 1.0.0rc3. `10xgraph-client` stays at 1.0.0-rc.2 (no client changes).

## Before you upgrade (changes in rc3)

- **`BaseStore.aping()` is now abstract.** A custom store must implement `async def aping(self) -> None`: return when its backend is reachable and raise when it is not. Keep it read-only and cheap (no writes, no embedding, no collection creation); an in-process store just returns. A store without it raises `TypeError` when created.

## What changed in rc3

- `QdrantStore.aping()` lists collections (read-only, needs no collection to exist).
- `Mem0Store.aping()` lists the collections of its vector store.
- `InMemoryStore.aping()` returns at once.

`10xgraph-api` 1.0.0rc3 calls `aping()` in its startup check and in `/health/ready`, so a stopped Qdrant is now reported instead of skipped.

See [CHANGELOG.md](CHANGELOG.md) for the full list. The rc2 notes follow.

---

## 10xgraph 1.0.0rc2

The second 1.0 release candidate: it is the `1.0.0rc1` code plus a round of fixes for multi-agent handoffs, streaming, realtime sessions and evals, graceful shutdown with crash recovery, and new defaults for OpenAI and reasoning.

**This is a release candidate.** Pip skips it unless you pass `--pre`. This release has **breaking changes**. Read "Before you upgrade" first.

```bash
pip install -U --pre "10xgraph==1.0.0rc2"
```

`10xgraph-api` and `10xgraph-client` ship 1.0 release candidates alongside (`10xgraph-api` 1.0.0rc2, `10xgraph-client` 1.0.0-rc.2). From 1.0 the three packages are versioned together: major.minor is identical across them (`10xgraph` 1.1.x works with `10xgraph-api` 1.1.x and `10xgraph-client` 1.1.x), and patch numbers move independently. A public API is removed only in a major release, after at least one minor of deprecation.

## Before you upgrade (changes in rc2)

If you are on `1.0.0rc1`, this section is all that is new. If you are coming from 0.10.x, read it and then "Upgrading from 0.10.x" below.

**Reasoning and OpenAI defaults**
- Default reasoning is now each provider's lowest effort. An unset `reasoning_config` or `True` (the `ReactAgent` default) used to mean `{"effort": "medium"}`; it now means OpenAI `"minimal"`, Anthropic `"low"`, Google `"low"` (`thinking_budget=512`), resolved per request. Pass `{"effort": "medium"}` to keep the old depth. `None` or `False` still turns reasoning off.
- `Agent` uses the OpenAI Responses API by default (`api_style="responses"`). With a custom `base_url` and no `api_style`, it stays on Chat Completions. Pass `api_style="chat"` to keep the old behaviour. The `openai` extra is now `openai>=1.93.1,<4`; on openai 3.x an `http_client` in `llm_kwargs` must be an `httpx2.AsyncClient`.
- On the Responses API, reasoning text now comes back by default (`reasoning.summary: "auto"`). If a model such as gpt-4o rejects the default reasoning parameter, the call is retried once without it.
- A custom `base_url` falls back to Chat Completions only on 404, 405 or 501. Other errors, such as a 429 or timeout, are raised instead of switching for good.

**Timeouts, tools and events**
- Tool and node timeouts are off by default (they were 300 s and 900 s). Set `tool_timeout` or `node_timeout` in the run config for a limit; it now applies to both `invoke` and `stream`.
- Tool arguments are checked against the declared types. Values that convert cleanly are converted; values that cannot convert are refused with a failed tool result.
- Published events are snapshots, copied before they are queued. Consumers no longer see later edits.
- `pass_user_info_to_mcp=True` sends user info in the MCP call's request `meta`, not as a `user` argument.

**Runs and APIs**
- Sync `invoke()`, `stream()`, `stop()` and sync checkpointer methods raise `RuntimeError` inside a running event loop. Use the async method instead.
- `LiveAgent.arun()` raises `RuntimeError` if the instance already runs a session. `arealtime()` now gives each connection its own copy; call `new_session()` if you run `arun()` yourself.
- A stop request is saved as `StopRequestStatus.STOPPED`. Code that checks for `STOP_REQUESTED` after a stop must accept `STOPPED` too.
- `Message.reasoning` is removed. Read the `ReasoningBlock`s in `Message.content`.
- `mem0ai` is bounded to `>=2.0,<3`.

**Postgres rollout.** `PgCheckpointer` schema migrations (v4 from 0.10.x, v5 drops `messages.reasoning`) cannot run alongside old pods. Do not roll pod by pod: stop all old pods, run the migration (one new pod, or `await checkpointer.asetup()`), then start the rest. The migration is not reversible; a rollback to 0.10.x needs a database restore.

**Dependency minimums** (extras): `Pillow>=13.0.0` (`images`), `mcp>=1.30.0` (`mcp`), `opentelemetry-api` and `opentelemetry-sdk` `>=1.45.1` (`otel`, `langsmith`), `mem0ai>=2.0,<3` (`mem0`).

## What changed in rc2

**Fixes you may have hit**
- Handoffs: the target agent now gets a tool result for the handoff call and is told it is in charge, other tool calls in the same message are no longer dropped, and `SwarmAgent` no longer leaves an unanswered tool call.
- Streaming: clients now receive the real deltas instead of the final message repeated for each chunk.
- Realtime (`AudioAgent`) sessions save their transcripts and state, and a tool call re-sent on the same thread runs once.
- `SupervisorTeamAgent` no longer re-dispatches workers that already answered. `PlanActReflectAgent` shows the planner's reply to the user and no longer loops on empty replies. `StructuredOutputAgent` with tools works on Google.
- Empty model replies are retried once, then fail the node.
- Evals: a failing judge model is reported as an error (not a 0.5 score), trajectory tool arguments are the ones the model sent, and case results carry `expected`.
- Gemini memory search ranks better (document and query task types for `GoogleEmbedding`).
- A Redis outage no longer slows every step of a `PgCheckpointer` run (`redis_breaker_cooldown`).

**Shutdown and recovery**
- `await graph.aclose(...)` ends active runs in phases (grace, stop, cancel and drain); `install_shutdown_signals(graph)` is an opt-in SIGTERM/SIGINT helper. New runs after close raise `GraphClosedError`.
- A cancelled, crashed or shut-down run leaves a usable thread: unanswered tool calls are closed and the thread is marked INTERRUPTED. A thread left RUNNING by a crash is repaired on the next load.
- New errors: `NothingToResumeError`, `InterruptPendingError`, `StoredNodeMissingError` (with `compile(node_aliases=...)` for renamed nodes) and `StateFormatVersionError`.

**New options (all opt-in unless noted)**
- Tool limits and MCP controls on `ToolNode`: `max_result_chars`, `mcp_include`, `mcp_exclude`, `mcp_prefix`, `mcp_connect_timeout`, and `mcp_cache_tools` (on by default).
- Tool factories `create_fetch_url`, `create_google_web_search` and `create_vertex_ai_search` that fix hosts, model or datastore in code.
- LLM resilience: `RetryConfig.jitter` and concurrency caps (including a cross-process cap through Redis), `stream_idle_timeout`, `TENXGRAPH_LLM_TIMEOUT`, and `stop_issue` message metadata.
- `PgCheckpointer`: `redis_key_prefix` and safe default timeouts. Media: image pixel limit and size caps.
- The HTML eval report is redesigned as an offline scorecard, and the user simulator emits progress events.
- Security: exception context and node error events no longer expose secrets or foreign exception text, and log lines neutralise CR and LF.

See `CHANGELOG.md` for the full list.

## Upgrading from 0.10.x (changes since rc1 also apply)

Everything below was the `1.0.0rc1` guidance and still applies when you jump from 0.10.x to rc2.

### Breaking changes from 0.10.x

**Removed APIs**
- The graph-level memory API (`MemoryIntegration`, `create_memory_preload_node`, `get_memory_system_prompt`, `memory_tool`) is gone. Pass `Agent(memory=MemoryConfig(...))`.
- `fetch_and_cache` is gone from `tenxgraph.storage.media.temp_cache`. Download through `fetch_media_url` with a `MediaFetchPolicy`.
- The old example folders and the `a2a_sdk` example are removed. Use the reorganised `examples/` folders.

**Stores and memory**
- Store operations with a missing or empty `user_id` now raise `ValueError` (`QdrantStore`, `Mem0Store`).
  - Pass a concrete `user_id`, or `system_user_id(name)` for shared data. Re-index records written with an empty owner.
  - `RAGAgent` searches as `system_user_id("knowledge_base")` unless `store_config` sets `user_id`. `store_config_builder` output must include `user_id`.
- Agent memory is owned by `system_user_id(agent_id)` (or `app_id`), not the current user. `Agent(memory=...)` raises `ValueError` if agent memory is on with neither id. Re-store memories written under a user by the old behaviour.
- A `collection` in the run config no longer overrides the memory scope's collection. Set `collection` on `UserMemoryConfig` / `AgentMemoryConfig`.
- `PRELOAD` memories are no longer a system message. They arrive as a fenced block in the latest user message of each request and are never saved to `state.context`. `Agent._build_memory_prompts` is replaced by `Agent._build_memory_block`; update prompts or tests that look for `[Long-term Memory Context]`.
- `RAGAgent` puts retrieved documents in a request-only block and no longer rewrites `state.context`. `execution_meta.internal_data["rag_docs"]` and `_RAG_DOCS_KEY` are gone. A custom agent passed to `RAGAgent` must accept the keyword-only `request_blocks` argument on `execute`, or `RAGAgent` raises `TypeError`.
- Skill and memory text is no longer run through `str.format`. Only your own `system_prompt` is interpolated, and a bad placeholder is logged instead of failing the run. Remove any reliance on `{placeholder}` expansion inside skill or memory text.
- `QdrantStore.aforget_memory` applies `memory_type`, `category` and `filters`. `Mem0Store.aforget_memory` raises `ValueError` if any of them is passed. The `qdrant` extra needs `qdrant-client>=1.10.0` and a Qdrant server 1.10 or newer.

**Tools and handoffs**
- Handoffs need a registered handoff tool. A `transfer_to_<X>` call only jumps if a tool made with `create_handoff_tool` is on that ToolNode. A `Command.goto` to an unknown node raises `GraphError` (`GRAPH_007`). `SwarmAgent` now enforces `can_handoff_to`.
- Tools hidden by `tools_tags` can no longer be called. Add the tag or the tool if the call was intended.
- An MCP tool with the same name as a local or remote tool raises `ValueError`. Rename one of them.
- Parallel tool calls are capped at 10 per turn. Set `max_parallel_tool_calls` in the run config, or `TENXGRAPH_MAX_PARALLEL_TOOL_CALLS`, to change it.
- Unexpected tool errors no longer reach the model verbatim. The model sees `Tool '<name>' failed with an internal error (reference <ref>)` and the full error is logged. To show a message to the model, return it as a string; a raised `ValueError` is no longer shown.
- A tool call with invalid JSON arguments is answered with a tool error instead of running with empty arguments.
- Tool metrics label unknown tools as `"unknown"` instead of the model-invented name.
- File tools refuse secret files (`.env`, keys, `.ssh`, `.aws`, `.git` and similar) for any root, and `file_search` stays inside the root. Change the list with `create_file_tools(root, deny=..., use_default_deny=False)`. Prefer `create_file_tools(root=...)` to the run-config root, which is deprecated and will be removed in 2.0.
- `fetch_url` no longer follows redirects to internal addresses or uses `HTTP_PROXY` / `HTTPS_PROXY`. Fetches that relied on a proxy or an internal redirect now fail.

**Runs and resume**
- Multi-agent loop limits (`max_rounds`, `max_iterations`, `max_attempts`) are per turn. They reset on each new turn instead of being a lifetime budget per thread.
- Resume is checked. A `resume_interrupt_id` that is not the pending interrupt's raises `InterruptMismatchError`, and a value that fails the interrupt's `response_schema` raises `ResumeValidationError`.
- `PromptInjectionValidator` behaves differently. Non-strict mode scans only the first `max_length` characters, strict mode raises on bad base64, and text in other languages is no longer rejected.

**Media and storage**
- Client-supplied media URLs are fetched under a `MediaFetchPolicy`: only `http` and `https` to public hosts, 20 MB, 10 s, and `image/*` and `application/pdf` by default.
  - Audio and video links on Google graphs stop downloading until you allow those content types. Pass `compile(media_policy=MediaFetchPolicy(...))`.
  - The private `_convert_to_google_format` no longer downloads.
- `CloudMediaStore` raises `ValueError` for storage keys that are not 32 lowercase hex characters. Objects stored under other keys by custom code are no longer reachable through the store.
- Local media and the SQLite checkpointer create owner-only files (`0o700` folders, `0o600` files). Other OS users or services that read them lose access; widen the mode yourself if that is intended.

**Checkpointers**
- `alist_messages` and `alist_threads` raise `ValueError` for a negative `limit` or `offset`, and an explicit `limit` is lowered to `MAX_LIST_LIMIT` (1000).
- `PgCheckpointer` schema version 4 changes the messages primary key to `(thread_id, message_id)`. `aget_message` / `adelete_message` require `thread_id` in the config. The migration runs automatically on first setup. `aget_tool_result` is scoped to the thread owner.
- Checkpointers apply the owner rule to state-cache writes, and the in-memory checkpointer returns copies of state, so mutating a loaded state no longer changes the store until you save it.

### Security

- Handoffs, hidden tools, MCP name clashes and parallel tool calls are now enforced or bounded (see above).
- `fetch_url` and client-supplied media URLs are fetched through a checked HTTP client: every redirect is verified, the connection is pinned to the checked address, and internal and cloud-metadata addresses are refused.
- `PromptInjectionValidator` and the `<think>` / `<reasoning>` / `<thought>` parsers no longer backtrack on crafted input.
- Tool failures, metrics labels and `RedisPublisher` connection errors no longer leak raw exception text, model-invented names or the Redis password.
- Memory and skill text can no longer be used to build oversized or state-leaking prompts, and `PRELOAD` and RAG context stay out of saved history.
- Checkpointer reads, writes and paging are scoped to the owner and bounded, and local files are owner-only.
- New `create_file_tools(root)` returns `file_read`, `file_write` and `file_search` locked to a directory set in code.

### New

- `MediaFetchPolicy`, `MediaURLError` and `fetch_media_url` in `tenxgraph.storage.media`, plus `compile(media_policy=...)` and `CompiledGraph.media_policy`.
- `collection` on `UserMemoryConfig` and `AgentMemoryConfig`: a name, or a function of the run config.
- `system_user_id`, `is_system_user_id` and `SYSTEM_USER_PREFIX` for shared, ownerless data.
- `RAGAgent(store_config_builder=...)` to scope the knowledge-base search per user or tenant.
- `user_id_mode` (`"raw"`, `"hash"`, `"mask"`) on the OTEL, Logfire and LangSmith setup, to control how the user id appears in traces.
- Opt-in TLS and SASL for `KafkaPublisher`.
- `BaseCheckpointer.aupdate_message_metadata`, `BaseMediaStore.update_metadata`, and run tracking with `abegin_run`, `aend_run` and `aget_active_run`.
- `InterruptMismatchError`, `ResumeValidationError` and the `resume_interrupt_id` resume input.
- The `checkpointer:import` scope constant (`CHECKPOINT_IMPORT`).
- Examples rewritten and reorganised by feature.

See `CHANGELOG.md` for details.
