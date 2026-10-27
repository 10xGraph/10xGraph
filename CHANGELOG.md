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

### Security

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

### Deprecated

- Setting the file tool root through the run config (`file_tool_root` / `workspace_root`)
  now emits a `DeprecationWarning` and will be removed in 0.11. Use `create_file_tools(root=...)`.

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
