# 10xgraph 1.0.0

The first stable release of `10xgraph`, the core library of 10xGraph: open-source, self-hosted
infrastructure for AI applications. From 1.0 the public API is stable. There are no breaking
changes before 2.0, and a public API is removed only after at least one minor release of
deprecation.

```bash
pip install -U 10xgraph 10xgraph-api
npm install 10xgraph-client
```

Released together with `10xgraph-api` 1.0.0 and `10xgraph-client` 1.0.0. Python 3.12 to 3.15.

## Versioning from 1.0

- `major.minor` is identical across `10xgraph`, `10xgraph-api` and `10xgraph-client`.
  `10xgraph` 1.1.x works with `10xgraph-api` 1.1.x and `10xgraph-client` 1.1.x.
- Patch numbers move independently. A patch contains only bug or security fixes.
- A new feature means all three packages release the next `X.Y.0` together.
- The deprecated `agentflow` import keeps working until 2.0, with a `DeprecationWarning`.

## What 1.0 brings since 0.10

- **Tenant isolation in storage.** `PgCheckpointer` and `PgVectorStore` take `tenant_columns`
  (and data `extra_columns`) whose values come only from the verified identity and filter every
  query. Postgres row-level security is opt-in (`row_level_security=True`).
- **Identity with reach.** The caller is an `Identity` in `config["user"]` with reach `own`,
  `tenant` or `all`. Reads widen with reach, writes stay with the owner, and a thread outside the
  caller's reach raises `ThreadNotFoundError`.
- **Long-term memory in the same Postgres.** `PgVectorStore` (extra `pgvector`) with similarity,
  full-text and hybrid search and safe filters. The model can remember, replace and forget a
  memory by key.
- **Calling services as the user.** `UserCredential` carries the caller's credential to tools,
  and `ToolNode(mcp_auth=...)` sets per-call MCP headers. The credential never reaches traces,
  publishers or callbacks.
- **Graceful shutdown and crash recovery.** `await graph.aclose()` ends active runs in phases. A
  cancelled, crashed or shut-down run leaves a usable thread, and a thread left running by a crash
  is repaired on its next load.
- **Tracing.** One trace per run with OTEL GenAI span names (`invoke_agent`, `chat {model}`,
  `execute_tool {tool}`), and publishers for OpenTelemetry, Phoenix, Logfire and LangSmith.
- **Safer by default.** Client media URLs and `fetch_url` go through a checked HTTP client
  (redirects verified, internal and cloud-metadata addresses refused). `PromptInjectionValidator`
  no longer backtracks on crafted input. Tool errors no longer leak raw exception text to the
  model. File tools refuse secret files.
- **Text that breaks Postgres is handled.** A NUL character or lone surrogate in client input is
  refused with a `ValidationError`. In model or tool output it is cleaned, so a thread can no
  longer get stuck on a failed checkpoint write.
- **Providers.** OpenAI uses the Responses API by default. `base_url` works for Gemini (proxies,
  LiteLLM, self-hosted endpoints). Claude reasoning comes back readable by default. Multi-agent
  graphs on Claude no longer fail on signed thinking blocks.

## Before you upgrade

**From 0.10.x.** 1.0 has breaking changes in memory, stores, tools, handoffs, resume, media and
the Postgres schema. The full list, with the migration for each, is in the upgrade guide:
https://10xgraph.com/docs/project/upgrade-to-1.0

The ones most projects hit:

- `Mem0Store` and the `mem0` extra are removed. Move to `PgVectorStore` or `QdrantStore`.
- Every store operation needs a non-empty `user_id`. Shared data uses `system_user_id(name)`.
- `@tool` takes no `name`. A tool's name is its function name.
- The `mcp` extra needs fastmcp 4.1.7 and mcp 2.3 or later.
- Default reasoning is each provider's lowest effort (OpenAI `"minimal"`, Anthropic and Google
  `"low"`). Pass `reasoning_config={"effort": "medium"}` to keep the 0.10 depth.
- Tracing sends message content and state by default (`ObservabilityLevel.FULL`), and span names
  changed. Set `level=ObservabilityLevel.STANDARD` to keep content out of traces, and update
  dashboards that match the old `tenxgraph.*` span names.
- `PgCheckpointer` migrates its schema on first setup. Stop all old pods before the migration;
  it is not reversible.

**From a release candidate.** Read the `CHANGELOG.md` sections after your rc. Since 1.0.0rc7:

- Claude reasoning is readable by default: `reasoning_config=True` (or unset) sends
  `thinking.display: "summarized"`. An explicit `reasoning_config` dict sends `display` only when
  it has a `display` key.
- `Agent(provider="google", base_url=...)` sets the Gemini client's base URL.
- A tool's returned dict reaches the model unchanged (`status`, `error`, `success` and
  `is_error` are no longer removed).
- `ReactAgent`, `StructuredOutputAgent` and `PlanActReflectAgent` send reasoning parameters on
  Chat Completions only when `reasoning_config` is set.
- `InMemoryCheckpointer` and `SqliteCheckpointer` report an existing thread as not new.
- NUL characters and lone surrogates are refused in input and cleaned in output.
- `CloudMediaStore` fixes for a missing object and an owner's last deleted file.

See [CHANGELOG.md](CHANGELOG.md) for every change.
