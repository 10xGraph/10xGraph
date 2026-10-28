# 10xgraph 1.0.0rc1

The 1.0 release candidate: the public API is now stable under a deprecation policy, with a security hardening round across tools, memory, media, checkpointers and handoffs.

**This is a release candidate.** Pip skips it unless you pass `--pre`. This release has **breaking changes**. Read "Before you upgrade" first.

```bash
pip install -U --pre "10xgraph==1.0.0rc1"
```

`10xgraph-api` and `10xgraph-client` ship 1.0 release candidates alongside. From 1.0 the three packages are versioned together: major.minor is identical across them (`10xgraph` 1.1.x works with `10xgraph-api` 1.1.x and `10xgraph-client` 1.1.x), and patch numbers move independently. A public API is removed only in a major release, after at least one minor of deprecation.

## Before you upgrade

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

## Security

- Handoffs, hidden tools, MCP name clashes and parallel tool calls are now enforced or bounded (see above).
- `fetch_url` and client-supplied media URLs are fetched through a checked HTTP client: every redirect is verified, the connection is pinned to the checked address, and internal and cloud-metadata addresses are refused.
- `PromptInjectionValidator` and the `<think>` / `<reasoning>` / `<thought>` parsers no longer backtrack on crafted input.
- Tool failures, metrics labels and `RedisPublisher` connection errors no longer leak raw exception text, model-invented names or the Redis password.
- Memory and skill text can no longer be used to build oversized or state-leaking prompts, and `PRELOAD` and RAG context stay out of saved history.
- Checkpointer reads, writes and paging are scoped to the owner and bounded, and local files are owner-only.
- New `create_file_tools(root)` returns `file_read`, `file_write` and `file_search` locked to a directory set in code.

## New

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
