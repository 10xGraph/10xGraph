---
title: Long-Term Agent Memory with Qdrant in 10xGraph
description: Give a 10xGraph Python agent long-term memory across sessions with MemoryConfig, a local Qdrant vector store, user memory and shared agent memory.
section: Examples
group: "Memory and knowledge"
order: 150
label: Long-term memory
updated: "2026-10-28"
faq:
  - question: How does the agent know which user a memory belongs to?
    answer: Every store operation uses the user_id in the run config. Memories written for one user_id are never returned for another, and the store raises if user_id is empty.
  - question: What is the difference between user memory and agent memory?
    answer: User memory is private to a user_id and the model can both search and write it. Agent memory is shared by all users, owned by system_user_id(agent_id), and the model can only search it.
  - question: Do memory writes block the response?
    answer: No. Writes run as background tasks so streaming is not delayed. Call aclose() on the compiled graph to flush pending writes before the process exits.
---

## What you will build

A Claude chat agent that remembers facts between conversations. In the first session a user says they are vegetarian and live in Lisbon. In a second session, on a new thread with the same `user_id`, the agent recalls both facts. A different user sees none of it.

`qdrant_memory.py` uses a local Qdrant store, so no cloud account is needed.

## Key concepts

- `MemoryConfig`: passed to `ReactAgent(memory=...)` or `Agent(memory=...)`. Registers the memory tools and adds a short memory section to the system prompt.
- `ReadMode`: `POSTLOAD` (default) gives the model a `user_memory_tool` to search and remember. `PRELOAD` searches on every turn and attaches results to the user message, with no write tool. `NO_RETRIEVAL` turns retrieval off.
- `UserMemoryConfig`: private per-user memory, keyed by `user_id` in the run config.
- `AgentMemoryConfig`: shared memory that the model can only search. It is off by default and needs `agent_id` or `app_id`.
- `system_user_id(name)`: the reserved owner id (`__system__:<name>`) for shared data such as agent memory.
- `create_local_qdrant_store(path, embedding)`: file-based Qdrant, no server.
- `OpenAIEmbedding`: embeds memories with `text-embedding-3-small`.

## Prerequisites

```bash
pip install "10xgraph[anthropic,openai,qdrant]"
export ANTHROPIC_API_KEY=...   # chat model
export OPENAI_API_KEY=...      # embeddings
```

No Docker services are required. The script stores vectors in a temporary directory and delete it at the end.

## Run it

```bash
python examples/long_term_memory/qdrant_memory.py
```

## How it works

The store and memory configuration:

```python
store = create_local_qdrant_store(
    path=QDRANT_PATH,
    embedding=OpenAIEmbedding(model="text-embedding-3-small"),
    collection="assistant_memory",
)

memory = MemoryConfig(
    store=store,
    retrieval_mode=ReadMode.POSTLOAD,
    user_memory=UserMemoryConfig(),
    agent_memory=AgentMemoryConfig(enabled=True, agent_id="support-bot"),
)
```

The agent only needs `memory=memory`. The `ReactAgent` forwards it to its `Agent`, which registers `user_memory_tool` and `agent_memory_tool`:

```python
agent = ReactAgent(model="claude-sonnet-5-5", system_prompt=[...], memory=memory)
```

Memory is scoped by the run config, not by the thread. Session 2 uses a new `thread_id` but the same `user_id`, so the chat history is empty and the only source of the facts is the store:

```python
config = {"user_id": user_id, "thread_id": thread_id}
await app.ainvoke({"messages": [Message.text_message(text)]}, config)
```

Writes happen in the background. The script calls `await app.aclose()` between sessions so they are flushed before session 2 searches.

Agent memory is seeded by writing as `system_user_id("support-bot")`, which is the owner the agent memory tool searches.

## Expected output

Illustrative output. Wording varies by run.

```text
[alice/s1] user: I'm vegetarian and I live in Lisbon. Please remember that.
assistant: Saved. I'll keep in mind that you're vegetarian and based in Lisbon.

[alice/s2] user: Suggest a dinner idea for tonight, close to home.
assistant: Since you're vegetarian and in Lisbon, try a bean and chestnut stew...

[alice/s2] user: What is the refund policy?
assistant: Refunds are available within 30 days of purchase.

[bob/s3] user: What do you know about my diet?
assistant: I don't have anything saved about your diet yet.
```

## Postgres and pgvector

Two more scripts keep memory in the same Postgres as `PgCheckpointer`:

- `pgvector_memory.py`: the same chat scenario on `PgVectorStore`. The store sets `retrieval_strategy=RetrievalStrategy.HYBRID` (vector and full-text search fused with RRF) and the agent memory scope overrides it with `SIMILARITY`. Needs `pip install "10xgraph[anthropic,openai,pgvector,pg_checkpoint]"`, the pgvector extension, and `POSTGRES_DSN`, `REDIS_URL`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.
- `multi_tenant_memory.py`: no LLM key. Declares `tenant_columns=["org_id"]` and `extra_columns={"public_id": "text", "plan": "text"}` on both `PgCheckpointer` and `PgVectorStore`. Two orgs reuse one `user_id` and stay isolated, and `alist_threads(config, filters={"plan": {"in": ["pro"]}})` narrows the listing by an extra column. Column values come only from `config["user"]`, which your trusted auth fills (here the script builds it). Never read them from request config, metadata, state or model output.

```bash
export POSTGRES_DSN=postgresql://tenx:tenx@localhost:5432/tenx
export REDIS_URL=redis://localhost:6379/0
python examples/long_term_memory/pgvector_memory.py
python examples/long_term_memory/multi_tenant_memory.py
```

## Using a different model

Change the model string, for example `ReactAgent(model="gpt-5-mini", ...)` with `pip install "10xgraph[openai,qdrant]"`, or `model="gemini-2.5-flash"` with the `google-genai` extra. To embed with Google instead, swap `OpenAIEmbedding` for `GoogleEmbedding`. The memory configuration does not change.

## Next steps

- [RAG over a knowledge base](../rag/)
- [Checkpointers and thread state](../persistence/)
- [Skills](../skills/)
