---
title: Persist Agent Conversations with SQLite, PostgreSQL and Redis Checkpointers in 10xGraph
description: Save and resume a Python AI agent conversation by thread_id with 10xGraph's InMemory, SQLite and PostgreSQL plus Redis checkpointers, and read state back.
section: Examples
group: "Production"
order: 110
label: Persistence
updated: "2026-10-28"
faq:
  - question: Which checkpointer should I use?
    answer: InMemoryCheckpointer for tests and demos, SqliteCheckpointer for a single-user app or a local agent with one database file, and PgCheckpointer (PostgreSQL plus Redis) for servers that run several processes or need durable shared state.
  - question: How does a conversation continue across runs?
    answer: Pass the same thread_id in the run config. The graph loads the saved state for that thread, appends the new user message, and saves the result again.
  - question: Why does PgCheckpointer need Redis?
    answer: PostgreSQL is the durable store. Redis caches hot thread state so frequent reads during a run do not hit the database. You provide both connection settings.
  - question: Do I have to call setup before using a checkpointer?
    answer: SqliteCheckpointer and PgCheckpointer create their tables on first use. Calling await checkpointer.asetup() makes that explicit, and it is safe to repeat.
---

## What you will build

The same small weather agent, run against three checkpointers. Each script chats on a fixed `thread_id`, so a conversation continues where it left off. With SQLite and PostgreSQL you run the script twice, in two separate processes: the first run tells the agent your name and city, and the second run asks it to recall them. After each chat the script lists threads and reads the saved state and messages back.

The agent uses DeepSeek `deepseek-chat` through its OpenAI-compatible endpoint, set with `base_url` and `api_key`.

## Key concepts

- `InMemoryCheckpointer()`: the default. Keeps state in the process, so nothing survives a restart.
- `SqliteCheckpointer(db_path)`: one SQLite file for state, messages and threads. Defaults to `~/.10xgraph/checkpointer.db` when `db_path` is omitted.
- `PgCheckpointer(postgres_dsn=..., redis_url=...)`: PostgreSQL for durable storage plus Redis for the hot cache. Other arguments are `pg_pool`, `redis`, `schema`, `pool_config` and `redis_pool_config`.
- `graph.compile(checkpointer=...)`: attaches the checkpointer to the graph.
- `thread_id` and `user_id` in the run config: `thread_id` selects the conversation, `user_id` scopes thread ownership.
- Read methods: `aget_state(config)`, `alist_messages(config, search=, offset=, limit=)`, `alist_threads(config, search=, offset=, limit=)`, `aget_thread(config)`.
- `aput_thread(config, ThreadInfo(...))`: optionally registers thread metadata such as a name.
- `await graph.aclose()`: flushes pending writes and releases connections.

## Prerequisites

```bash
pip install "10xgraph[openai,sqlite_checkpoint,pg_checkpoint]" python-dotenv
export DEEPSEEK_API_KEY="..."
```

For `postgres.py`, start the services (Docker required):

```bash
docker compose -f examples/persistence/docker-compose.yml up -d
```

`POSTGRES_DSN` and `REDIS_URL` override the defaults (`postgresql://tenx:tenx@localhost:5432/tenx` and `redis://localhost:6379/0`).

## Run it

```bash
python examples/persistence/in_memory.py

python examples/persistence/sqlite.py   # run it twice
python examples/persistence/sqlite.py

python examples/persistence/postgres.py # run it twice
python examples/persistence/postgres.py
```

Delete `conversations.db` (created in the working directory) to start over with SQLite. For Postgres, use `docker compose ... down -v`.

## How it works

### Same thread, separate runs

```python
config = {"thread_id": "customer-42", "user_id": "demo-user"}
saved = await checkpointer.alist_messages(config)
print(f"Restored {len(saved)} message(s) for this thread.")
if not saved:
    await chat(graph, "Hi, my name is Sam and I live in Lisbon.", config)
else:
    await chat(graph, "What is my name, and what is the weather where I live?", config)
```

On the second process, the checkpointer returns the earlier messages before the agent runs, and the agent answers from them. `in_memory.py` cannot cross processes, so it makes both calls in one process on the same thread to show the same behaviour.

### Choosing a checkpointer

```python
checkpointer = InMemoryCheckpointer()
checkpointer = SqliteCheckpointer("conversations.db")
checkpointer = PgCheckpointer(
    postgres_dsn="postgresql://tenx:tenx@localhost:5432/tenx",
    redis_url="redis://localhost:6379/0",
)
```

Only this line changes between the three files. `PgCheckpointer` enforces user isolation by default, so reads with a different `user_id` do not see the thread. Pass `enforce_user_isolation=False` for a single-tenant app.

### Reading back

```python
state = await checkpointer.aget_state(config)       # full AgentState for the thread
messages = await checkpointer.alist_messages(config) # stored messages, supports paging
threads = await checkpointer.alist_threads(config)   # ThreadInfo records
```

The graph writes state and messages itself. Thread metadata is separate: the scripts register a `ThreadInfo` with `aput_thread` so `alist_threads` has something to list. `in_memory.py` prints `state.context` instead of `alist_messages`, because the in-memory checkpointer can return the messages of a finished run twice.

### Closing

`await graph.aclose()` waits for background checkpoint writes and then releases the checkpointer's connections. See the graceful shutdown example.

## Expected output

Illustrative; model wording will differ.

First run of `sqlite.py`:

```text
Restored 0 message(s) for this thread.

user:      Hi, my name is Sam and I live in Lisbon.
assistant: Nice to meet you, Sam! Anything I can help with in Lisbon?

state: 2 messages, step 1
threads: [('customer-42', 'Customer 42 chat')]
```

Second run:

```text
Restored 2 message(s) for this thread.

user:      What is my name, and what is the weather where I live?
assistant: Your name is Sam. It is 21 C and sunny in Lisbon.
```

## Using a different model

Change `model`, and remove `base_url` and `api_key`. For example `model="gpt-5-mini"` with `pip install "10xgraph[openai]"` and `OPENAI_API_KEY`, or `model="claude-sonnet-5-5"` with `pip install "10xgraph[anthropic]"` and `ANTHROPIC_API_KEY`.

## Next steps

- [Human in the loop](../human_in_the_loop/): paused runs are stored by the checkpointer and can resume in another process.
- [Context management](../context_management/): keep long saved threads within the model's context window.
- [Graceful shutdown](../graceful_shutdown/): release database connections cleanly on SIGTERM.
