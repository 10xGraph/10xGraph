---
title: Serve a 10xGraph Agent as a REST and SSE API
description: Deploy a 10xGraph Python AI agent with the 10xgraph-api server. Configure 10xgraph.json, call /v1/graph/invoke and /v1/graph/stream, and use the TypeScript client.
section: Examples
group: "Deployment"
order: 240
label: Serving an agent
updated: "2026-10-28"
faq:
  - question: How does the server find my graph?
    answer: 10xgraph.json has an "agent" field in module:attribute form, for example "graph.agent:app". The server imports that module from the project directory and serves the CompiledGraph it finds there.
  - question: What is the difference between /v1/graph/invoke and /v1/graph/stream?
    answer: invoke waits for the run to finish and returns the final messages in one JSON body. stream returns the same run as newline-delimited JSON chunks as they are produced, which suits chat UIs.
  - question: Do I need a checkpointer to serve an agent?
    answer: Yes for multi-turn threads. The server keeps each conversation under the thread_id you send, and the checkpointer stores it. InMemoryCheckpointer loses threads on restart, so use PgCheckpointer in production.
  - question: Are the old "10xgraph api" and "10xgraph play" commands still there?
    answer: No. The current CLI has one server command, "10xgraph dev". It starts the API and opens the hosted playground. Add --no-open to run only the API.
---

# Serve a 10xGraph agent as an API

## What you will build

A small travel assistant on Anthropic `claude-sonnet-5-5`, compiled into a graph, and served by the `10xgraph-api` package. You get REST and streaming endpoints, a Dockerfile and a TypeScript client call, without writing server code.

The project is two files: `graph/agent.py` builds the graph, and `10xgraph.json` tells the server where to find it.

## Key concepts

- `ReactAgent(...).compile(checkpointer=...)`: builds the agent graph. The compiled graph must be a module-level variable.
- `10xgraph.json`: the server config. `"agent": "graph.agent:app"` points at that variable.
- `/v1/graph/invoke`: runs the graph and returns the final messages.
- `/v1/graph/stream`: runs the graph and streams chunks as newline-delimited JSON.
- `thread_id` in `config`: keeps a conversation across requests, through the checkpointer.
- `10xgraph-client`: the npm package that wraps these endpoints.

## Prerequisites

```bash
pip install "10xgraph[anthropic]" 10xgraph-api python-dotenv
export ANTHROPIC_API_KEY=...        # or put it in a .env file next to 10xgraph.json
npm install 10xgraph-client         # only for the TypeScript example
```

Docker is needed only for `10xgraph build` output.

## Run it

Run everything from this folder, because the `graph.agent` import path is relative to it.

```bash
cd examples/serving
python graph/agent.py               # optional: check the graph without the server
10xgraph dev --no-open              # API only, on http://127.0.0.1:8000
10xgraph dev                        # API plus the hosted playground in your browser
10xgraph build --docker-compose     # writes Dockerfile, .dockerignore and docker-compose.yml
```

Call the running server:

```bash
curl -s http://127.0.0.1:8000/ping

curl -s http://127.0.0.1:8000/v1/graph/invoke \
  -H 'Content-Type: application/json' \
  -d '{
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Weather in Lisbon?"}]}],
        "config": {"thread_id": "demo-1"}
      }'

curl -N http://127.0.0.1:8000/v1/graph/stream \
  -H 'Content-Type: application/json' \
  -d '{
        "messages": [{"role": "user", "content": [{"type": "text", "text": "And the local time there?"}]}],
        "config": {"thread_id": "demo-1"}
      }'
```

Message content is a list of typed blocks, so a plain string is rejected. Reusing `thread_id` continues the same conversation.

The TypeScript version is in `client.ts`:

```bash
npx tsx client.ts
```

## How it works

`graph/agent.py` builds the agent and exposes the compiled graph as `app`:

```python
agent = ReactAgent(
    model="claude-sonnet-5-5",
    system_prompt=[...],
    tools=[get_weather, get_time],
)
app = agent.compile(checkpointer=InMemoryCheckpointer())
```

`10xgraph.json` is the only wiring the server needs:

```json
{
  "agent": "graph.agent:app",
  "env": ".env",
  "auth": null,
  "remote_tools": []
}
```

`env` names the dotenv file loaded before the graph is imported, and `auth: null` leaves the API open for local work. For anything reachable from a network, set `"auth": "jwt"` and `JWT_SECRET_KEY`. `10xgraph init` scaffolds a project with the same layout, including a production template with auth, rate limiting and tests.

The server wraps results in an envelope. An invoke response looks like `{"data": {"messages": [...]}, "metadata": {...}}`. The stream endpoint sends one JSON object per line, each with an `event` (`message`, `state`, `updates` or `error`), and the message chunks carry the assistant text.

In `client.ts` the TypeScript client does the same two calls:

```ts
const client = new TenxGraphClient({ baseUrl: 'http://localhost:8000' });
const result = await client.invoke([Message.text_message('Weather in Lisbon?', 'user')], {
  config: { thread_id: 'ts-demo' },
});
for await (const chunk of client.stream(messages, { config: { thread_id: 'ts-demo' } })) {
  if (chunk.event === 'message') console.log(chunk.message?.content);
}
```

`InMemoryCheckpointer` is fine for a demo. It loses threads on restart and is not shared between workers, so switch to `PgCheckpointer` (the `pg_checkpoint` extra) before running more than one process.

## Expected output

Illustrative; text and ids will differ.

```text
$ curl -s http://127.0.0.1:8000/ping
{"data":"pong", ...}

$ curl -s http://127.0.0.1:8000/v1/graph/invoke ...
{"data":{"messages":[{"role":"assistant","content":[{"type":"text","text":"It is sunny in Lisbon, 24 C."}], ...}]}, ...}
```

## Using a different model

Change the model string in `graph/agent.py` and install the matching extra. For OpenAI use `model="gpt-5-mini"` with `pip install "10xgraph[openai]"` and `OPENAI_API_KEY`; for Google use `model="gemini-2.5-flash"` with `pip install "10xgraph[google-genai]"` and `GEMINI_API_KEY`. The server, config and curl calls do not change.

## Next steps

- [Evaluation](../evaluation/) runs the same kind of agent through `10xgraph eval`.
- [Testing](../testing/) covers offline tests for the graph before you deploy it.
