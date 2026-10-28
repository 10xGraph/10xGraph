---
title: 10xGraph Examples
description: Runnable Python examples for 10xGraph, the multi-agent framework. Covers ReAct agents, MCP, streaming, human-in-the-loop, memory, RAG, evaluation and deployment.
section: Examples
group: "Overview"
order: 1
label: All examples
updated: "2026-10-28"
faq:
  - question: Which LLM providers do the examples use?
    answer: They rotate between OpenAI, Google Gemini, Anthropic Claude and DeepSeek. Each example names its model in the code, and any example can switch provider by changing the model string and installing the matching extra.
  - question: Do I need API keys to try 10xGraph?
    answer: Not for graph_basics, testing and parts of resilience and callbacks. Those run without any LLM. The rest need the key for the provider the example uses.
  - question: Where do I start?
    answer: Start with quickstart, then graph_basics and streaming. After that, pick the feature you need from the table below.
---

## Setup

```bash
git clone https://github.com/10xGraph/10xGraph.git
cd 10xGraph
pip install "10xgraph[openai,google-genai,anthropic]"
cp examples/.env.example .env   # fill in the keys you need
python examples/quickstart/quickstart.py
```

Run every example from the repository root. Each folder's README lists its extra dependencies and the provider it uses.

10xGraph is provider-agnostic. Each example picks its model in code, and the examples are spread across OpenAI, Google Gemini, Anthropic Claude and DeepSeek. Multi-agent examples deliberately use a different company for each agent. To switch, change the model string (for example `gpt-5-mini` to `gemini-2.5-flash`) and install the matching extra.

## Learning path

### Getting started

| Example | What it shows |
|---|---|
| [quickstart](quickstart/) | A tool-using `ReactAgent` in a few lines, sync and async |
| [graph_basics](graph_basics/) | `StateGraph` by hand: custom state, reducers, conditional edges, `Command` |
| [streaming](streaming/) | `astream` events, progress from tools, stopping a run |

### Tools and integrations

| Example | What it shows |
|---|---|
| [tools](tools/) | `@tool`, injected params, `ToolResult` state updates, prebuilt tools, client-side tools |
| [mcp](mcp/) | Model Context Protocol servers: local FastMCP, multiple servers, GitHub, Xquik |
| [providers](providers/) | OpenAI, Gemini, Vertex AI, Anthropic, Bedrock, DeepSeek, Ollama, reasoning |
| [dependency_injection](dependency_injection/) | Injecting your own services into tools and nodes with InjectQ |

### Production

| Example | What it shows |
|---|---|
| [graceful_shutdown](graceful_shutdown/) | SIGTERM handling, protected sections, `aclose()` flushing state |
| [resilience](resilience/) | Retries, circuit breaker, cross-provider fallback models, recursion limits |
| [human_in_the_loop](human_in_the_loop/) | `interrupt()` approvals with schema-validated resume, `interrupt_before` |
| [persistence](persistence/) | In-memory, SQLite and Postgres + Redis checkpointers |
| [context_management](context_management/) | Message windows and LLM summarisation for long chats |
| [callbacks](callbacks/) | Invoke hooks, lifecycle hooks, input validators, secret redaction |
| [observability](observability/) | Event publishers: console, OpenTelemetry, Redis, Kafka, RabbitMQ |

### Memory and knowledge

| Example | What it shows |
|---|---|
| [long_term_memory](long_term_memory/) | Per-user memory with `MemoryConfig` on Qdrant or Mem0 |
| [rag](rag/) | `RAGAgent` with retrieval and optional reranking |

### Agent patterns

| Example | What it shows |
|---|---|
| [structured_output](structured_output/) | Pydantic output with validation and repair |
| [multi_agent](multi_agent/) | Supervisor, swarm, manual handoff, plan-act-reflect |
| [skills](skills/) | Agent Skills (SKILL.md) loaded on demand or per session |

### Multimodal and realtime

| Example | What it shows |
|---|---|
| [multimodal](multimodal/) | Images, documents, audio and video input; media stores |
| [realtime_audio](realtime_audio/) | Voice agent with Gemini Live |

### Quality

| Example | What it shows |
|---|---|
| [testing](testing/) | Unit tests with `TestAgent`, `QuickTest` and mocks, no LLM calls |
| [evaluation](evaluation/) | Eval sets, trajectory and LLM-judge criteria, user simulation, reports |

### Deployment

| Example | What it shows |
|---|---|
| [serving](serving/) | Serve a graph with `10xgraph dev` or Docker, call it over REST, SSE and the TS client |

### Use cases (Python SDK)

| Example | What it shows |
|---|---|
| [use_cases/customer_support](use_cases/customer_support/) | Swarm support desk with refund approvals, memory and persistence |
| [use_cases/research_assistant](use_cases/research_assistant/) | Plan-act-reflect researcher that writes a structured report |

### Full-stack apps

| Example | What it shows |
|---|---|
| [apps/rag_assistant](apps/rag_assistant/) | ReAct + RAG docs assistant: API server, web UI, Postgres, Redis, Qdrant, Docker Compose |
| [apps/copilotkit_agui](apps/copilotkit_agui/) | CopilotKit frontend talking to a 10xGraph agent over AG-UI |
