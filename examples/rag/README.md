---
title: RAG Agent with Qdrant and Reranking in 10xGraph (Python)
description: Build a retrieval-augmented generation agent in Python with 10xGraph RAGAgent, a local Qdrant store, Gemini embeddings and optional Cohere or cross-encoder reranking.
section: Examples
group: "Memory and knowledge"
order: 160
label: RAG
updated: "2026-10-28"
faq:
  - question: Why does the example index with system_user_id("knowledge_base")?
    answer: The store requires an owner for every record. RAGAgent searches as system_user_id("knowledge_base") by default, so shared documents indexed under that id are visible to every user and never mix with per-user memory.
  - question: What do top_k and top_n do?
    answer: top_k is how many candidates the store returns. With a reranker, top_n is how many of those are passed to the model. Without a reranker all top_k documents are used and top_n is ignored.
  - question: Are retrieved documents saved in the thread?
    answer: No. They are held in a per-run cache and added to the outgoing request only, so the saved thread keeps the user's original message.
---

## What you will build

A support agent that answers questions from a small policy knowledge base. The script ingests ten text chunks into a local Qdrant store, then runs `RAGAgent` three ways: plain vector search, with Cohere reranking, and with a local cross-encoder reranker. Gemini writes the answers and also embeds the documents.

## Key concepts

- `RAGAgent(store, agent, reranker, top_k, top_n)`: a prebuilt graph, `RETRIEVE` then optional `RERANK` then `SYNTHESIZE`.
- `create_local_qdrant_store`: file-based Qdrant for the knowledge base.
- `GoogleEmbedding`: embeds chunks and queries with `gemini-embedding-001`.
- `system_user_id("knowledge_base")`: the reserved owner of shared documents. `RAGAgent` searches as this owner unless you set `store_config` or `store_config_builder`.
- `CohereReranker` and `CrossEncoderReranker`: rerankers that reorder `top_k` candidates and keep `top_n`.
- `store_config_builder`: builds the search config from the run config, for per-user or per-tenant knowledge bases.

## Prerequisites

```bash
pip install "10xgraph[google-genai,qdrant]"
export GEMINI_API_KEY=...
# optional rerankers
pip install cohere && export COHERE_API_KEY=...
pip install sentence-transformers
```

No Docker services are required.

## Run it

```bash
python examples/rag/rag_example.py
```

The Cohere and cross-encoder runs are skipped when their requirements are missing.

## How it works

Ingest once. Every chunk is written under the shared knowledge-base owner:

```python
owner = {"user_id": system_user_id("knowledge_base")}
for chunk in DOCUMENTS:
    await store.astore(owner, chunk, category="support-policy")
```

`RAGAgent` takes a plain `Agent` and calls it itself, adding the retrieved documents to the request:

```python
rag = RAGAgent(store=store, agent=Agent(model="gemini-2.5-flash", system_prompt=SYSTEM_PROMPT), top_k=4)
app = rag.compile()
```

With a reranker, retrieve more candidates than the model needs:

```python
RAGAgent(store=store, agent=build_agent(), reranker=reranker, top_k=8, top_n=3)
```

The documents are fenced in a block marked as not-instructions and are never written to `state.context`. To scope search per user instead of one shared base, pass `store_config_builder=lambda cfg: {"user_id": cfg["user_id"]}` and index with the same ids.

## Expected output

Illustrative output.

```text
--- plain
Q: Can I return a digital download?
A: No. Digital downloads are non-refundable once accessed.

--- cohere
Q: Can I return a damaged item?
A: Yes. Damaged or defective items are replaced free of charge at any time.
```

## Using a different model

Change the `model` string in `build_agent()`, for example `"claude-sonnet-5-5"` with `pip install "10xgraph[anthropic]"`, or `"gpt-5-mini"` with the `openai` extra. Embeddings are separate: swap `GoogleEmbedding` for `OpenAIEmbedding` if you prefer, and re-ingest, since vector sizes differ.

## Next steps

- [Long-term memory](../long_term_memory/)
- [Prebuilt ReAct agent](../quickstart/)
- [Evaluation](../evaluation/)
