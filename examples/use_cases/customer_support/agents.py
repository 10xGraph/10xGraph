"""Builds the support swarm: three agents, three companies, one shared memory store.

SwarmAgent wires peer-to-peer handoffs. Each member is an ordinary `Agent`
with its own model, tools and memory settings.
"""

from __future__ import annotations

import prompts
from config import MEMORY_COLLECTION, QDRANT_PATH
from tools import (
    issue_refund,
    lookup_customer,
    lookup_order,
    run_diagnostics,
    search_knowledge_base,
)

from tenxgraph.core.graph import Agent, ToolNode
from tenxgraph.prebuilt.agent import SwarmAgent
from tenxgraph.prebuilt.agent.swarm import SwarmMemberConfig
from tenxgraph.storage.store import (
    MemoryConfig,
    OpenAIEmbedding,
    QdrantStore,
    UserMemoryConfig,
)


def build_memory_store() -> QdrantStore:
    """Local file-based Qdrant. Embeddings come from OpenAI (needs OPENAI_API_KEY)."""
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantStore(
        embedding=OpenAIEmbedding(model="text-embedding-3-small"),
        path=str(QDRANT_PATH),
        collection=MEMORY_COLLECTION,
    )


def build_swarm(store: QdrantStore) -> SwarmAgent:
    # Memory is keyed by config["user_id"], so each customer only sees their own notes.
    # PRELOAD: the store is searched once per run and the hits are added to the
    # request, so the model needs no memory tool (good for a cheap triage model).
    preload = MemoryConfig(
        store=store,
        retrieval_mode="preload",
        limit=5,
        user_memory=UserMemoryConfig(),
        agent_memory=None,
    )
    # POSTLOAD: the model gets `user_memory_tool` and decides when to search
    # or remember. The specialists use this so they can write notes.
    postload = MemoryConfig(
        store=store,
        retrieval_mode="postload",
        limit=5,
        user_memory=UserMemoryConfig(),
        agent_memory=None,
    )

    triage = Agent(
        model="gemini-2.5-flash",
        system_prompt=prompts.TRIAGE,
        tool_node=ToolNode([lookup_customer]),
        memory=preload,
        reasoning_config=None,  # routing is easy; skip thinking tokens
    )
    billing = Agent(
        model="gpt-5-mini",
        system_prompt=prompts.BILLING,
        tool_node=ToolNode([lookup_order, issue_refund]),
        memory=postload,
    )
    technical = Agent(
        model="claude-sonnet-5-5",
        system_prompt=prompts.TECHNICAL,
        tool_node=ToolNode([search_knowledge_base, run_diagnostics]),
        memory=postload,
    )

    return SwarmAgent(
        members={
            "TRIAGE": SwarmMemberConfig(
                agent=triage,
                can_handoff_to=["BILLING", "TECHNICAL"],
                description="Front desk. Greets customers and routes them.",
            ),
            "BILLING": SwarmMemberConfig(
                agent=billing,
                can_handoff_to=["TECHNICAL"],
                description="Billing: charges, invoices and refunds.",
            ),
            "TECHNICAL": SwarmMemberConfig(
                agent=technical,
                can_handoff_to=["BILLING"],
                description="Technical support: errors, API problems, how-to questions.",
            ),
        },
        entry="TRIAGE",
    )
