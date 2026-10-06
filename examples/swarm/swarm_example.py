"""Swarm agent example — peer-to-peer multi-agent handoff.

Each member is a fully-configured ``Agent`` created independently, so every
member can have its own model, tools, memory, skills, retry config, etc.
``SwarmAgent`` auto-injects the handoff tools; you never add them yourself.

Run::

    OPENAI_API_KEY=sk-... python examples/swarm/swarm_example.py
"""

from __future__ import annotations

import asyncio

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.prebuilt.agent.swarm import SwarmAgent, SwarmMemberConfig


# ---------------------------------------------------------------------------
# Domain tools
# ---------------------------------------------------------------------------


def web_search(query: str) -> str:
    """Search the web and return relevant results."""
    return f"[web_search] Results for '{query}': lots of relevant information."


def draft_document(topic: str, research_notes: str) -> str:
    """Draft a polished document on the given topic using research notes."""
    return f"[draft_document] Document on '{topic}':\n{research_notes}"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email with the given subject and body."""
    return f"[send_email] Email sent to {to!r} with subject {subject!r}."


# ---------------------------------------------------------------------------
# Scenario 1 — 3-member research/writing swarm
#
# * TRIAGE routes to the right specialist.
# * RESEARCHER does deep research (has memory for context).
# * WRITER writes the final document (specialised model).
# ---------------------------------------------------------------------------

SYSTEM_TRIAGE = [
    {
        "role": "system",
        "content": (
            "You are a triage coordinator. Analyse the request and route it to "
            "the best specialist. Use transfer_to_researcher for research tasks "
            "and transfer_to_writer for writing-only tasks."
        ),
    }
]

SYSTEM_RESEARCHER = [
    {
        "role": "system",
        "content": (
            "You are a meticulous researcher. Use your web_search tool to gather "
            "information, then hand off to the writer with your findings."
        ),
    }
]

SYSTEM_WRITER = [
    {
        "role": "system",
        "content": (
            "You are an expert technical writer. Produce clear, well-structured "
            "documents from the researcher's notes."
        ),
    }
]


def build_research_swarm() -> SwarmAgent:
    # Each member is built completely independently
    triage_agent = Agent(
        model="gpt-4o-mini",
        system_prompt=SYSTEM_TRIAGE,
        # No tools needed — just routes
    )

    researcher_agent = Agent(
        model="gpt-4o",
        system_prompt=SYSTEM_RESEARCHER,
        tool_node=ToolNode([web_search]),
        # Could add: memory=MemoryConfig(...), retry_config=..., etc.
    )

    writer_agent = Agent(
        model="gpt-4o",
        system_prompt=SYSTEM_WRITER,
        tool_node=ToolNode([draft_document]),
        # Could add: skills=SkillConfig(...), multimodal_config=..., etc.
    )

    return SwarmAgent(
        members={
            "TRIAGE": SwarmMemberConfig(
                agent=triage_agent,
                can_handoff_to=["RESEARCHER", "WRITER"],
                description="Triages requests and routes to the right specialist.",
            ),
            "RESEARCHER": SwarmMemberConfig(
                agent=researcher_agent,
                can_handoff_to=["WRITER"],
                description="Performs deep web research on a topic.",
            ),
            "WRITER": SwarmMemberConfig(
                agent=writer_agent,
                # can_handoff_to=None → auto-routes to all others if needed,
                # but in practice WRITER is a terminal node.
                description="Writes polished documents from research notes.",
            ),
        },
        entry="TRIAGE",
    )


# ---------------------------------------------------------------------------
# Scenario 2 — bidirectional 2-member swarm
#
# Each member can route to the other, enabling iterative collaboration.
# ---------------------------------------------------------------------------


def build_bidirectional_swarm() -> SwarmAgent:
    analyst_agent = Agent(
        model="gpt-4o-mini",
        system_prompt=[
            {
                "role": "system",
                "content": (
                    "You analyse requests and gather data. When ready for "
                    "delivery, hand off to the communicator."
                ),
            }
        ],
        tool_node=ToolNode([web_search]),
    )

    communicator_agent = Agent(
        model="gpt-4o-mini",
        system_prompt=[
            {
                "role": "system",
                "content": (
                    "You communicate results. If you need more data, hand off back to the analyst."
                ),
            }
        ],
        tool_node=ToolNode([send_email]),
    )

    return SwarmAgent(
        members={
            "ANALYST": SwarmMemberConfig(
                agent=analyst_agent,
                can_handoff_to=["COMMUNICATOR"],
                description="Analyses data and prepares reports.",
            ),
            "COMMUNICATOR": SwarmMemberConfig(
                agent=communicator_agent,
                can_handoff_to=["ANALYST"],
                description="Delivers results to stakeholders via email.",
            ),
        },
        entry="ANALYST",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_scenario_1() -> None:
    print("\n--- Scenario 1: Research & Writing Swarm ---")
    swarm = build_research_swarm()
    app = swarm.compile()

    result = await app.ainvoke(
        {"message": "Research and write a short report on quantum computing trends."},
        config={"thread_id": "research-1"},
    )
    print("Final messages:")
    for msg in result.get("context", []):
        print(f"  [{msg.get('role', '?')}] {str(msg.get('content', ''))[:120]}")


async def run_scenario_2() -> None:
    print("\n--- Scenario 2: Bidirectional Analyst-Communicator Swarm ---")
    swarm = build_bidirectional_swarm()
    app = swarm.compile()

    result = await app.ainvoke(
        {"message": "Find the latest AI news and email a summary to ceo@example.com."},
        config={"thread_id": "bidir-1"},
    )
    print("Final messages:")
    for msg in result.get("context", []):
        print(f"  [{msg.get('role', '?')}] {str(msg.get('content', ''))[:120]}")


async def main() -> None:
    await run_scenario_1()
    await run_scenario_2()


if __name__ == "__main__":
    asyncio.run(main())
