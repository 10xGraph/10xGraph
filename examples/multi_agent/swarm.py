"""Swarm: peer agents hand work to each other, and each peer runs on a different company's model.

SwarmAgent injects a `transfer_to_<member>` tool into every member. When a model calls
one, control moves to that member. No supervisor is involved. Triage runs on OpenAI, the
researcher on Anthropic and the writer on Gemini.

Run: python examples/multi_agent/swarm.py
     (needs OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY)
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.core.graph import Agent, ToolNode
from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import SwarmAgent, SwarmMemberConfig


load_dotenv()


def web_search(query: str) -> str:
    """Search the web and return findings."""
    return f"Findings for '{query}': solid-state batteries promise 2x energy density by 2030."


def save_document(title: str, body: str) -> str:
    """Save a finished document."""
    return f"Saved '{title}' ({len(body)} characters)."


triage = Agent(
    model="gpt-5-mini",  # OpenAI
    system_prompt=[
        {
            "role": "system",
            "content": "Route the request. Research questions go to the researcher; "
            "writing-only requests go to the writer. Do not answer yourself.",
        }
    ],
)

researcher = Agent(
    model="claude-sonnet-5-5",  # Anthropic
    system_prompt=[
        {
            "role": "system",
            "content": "Research with web_search. When you have findings, hand off to the writer.",
        }
    ],
    tool_node=ToolNode([web_search]),
)

writer = Agent(
    model="gemini-2.5-flash",  # Google
    system_prompt=[
        {"role": "system", "content": "Write a short article from the conversation, then save it."}
    ],
    tool_node=ToolNode([save_document]),
)

swarm = SwarmAgent(
    members={
        "TRIAGE": SwarmMemberConfig(
            agent=triage,
            can_handoff_to=["RESEARCHER", "WRITER"],
            description="Routes requests to the right specialist.",
        ),
        "RESEARCHER": SwarmMemberConfig(
            agent=researcher,
            can_handoff_to=["WRITER"],
            description="Researches a topic and gathers facts.",
        ),
        # No can_handoff_to: the writer ends the run when it replies without a handoff.
        "WRITER": SwarmMemberConfig(agent=writer, description="Writes and saves articles."),
    },
    entry="TRIAGE",
)


async def main() -> None:
    app = swarm.compile()
    result = await app.ainvoke(
        {"messages": [Message.text_message("Write a short article on solid-state batteries.")]},
        config={"thread_id": "swarm-1", "recursion_limit": 30},
    )
    for msg in result["messages"]:
        if msg.role == "assistant" and msg.text():
            model = (msg.metadata or {}).get("model") or "unknown model"
            print(f"[{model}] {msg.text()[:300]}\n")
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
