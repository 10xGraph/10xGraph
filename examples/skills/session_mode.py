"""Session-mode skills: a fixed persona per tenant, no activate_skill round trip.

With `mode="session"` the framework reads a state field (`preload_from`) before every
model call and injects that skill's instructions. The same agent serves two tenants,
each with its own persona from `personas/<name>/SKILL.md`.

Run:  python examples/skills/session_mode.py
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from tenxgraph.core.skills import SkillConfig
from tenxgraph.core.state import AgentState, Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()


class TenantState(AgentState):
    # Name of the skill to preload. Set it from the authenticated tenant, never from
    # client input, or users could switch to another tenant's persona.
    persona: str = ""


agent = ReactAgent(
    model="claude-sonnet-5-5",
    state=TenantState(),
    system_prompt=[{"role": "system", "content": "You answer customer questions."}],
    skills=SkillConfig(
        skills_dir=str(Path(__file__).parent / "personas"),
        mode="session",
        preload_from="persona",  # reads state.persona on every call
    ),
)


async def ask(app, tenant: str, persona: str, text: str) -> None:
    result = await app.ainvoke(
        {"messages": [Message.text_message(text)], "state": {"persona": persona}},
        {"thread_id": f"{tenant}-1"},
    )
    print(f"[{tenant}] user: {text}")
    print("assistant:", result["messages"][-1].text(), "\n")


async def main() -> None:
    app = agent.compile(checkpointer=InMemoryCheckpointer())
    question = "I want my money back. How do I do that?"
    await ask(app, "acme", "acme-support", question)
    await ask(app, "globex", "globex-support", question)
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
