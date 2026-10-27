"""On-demand skills: the model decides when to load a skill.

`SkillConfig(skills_dir=...)` discovers `skills/<name>/SKILL.md` folders and adds an
<available_skills> catalog (name + description) to the system prompt. The model calls
`activate_skill` when a request matches, and `read_skill_resource` to open bundled files.

Run:  python examples/skills/on_demand.py
      python examples/skills/on_demand.py --chat     # small REPL
"""

import asyncio
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from tenxgraph.core.skills import SkillConfig, validate_skill
from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()

SKILLS_DIR = Path(__file__).parent / "skills"
SKILL_LOADED = re.compile(r'<skill_content name="([^"]+)">')


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 18 C, light rain"


agent = ReactAgent(
    model="claude-sonnet-5-5",
    system_prompt=[{"role": "system", "content": "You are a helpful multi-skilled assistant."}],
    tools=[get_weather],  # ordinary tools sit next to the skill tools
    skills=SkillConfig(
        skills_dir=str(SKILLS_DIR),
        hot_reload=True,  # re-read a SKILL.md when it changes on disk
    ),
)


async def ask(app, text: str, thread_id: str) -> None:
    result = await app.ainvoke(
        {"messages": [Message.text_message(text)]},
        {"thread_id": thread_id, "recursion_limit": 15},
    )
    # activate_skill results come back as tool messages wrapped in <skill_content>.
    for msg in result["messages"]:
        if msg.role == "tool" and (match := SKILL_LOADED.match(msg.text() or "")):
            print(f"  skill loaded: {match.group(1)}")
    print("assistant:", result["messages"][-1].text(), "\n")


async def main() -> None:
    # Check skills against the Agent Skills spec before shipping them.
    # The CLI equivalent is `10xgraph skills --validate`.
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        problems = [d for d in validate_skill(skill_dir) if d.level == "error"]
        print(f"{skill_dir.name}: {problems or 'ok'}")

    app = agent.compile(checkpointer=InMemoryCheckpointer())

    if "--chat" in sys.argv:
        while (line := input("you: ").strip()) not in {"", "quit", "exit"}:
            await ask(app, line, "skills-chat")
    else:
        await ask(app, "Review this: def avg(xs): return sum(xs) / len(xs)", "t1")
        await ask(app, "What is the weather in Paris?", "t2")  # no skill needed

    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
