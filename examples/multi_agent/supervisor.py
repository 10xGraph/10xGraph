"""Supervisor team: one model routes work to specialist workers, each from a different company.

SupervisorTeamAgent builds the supervisor itself from `supervisor_model` (Anthropic here).
Each worker is an ordinary Agent you construct, so it carries its own model, tools and
endpoint: OpenAI, Gemini and DeepSeek below. The supervisor answers with a worker name or
FINISH, and the graph follows.

Run: python examples/multi_agent/supervisor.py
     (needs ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY)
"""

import asyncio
import os

from dotenv import load_dotenv

from tenxgraph.core.graph import Agent, ToolNode
from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import SupervisorTeamAgent, WorkerConfig


load_dotenv()


def web_search(query: str) -> str:
    """Search the web and return a short list of findings."""
    return (
        f"Findings for '{query}': global EV sales grew about 25% year over year; "
        "China accounts for roughly 60% of new registrations; battery pack prices fell 14%."
    )


def calculate_growth(start: float, end: float) -> str:
    """Return the percentage change from start to end."""
    return f"{(end - start) / start * 100:.1f}% change"


researcher = Agent(
    model="gpt-5-mini",  # OpenAI
    system_prompt=[
        {"role": "system", "content": "You research facts. Use web_search, then report findings."}
    ],
    tool_node=ToolNode([web_search]),
)

analyst = Agent(
    model="gemini-2.5-flash",  # Google
    system_prompt=[
        {"role": "system", "content": "You analyse numbers. Use calculate_growth when useful."}
    ],
    tool_node=ToolNode([calculate_growth]),
)

writer = Agent(
    model="deepseek-chat",  # DeepSeek, through its OpenAI-compatible endpoint
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    system_prompt=[
        {"role": "system", "content": "You write a concise summary from the work done so far."}
    ],
)

team = SupervisorTeamAgent(
    supervisor_model="claude-sonnet-5-5",  # Anthropic
    workers={
        "RESEARCHER": WorkerConfig(agent=researcher, description="Finds facts with web search."),
        "ANALYST": WorkerConfig(agent=analyst, description="Does calculations and comparisons."),
        "WRITER": WorkerConfig(agent=writer, description="Writes the final answer for the user."),
    },
    max_rounds=6,  # Upper bound on supervisor -> worker hops per turn.
)


async def main() -> None:
    app = team.compile()
    result = await app.ainvoke(
        {"messages": [Message.text_message("Brief me on the electric vehicle market.")]},
        config={"thread_id": "supervisor-1", "recursion_limit": 40},
    )
    for msg in result["messages"]:
        if msg.role == "assistant" and msg.text():
            model = (msg.metadata or {}).get("model") or "unknown model"
            print(f"[{model}] {msg.text()[:300]}\n")
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
