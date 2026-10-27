"""Helper agent: breaks a goal into short task titles, served by DeepSeek.

It is a second, separate 10xGraph graph, called from a tool of the main assistant. Using a
model from another company shows that agents in one app do not have to share a provider.
"""

import os
import uuid

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent


load_dotenv()

# No tools, so ReactAgent wires a single MAIN node. base_url and api_key are forwarded to the
# Agent; unknown model names use the OpenAI-compatible client.
_planner = ReactAgent(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    system_prompt=[
        {
            "role": "system",
            "content": (
                "You are a task planner. Break the user's goal into 3 to 5 concrete steps. "
                "Reply with one step per line, no numbering, no extra text. "
                "Keep each step under 8 words."
            ),
        }
    ],
).compile()


async def plan_steps(goal: str) -> list[str]:
    """Ask the DeepSeek planner for steps. Each call uses a fresh thread."""
    result = await _planner.ainvoke(
        {"messages": [Message.text_message(goal)]},
        {"thread_id": f"planner-{uuid.uuid4()}"},
    )
    text = result["messages"][-1].text() if result.get("messages") else ""
    steps = [line.strip(" -*\t") for line in text.splitlines()]
    return [s for s in steps if s][:5]
