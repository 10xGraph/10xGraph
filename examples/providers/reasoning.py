"""Turn on model reasoning per provider and read the reasoning back from the reply.

reasoning_config is one dict for all providers; each provider uses the keys it understands.
Run: python examples/providers/reasoning.py [openai|gemini|claude]
"""

import asyncio
import sys

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.core.state.message_block import ReasoningBlock
from tenxgraph.prebuilt.agent import ReactAgent


load_dotenv()

REASONING: dict[str, dict] = {
    # effort goes to the model; summary asks OpenAI for a readable reasoning summary.
    "openai": {
        "model": "gpt-5-mini",
        "api_style": "responses",  # reasoning summaries come from the Responses API
        "reasoning_config": {"effort": "low", "summary": "auto"},
    },
    # thinking_budget is an exact token budget and overrides effort. Without it, effort maps
    # to a budget: low=512, medium=8192, high=24576. include_thoughts is set for you.
    "gemini": {
        "model": "gemini-2.5-flash",
        "reasoning_config": {"thinking_budget": 2048},
    },
    # For current Claude models effort selects adaptive thinking depth. A token budget is
    # not supported and is ignored with a warning.
    "claude": {
        "model": "claude-sonnet-5-5",
        "reasoning_config": {"effort": "medium"},
    },
}


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "gemini"
    agent = ReactAgent(**REASONING[name])
    app = agent.compile()

    question = "A bat and a ball cost 1.10. The bat costs 1.00 more than the ball. Ball price?"
    result = await app.ainvoke(
        {"messages": [Message.text_message(question)]},
        config={"thread_id": f"reasoning-{name}"},
    )
    reply = result["messages"][-1]

    # Reasoning arrives as ReasoningBlock entries next to the TextBlock in reply.content.
    for block in reply.content:
        if isinstance(block, ReasoningBlock):
            print("reasoning:", block.summary[:300])
    print("answer:", reply.text())
    print("reasoning tokens:", reply.usages.reasoning_tokens if reply.usages else 0)
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
