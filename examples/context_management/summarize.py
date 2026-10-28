"""Compress old turns into a running summary with SummaryContextManager.

The agent talks to Anthropic, but the summaries are written by Gemini, a cheaper model
from another company. Old messages move into state.context_summary and only the recent
ones stay verbatim. Run: python examples/context_management/summarize.py
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.core.state import Message, SummaryContextManager
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity


load_dotenv()

QUESTIONS = [
    "I'm planning a 5 day trip to Japan in April. I like food and quiet places.",
    "I will land in Tokyo and leave from Osaka.",
    "My budget is about 2000 euros without flights.",
    "I am vegetarian.",
    "Suggest a rough day-by-day plan that respects everything I told you.",
]


async def main() -> None:
    summariser = SummaryContextManager(
        # Summarising is a simple task, so a small fast model is enough. Any provider works
        # here; it is detected from the model name, like for an Agent.
        model="gemini-2.5-flash",
        max_messages=6,  # summarise when the context holds more than 6 messages
        keep_recent=4,  # always keep the last 4 non-system messages verbatim
        token_budget=None,  # or set a rough token cap (1 token ~ 4 characters)
        max_summary_tokens=400,
        # api_style only matters for OpenAI summary models: "responses" (default) or "chat".
        # With a Gemini model it is ignored.
    )
    agent = ReactAgent(
        model="claude-sonnet-5-5",
        system_prompt=[{"role": "system", "content": "You are a concise travel planner."}],
        context_manager=summariser,
        trim_context=True,  # required, or the manager is never called
    )
    graph = agent.compile(checkpointer=InMemoryCheckpointer())
    config = {"thread_id": "summary-demo", "user_id": "demo-user"}

    async with graph:
        for question in QUESTIONS:
            result = await graph.ainvoke(
                {"messages": [Message.text_message(question)]},
                config,
                response_granularity=ResponseGranularity.PARTIAL,
            )
            print(f"user: {question}")
            print(f"assistant: {result['messages'][-1].text()[:200]}")
            print(f"  -> context: {len(result['context'])} messages")
            if result["summary"]:
                print(f"  -> context_summary: {result['summary']}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
