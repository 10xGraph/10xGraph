"""Keep only the most recent turns in the prompt with MessageContextManager.

After each run the context holds at most the last 3 user messages (plus any system
messages), so token use stays flat in a long chat. Needs trim_context=True on the agent.
Run: python examples/context_management/message_window.py
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.core.state import Message, MessageContextManager
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity


load_dotenv()

QUESTIONS = [
    "My favourite colour is green. Remember that.",
    "I work as a nurse.",
    "I have a dog called Biscuit.",
    "I live in Porto.",
    "What is my favourite colour?",  # the first turn is gone by now
]


async def main() -> None:
    agent = ReactAgent(
        model="claude-sonnet-5-5",
        system_prompt=[{"role": "system", "content": "Answer briefly."}],
        # Counts USER messages. The oldest turns are dropped; system messages stay.
        # remove_tool_msgs=True would also drop tool calls and tool results.
        context_manager=MessageContextManager(max_messages=3),
        # Without this flag the agent never calls the context manager.
        trim_context=True,
    )
    graph = agent.compile(checkpointer=InMemoryCheckpointer())
    config = {"thread_id": "window-demo", "user_id": "demo-user"}

    async with graph:
        for question in QUESTIONS:
            result = await graph.ainvoke(
                {"messages": [Message.text_message(question)]},
                config,
                # PARTIAL adds the stored context and summary to the reply.
                response_granularity=ResponseGranularity.PARTIAL,
            )
            users = sum(1 for m in result["context"] if m.role == "user")
            print(f"user: {question}")
            print(f"assistant: {result['messages'][-1].text()}")
            print(f"  -> context: {len(result['context'])} messages, {users} from the user\n")


if __name__ == "__main__":
    asyncio.run(main())
