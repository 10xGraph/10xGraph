"""Review the tool calls an agent wants to make before any of them run.

compile(interrupt_before=["TOOL"]) pauses the graph just before the tool node. We print
the pending calls, then continue the same thread. No code change is needed in the
tools. Run: python examples/human_in_the_loop/interrupt_before.py
"""

import asyncio
import json

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity


load_dotenv()


def delete_account(user_id: str) -> str:
    """Permanently delete a user account."""
    return f"Account {user_id} deleted."


async def main() -> None:
    agent = ReactAgent(
        model="gpt-5-mini",
        system_prompt=[{"role": "system", "content": "You manage user accounts. Use your tools."}],
        tools=[delete_account],
    )
    # "TOOL" is the name ReactAgent gives its tool node. A checkpointer is required so the
    # paused thread can be loaded again.
    graph = agent.compile(checkpointer=InMemoryCheckpointer(), interrupt_before=["TOOL"])
    config = {"thread_id": "review-1", "user_id": "ops-1"}

    async with graph:
        paused = await graph.ainvoke(
            {"messages": [Message.text_message("Please delete the account for user u-381.")]},
            config,
            response_granularity=ResponseGranularity.FULL,
        )
        state = paused["state"]
        print("paused before node:", state.execution_meta.interrupted_node)
        for call in state.context[-1].tools_calls or []:
            fn = call["function"]
            print("  wants to call:", fn["name"], json.loads(fn["arguments"]))

        # The reviewer approves by running the thread again. The input carries no new
        # message: the paused node and its pending tool calls are picked up from the checkpoint.
        final = await graph.ainvoke({"messages": []}, config)
        print("assistant:", final["messages"][-1].text())


if __name__ == "__main__":
    asyncio.run(main())
