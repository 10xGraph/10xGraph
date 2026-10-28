"""Tools that receive framework values: state, config and tool_call_id.

Parameters with these reserved names are filled in by ToolNode and are hidden from
the schema the model sees, so the model never has to supply them.

Run: python examples/tools/injected_params.py   (needs DEEPSEEK_API_KEY)
"""

import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from tenxgraph.core.state import AgentState, Message
from tenxgraph.prebuilt.agent import ReactAgent


load_dotenv()


def whoami(
    topic: str,
    state: AgentState,  # the live graph state, including state.context
    config: dict[str, Any],  # the run config: thread_id, user_id, run_id, your own keys
    tool_call_id: str,  # id of this specific tool call
) -> str:
    """Describe the caller and the current conversation for a topic."""
    user = config.get("user_id")
    plan = config.get("plan", "free")  # custom config keys pass straight through
    return (
        f"topic={topic} user={user} plan={plan} thread={config.get('thread_id')} "
        f"messages_so_far={len(state.context)} call_id={tool_call_id}"
    )


agent = ReactAgent(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    system_prompt=[{"role": "system", "content": "Call whoami once, then report its output."}],
    tools=[whoami],
)


async def main() -> None:
    async with agent.compile() as app:
        result = await app.ainvoke(
            {"messages": [Message.text_message("Who am I? Topic: billing.")]},
            config={"thread_id": "inject-1", "user_id": "u-42", "plan": "pro"},
        )
        for message in result["messages"]:
            if message.role == "tool":
                print("tool result:", message.text())
        print("assistant:", result["messages"][-1].text())


if __name__ == "__main__":
    asyncio.run(main())
