"""
Client2 — uses create_a2a_client_node inside an agentflow graph.

The CurrencyAgent server (graph.py served via `agentflow a2a`) is a node
in this client graph. thread_id flows as context_id automatically.

Usage::

    # 1. Start the server first:
    #    agentflow a2a --port 10000  (from currency_agent_cli/ dir)
    # 2. Then run this client:
    python examples/a2a_sdk/currency_agent_cli/client2.py
"""

from __future__ import annotations

import asyncio

from tenxgraph.a2a_integration.client import create_a2a_client_node
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState
from tenxgraph.core.state.message import Message as AFMessage
from tenxgraph.utils.constants import END

SERVER_URL = "http://localhost:10000"

# --------------------------------------------------------------------------- #
#  Client graph — single node that delegates to the remote A2A server         #
# --------------------------------------------------------------------------- #

checkpointer = InMemoryCheckpointer[AgentState]()

graph = StateGraph()
graph.add_node("currency", create_a2a_client_node(SERVER_URL))
graph.add_edge("currency", END)
graph.set_entry_point("currency")

app = graph.compile(checkpointer=checkpointer)

# --------------------------------------------------------------------------- #
#  CLI                                                                          #
# --------------------------------------------------------------------------- #

THREAD_ID = "currency-session-1"  # fixed = persistent history across turns


async def main() -> None:
    print("CurrencyAgent client2 (graph mode) — type 'quit' to exit")
    print(f"Server: {SERVER_URL}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input or user_input.lower() == "quit":
            print("Bye!")
            break

        result = await app.ainvoke(
            {"messages": [AFMessage.text_message(user_input, role="user")]},
            config={"thread_id": THREAD_ID},
        )

        # last message in the result is the agent reply
        messages = result.get("messages", [])
        last = messages[-1] if messages else None
        reply = last.text() if last else str(result)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
