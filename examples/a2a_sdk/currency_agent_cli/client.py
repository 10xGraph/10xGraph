"""
Client for the CurrencyAgent A2A server.

Usage::

    python examples/a2a_sdk/currency_agent_cli/client.py
"""

from __future__ import annotations

import asyncio
import uuid

from tenxgraph.runtime.protocols import delegate_to_a2a_agent


SERVER_URL = "http://localhost:10000"

# One context_id per session — ties all turns together so the agent
# retains conversation history across multiple messages.
SESSION_CONTEXT_ID = str(uuid.uuid4())


async def main() -> None:
    print("CurrencyAgent client — type 'quit' to exit")
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

        try:
            reply = await delegate_to_a2a_agent(
                SERVER_URL, user_input, context_id=SESSION_CONTEXT_ID
            )
            print(f"Agent: {reply}\n")
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
