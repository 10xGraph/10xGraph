"""Customer support swarm with approval for large refunds and saved memory.

Triage (Gemini), billing (OpenAI) and technical support (Anthropic) hand the
customer off to each other. Refunds above the threshold pause the run and ask
you to approve in the terminal. State persists in SQLite, memory in local Qdrant.

Run from the repo root: python examples/use_cases/customer_support/main.py
"""

from __future__ import annotations

import asyncio

from agents import build_memory_store, build_swarm
from config import CHECKPOINT_DB
from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.storage.checkpointer import SqliteCheckpointer
from tenxgraph.utils import ResponseGranularity, pending_interrupt


load_dotenv()


def ask_reviewer(request) -> dict:
    """Show the pending approval to a person and return the resume value."""
    print(f"\n[approval needed] {request.message}")
    print(f"  details: {request.value}")
    answer = input("  approve? [y/N] ").strip().lower()
    note = input("  note for the customer (optional): ").strip()
    return {"approved": answer == "y", "note": note}


async def send(app, text: str, config: dict) -> None:
    """Send one customer message and keep resuming while the graph asks for approval."""
    result = await app.ainvoke(
        {"messages": [Message.text_message(text)]},
        config,
        response_granularity=ResponseGranularity.FULL,
    )
    # A paused thread has a pending interrupt on its state. Resume the same
    # thread with the answer; the refund tool re-runs and interrupt() returns it.
    while (request := pending_interrupt(result["state"])) is not None:
        result = await app.ainvoke(
            {"resume": ask_reviewer(request), "resume_interrupt_id": request.id},
            config,
            response_granularity=ResponseGranularity.FULL,
        )

    replies = [m for m in result["messages"] if m.role == "assistant" and m.text()]
    print("assistant:", replies[-1].text() if replies else "(no reply)")


async def main() -> None:
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    store = build_memory_store()
    app = build_swarm(store).compile(
        checkpointer=SqliteCheckpointer(CHECKPOINT_DB),
        store=store,
    )

    customer_id = input("Customer id [C-1001]: ").strip() or "C-1001"
    # The same thread_id on a later run continues this conversation from SQLite.
    config = {"thread_id": f"support-{customer_id}", "user_id": customer_id}
    print("Type a message, or 'quit' to exit.")

    try:
        while (text := input("\nyou: ").strip()) not in {"quit", "exit"}:
            if text:
                await send(app, text, config)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
