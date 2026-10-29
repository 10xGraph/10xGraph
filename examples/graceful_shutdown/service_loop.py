"""A long-running agent service that shuts down cleanly on SIGINT and SIGTERM.

Shows GracefulShutdownManager, a delayed_keyboard_interrupt critical section,
compile(shutdown_timeout=...) and the stats dict returned by graph.aclose().
Run: python examples/graceful_shutdown/service_loop.py   (then press Ctrl+C)
"""

import asyncio
import contextlib
import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.shutdown import GracefulShutdownManager, delayed_keyboard_interrupt


load_dotenv()

LEDGER = Path(tempfile.gettempdir()) / "10xgraph_shutdown_ledger.jsonl"
JOBS = [
    "Where is order A-1001?",
    "Where is order B-2002?",
    "Where is order C-3003?",
    "Where is order D-4004?",
]


def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order."""
    return f"Order {order_id} left the warehouse and arrives in 2 days."


def commit_result(job: str, answer: str) -> None:
    """Append one line to the ledger. A half-written line would corrupt the file."""
    # delayed_keyboard_interrupt() swaps in handlers that only record the signal.
    # If Ctrl+C or SIGTERM lands inside this block it is held back until the block
    # exits, so the write and fsync either finish or never start.
    with delayed_keyboard_interrupt(), LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"job": job, "answer": answer}) + "\n")
        f.flush()
        os.fsync(f.fileno())


async def main() -> None:
    manager = GracefulShutdownManager(shutdown_timeout=20.0)
    # Installs asyncio signal handlers for SIGINT and SIGTERM. They only set
    # manager.shutdown_requested, so the loop below decides when to stop.
    manager.register_signal_handlers()

    agent = ReactAgent(
        model="gemini-2.5-flash",
        system_prompt=[{"role": "system", "content": "You are an order-support assistant."}],
        tools=[get_order_status],
    )
    # shutdown_timeout sets the aclose() defaults: 50% grace, 25% stop, 25% cancel and drain.
    graph = agent.compile(checkpointer=InMemoryCheckpointer(), shutdown_timeout=20.0)

    print("Service running. Press Ctrl+C (or send SIGTERM) to stop.")
    try:
        for number, job in enumerate(JOBS, start=1):
            if manager.shutdown_requested:
                break  # stop taking new jobs; the last one was already committed
            result = await graph.ainvoke(
                {"messages": [Message.text_message(job)]},
                {"thread_id": f"job-{number}", "user_id": "support-bot"},
            )
            answer = result["messages"][-1].text()
            commit_result(job, answer)
            print(f"[job {number}] {job} -> {answer}")

            # Idle between jobs, but wake up the moment a signal arrives.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(manager.wait_for_shutdown(), timeout=3.0)
    finally:
        print("Shutting down...")
        # New runs are refused. Active runs get 10s to finish, then 5s after a stop (each
        # saves its thread INTERRUPTED), then the rest are cancelled; background writes are
        # drained and the checkpointer, publisher and store are released.
        stats = await graph.aclose(grace=10.0, stop_timeout=5.0, timeout=5.0)
        print(json.dumps(stats, indent=2, default=str))
        manager.unregister_signal_handlers()


if __name__ == "__main__":
    asyncio.run(main())
