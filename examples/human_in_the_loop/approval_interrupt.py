"""Ask a person to approve a refund from inside a tool, then resume the same thread.

issue_refund() calls interrupt() with a response_schema. The run pauses and is saved
by the checkpointer. We read the pending interrupt, then resume once with an
approval and once with a rejection. Run: python examples/human_in_the_loop/approval_interrupt.py
"""

import asyncio
from typing import Any

from dotenv import load_dotenv

from tenxgraph.core.exceptions import ResumeValidationError
from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity, interrupt, pending_interrupt


load_dotenv()

DECISION_SCHEMA = {
    "type": "object",
    "properties": {"approved": {"type": "boolean"}, "note": {"type": "string"}},
    "required": ["approved"],
}


def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order. A human must approve it first."""
    # On the first run interrupt() raises and the graph pauses here. On resume this
    # function runs again from the top and interrupt() returns the resume value, so
    # keep side effects AFTER the call.
    decision: dict[str, Any] | None = interrupt(
        {"order_id": order_id, "amount": amount},
        message=f"Approve a ${amount:.2f} refund for order {order_id}?",
        reason="tool_approval",
        response_schema=DECISION_SCHEMA,
    )
    if not decision or not decision.get("approved"):  # None means the client cancelled
        return f"Refund rejected. Reviewer note: {(decision or {}).get('note', 'none')}"
    return f"Refund of ${amount:.2f} issued for order {order_id}."


async def run_case(graph, thread_id: str, decision: dict[str, Any]) -> None:
    config = {"thread_id": thread_id, "user_id": "support-agent-1"}
    # FULL granularity returns the whole state, which holds the pending interrupt.
    result = await graph.ainvoke(
        {"messages": [Message.text_message("Refund $42.50 for order A-1001, it arrived broken.")]},
        config,
        response_granularity=ResponseGranularity.FULL,
    )
    request = pending_interrupt(result["state"])
    if request is None:
        print("no approval needed:", result["messages"][-1].text())
        return
    print(f"[{thread_id}] paused:", request.message)
    print(f"[{thread_id}] payload:", request.value, "| interrupt id:", request.id)

    # A wrong shape is rejected before anything is recorded; the thread stays paused.
    try:
        await graph.ainvoke({"resume": {"approved": "yes"}}, config)
    except ResumeValidationError as exc:
        print(f"[{thread_id}] bad resume refused:", exc.context["errors"])

    # Naming the interrupt id stops a stale or duplicated click from answering a later prompt.
    final = await graph.ainvoke(
        {"resume": decision, "resume_interrupt_id": request.id},
        config,
    )
    print(f"[{thread_id}] assistant:", final["messages"][-1].text())


async def main() -> None:
    agent = ReactAgent(
        model="gpt-5-mini",
        system_prompt=[
            {"role": "system", "content": "You handle refund requests. Use issue_refund."}
        ],
        tools=[issue_refund],
    )
    # interrupt() needs a checkpointer: the paused thread is stored there.
    graph = agent.compile(checkpointer=InMemoryCheckpointer())
    async with graph:
        await run_case(graph, "refund-approve", {"approved": True, "note": "photo received"})
        await run_case(graph, "refund-reject", {"approved": False, "note": "outside return window"})


if __name__ == "__main__":
    asyncio.run(main())
