"""Graph lifecycle hooks: react to the start, end, interrupts and checkpoints of a run.

GraphLifecycleHook fires at structural events of the whole graph run, not around
single model or tool calls. Override only the methods you need; the rest are no-ops.

Run: python examples/callbacks/lifecycle_hooks.py   (needs GEMINI_API_KEY)
"""

import asyncio
import time

from dotenv import load_dotenv

from tenxgraph.core.state import AgentState, Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.callbacks import CallbackManager, GraphLifecycleContext, GraphLifecycleHook


load_dotenv()


def lookup_order(order_id: str) -> str:
    """Look up the status of an order."""
    return f"Order {order_id} shipped yesterday."


class AuditHook(GraphLifecycleHook[AgentState]):
    """Prints a one-line audit trail for each lifecycle event."""

    def __init__(self) -> None:
        self._started: dict[str, float] = {}

    async def on_graph_start(self, context: GraphLifecycleContext, state: AgentState):
        self._started[context.run_id] = time.perf_counter()
        print(f"[start]      thread={context.thread_id} messages={len(state.context)}")
        return None  # Return a new state here to rewrite it before the run begins.

    async def on_state_update(self, context, node_name, old_state, new_state, step):
        print(f"[state]      step {step}: node {node_name} -> {len(new_state.context)} messages")
        return None

    async def on_checkpoint(self, context, state, messages, is_context_trimmed):
        print(f"[checkpoint] saving {len(messages)} new messages")
        return None

    async def on_graph_end(self, context, final_state, messages, total_steps):
        started = self._started.pop(context.run_id, time.perf_counter())
        print(f"[end]        {total_steps} steps in {time.perf_counter() - started:.2f}s")
        return None

    async def on_graph_error(self, context, error, partial_state, messages, step, node_name):
        print(f"[error]      step {step} in {node_name}: {error}")
        return None  # The exception is always re-raised after the hook runs.

    # Fired when a run pauses at interrupt_before / interrupt_after / interrupt(),
    # and when it resumes. Not triggered in this run, shown for reference.
    async def on_interrupt(self, context, interrupted_node, interrupt_type, state):
        print(f"[interrupt]  paused at {interrupted_node} ({interrupt_type})")
        return None

    async def on_resume(self, context, resumed_node, state, resume_data):
        print(f"[resume]     continuing at {resumed_node}")
        return None

    # Realtime (voice) sessions only: one turn is one model generation.
    async def on_turn_start(self, context, state, turn_index):
        print(f"[turn {turn_index}] start")
        return None

    async def on_turn_end(self, context, state, turn_index):
        print(f"[turn {turn_index}] end")
        return None


async def main() -> None:
    manager = CallbackManager()
    manager.register_lifecycle_hook(AuditHook())

    agent = ReactAgent(model="gemini-2.5-flash", tools=[lookup_order])
    # A checkpointer makes the on_checkpoint hook fire on every durable write.
    app = agent.compile(checkpointer=InMemoryCheckpointer(), callback_manager=manager)

    result = await app.ainvoke(
        {"messages": [Message.text_message("Where is order A-1042?")]},
        config={"thread_id": "audit-1"},
    )
    print("assistant:", result["messages"][-1].text())
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
