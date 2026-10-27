"""Stop a runaway graph with recursion_limit and catch GraphRecursionError.

No LLM is called: the graph is two plain nodes that route back and forth forever,
which is what a confused agent loop looks like. Also shows the library exception
hierarchy. Run: python examples/resilience/recursion_limit.py
"""

import asyncio

from tenxgraph.core.exceptions import GraphError, GraphRecursionError
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import END


class LoopState(AgentState):
    attempts: int = 0


def think(state: LoopState) -> LoopState:
    state.attempts += 1
    return state


def act(state: LoopState) -> Message:
    return Message.text_message(f"attempt {state.attempts}: not done yet", role="assistant")


def route(state: LoopState) -> str:
    # A real agent would stop when the task is done. This one never is.
    return "THINK"


def build_graph():
    graph = StateGraph(LoopState())
    graph.add_node("THINK", think)
    graph.add_node("ACT", act)
    graph.add_edge("THINK", "ACT")
    graph.add_conditional_edges("ACT", route, {"THINK": "THINK", END: END})
    graph.set_entry_point("THINK")
    return graph.compile(checkpointer=InMemoryCheckpointer())


async def main() -> None:
    app = build_graph()
    inputs = {"messages": [Message.text_message("solve it")]}

    # recursion_limit counts node steps (default 25). Lower it for cheap, fast failure.
    config = {"thread_id": "loop-1", "recursion_limit": 6}
    try:
        await app.ainvoke(inputs, config)
    except GraphRecursionError as exc:
        print("stopped by recursion limit")
        print("  code:   ", exc.error_code)  # RECURSION_001
        print("  context:", exc.context)  # max_steps, current_step, current_node
        print("  dict:   ", exc.to_dict())  # structured form, safe to return from an API

    # Every library error derives from GraphError, so one handler can catch them all.
    try:
        await app.ainvoke(inputs, {"thread_id": "loop-2", "recursion_limit": 4})
    except GraphError as exc:
        print("caught as GraphError:", type(exc).__name__, "-", exc.message)

    # The checkpointer still holds the work done before the stop.
    saved = await app._checkpointer.aget_state({"thread_id": "loop-1", "user_id": "anonymous"})
    print("attempts recorded before the stop:", saved.attempts if saved else None)

    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
