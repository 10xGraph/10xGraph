"""A StateGraph with no LLM: custom state, plain function nodes, routing and Command.

Shows the engine on its own, so it runs without any API key.

Run: python examples/graph_basics/state_graph_no_llm.py
"""

import asyncio

from pydantic import Field

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import Command
from tenxgraph.utils.constants import END, START, ResponseGranularity


class TicketState(AgentState):
    """AgentState plus your own fields. They are saved and returned like `context`."""

    intent: str = ""
    amount: float = 0.0
    visited: list[str] = Field(default_factory=list)


def intake(state: TicketState) -> TicketState:
    # Nodes receive the current state by parameter name. Mutate it and return it.
    text = state.context[-1].text().lower()
    state.visited = [*state.visited, "intake"]
    if "refund" in text:
        state.intent = "refund"
        state.amount = 250.0 if "large" in text else 40.0
    elif "?" in text:
        state.intent = "faq"
    else:
        state.intent = "other"
    return state


def route_by_intent(state: TicketState) -> str:
    # A condition function returns a key; the path_map turns it into a node name.
    return state.intent


def faq(state: TicketState) -> Message:
    # Returning a Message appends it to state.context (the `add_messages` reducer on
    # AgentState.context drops duplicates by message_id).
    state.visited = [*state.visited, "faq"]
    return Message.text_message("See the help center for that one.", role="assistant")


def refund(state: TicketState) -> Command:
    # Command couples a state update with an explicit next node, so the node itself
    # decides where to go instead of a separate conditional edge.
    state.visited = [*state.visited, "refund"]
    if state.amount > 100:
        state.context.append(Message.text_message("Large refund, sending to review.", "assistant"))
        return Command(update=state, goto="review")
    state.context.append(Message.text_message("Refund approved automatically.", "assistant"))
    return Command(update=state, goto=END)


def review(state: TicketState) -> TicketState:
    state.visited = [*state.visited, "review"]
    state.context.append(Message.text_message("A person will review it today.", "assistant"))
    return state


def fallback(state: TicketState) -> Message:
    state.visited = [*state.visited, "fallback"]
    return Message.text_message("Thanks, we will get back to you.", role="assistant")


graph = StateGraph[TicketState](TicketState())
graph.add_node("INTAKE", intake)
graph.add_node("FAQ", faq)
graph.add_node("REFUND", refund)
graph.add_node("review", review)
graph.add_node("FALLBACK", fallback)

graph.add_edge(START, "INTAKE")  # also sets the entry point
graph.add_conditional_edges(
    "INTAKE",
    route_by_intent,
    {"faq": "FAQ", "refund": "REFUND", "other": "FALLBACK"},
)
graph.add_edge("FAQ", END)
graph.add_edge("FALLBACK", END)
graph.add_edge("review", END)
# REFUND has no outgoing edge: its Command(goto=...) picks "review" or END at runtime.

app = graph.compile()


async def main() -> None:
    for i, text in enumerate(
        ["How do I reset my password?", "Refund please", "Large refund please"]
    ):
        config = {"thread_id": f"ticket-{i}"}
        inp = {"messages": [Message.text_message(text)]}

        # LOW (default) returns messages only.
        low = await app.ainvoke(inp, config)
        print(f"\nuser: {text}")
        for msg in low["messages"]:
            print(f"  {msg.role}: {msg.text()}")

        # FULL also returns the final state, including the custom fields.
        full = await app.ainvoke(
            inp, {"thread_id": f"ticket-full-{i}"}, response_granularity=ResponseGranularity.FULL
        )
        state = full["state"]
        print(f"  path={state.visited} intent={state.intent} amount={state.amount}")

    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
