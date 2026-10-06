import random

from dotenv import load_dotenv

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils.constants import END


load_dotenv()


def main_agent(
    state: AgentState,
    config: dict,
):
    is_end = config.get("trail", 2)
    if is_end == 0:
        return Message.text_message("CV", role="assistant")
    elif is_end == 1:
        return Message.text_message("JD", role="assistant")
    else:
        return Message.text_message("Thank you for contacting me", role="assistant")


def cv_agent(
    state: AgentState,
    config: dict | None = None,
):
    return Message.text_message("CV Created Successfully", role="assistant")


def jd_agent(
    state: AgentState,
    config: dict | None = None,
):
    return Message.text_message("JD Created Successfully", role="assistant")


def should_use_tools(state: AgentState) -> str:
    """Determine if we should use tools or end the conversation."""
    if not state.context or len(state.context) == 0:
        return "TOOL"  # No context, might need tools

    last_message = state.context[-1]
    msg = last_message.text()
    if "cv" in msg.lower():
        return "CV"
    if "jd" in msg.lower():
        return "JD"

    return END


graph = StateGraph()
graph.add_node("MAIN", main_agent)
graph.add_node("CV", cv_agent)
graph.add_node("JD", jd_agent)

# Add conditional edges from MAIN
graph.add_conditional_edges(
    "MAIN",
    should_use_tools,
    {"CV": "CV", "JD": "JD", END: END},
)

# Always go back to MAIN after CV or JD execution
graph.add_edge("CV", END)
graph.add_edge("JD", END)
graph.set_entry_point("MAIN")


app = graph.compile()


# now run it
if __name__ == "__main__":
    # First message from user
    config = {"thread_id": "12345", "recursion_limit": 10, "trail": 0}
    inp = {"messages": [Message.text_message("HI")]}
    res = app.invoke(inp, config=config)
    print("**********************")
    print("First Message")
    for i in res["messages"]:
        print("Message Type: ", i.role)
        print(i)
        print("\n\n")

    print("**********************")

    inp = {"messages": [Message.text_message("HI")]}
    config = {"thread_id": "12345", "recursion_limit": 10, "trail": 1}
    res = app.invoke(inp, config=config)
    print("**********************")
    print("Second Message")
    for i in res["messages"]:
        print("Message Type: ", i.role)
        print(i)
        print("\n\n")

    print("**********************")

    inp = {"messages": [Message.text_message("HI")]}
    config = {"thread_id": "12345", "recursion_limit": 10, "trail": 2}
    res = app.invoke(inp, config=config)
    print("**********************")
    print("Third Message")
    for i in res["messages"]:
        print("Message Type: ", i.role)
        print(i)
        print("\n\n")

    print("**********************")
