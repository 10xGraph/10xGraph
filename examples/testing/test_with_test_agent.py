"""TestAgent: a drop-in replacement for Agent that returns canned responses.

Shows how to run a real StateGraph with no LLM call, then assert on what the
agent returned and on what it was sent.
Run: pytest examples/testing/test_with_test_agent.py -q
"""

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import Message
from tenxgraph.qa.testing import (
    TestAgent as CannedAgent,  # alias: pytest would try to collect "Test*"
)
from tenxgraph.utils.constants import END


def build_graph(agent: CannedAgent):
    graph = StateGraph()
    graph.add_node("MAIN", agent)
    graph.set_entry_point("MAIN")
    graph.add_edge("MAIN", END)
    return graph.compile()


def last_text(result: dict) -> str:
    # Message.text() joins the text blocks of a message.
    return result["messages"][-1].text()


async def test_returns_canned_response():
    agent = CannedAgent(responses=["Paris is the capital of France."])
    app = build_graph(agent)

    result = await app.ainvoke({"messages": [Message.text_message("Capital of France?")]})

    assert last_text(result) == "Paris is the capital of France."
    agent.assert_called_times(1)


async def test_responses_cycle_across_calls():
    # The list is reused in order, so one agent can script a whole conversation.
    agent = CannedAgent(responses=["first", "second"])
    app = build_graph(agent)

    first = await app.ainvoke({"messages": [Message.text_message("a")]})
    second = await app.ainvoke({"messages": [Message.text_message("b")]})

    assert last_text(first) == "first"
    assert last_text(second) == "second"
    assert agent.call_count == 2


async def test_agent_receives_system_prompt_and_user_message():
    agent = CannedAgent(
        system_prompt=[{"role": "system", "content": "You are terse."}],
        responses=["ok"],
    )
    app = build_graph(agent)

    await app.ainvoke({"messages": [Message.text_message("Hello there")]})

    # call_history records exactly what would have been sent to the model.
    sent = agent.get_last_messages()
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] == "You are terse."
    assert any(m["role"] == "user" and "Hello there" in str(m["content"]) for m in sent)


async def test_reset_clears_history():
    agent = CannedAgent(responses=["ok"])
    agent.assert_not_called()

    await build_graph(agent).ainvoke({"messages": [Message.text_message("hi")]})
    agent.assert_called()

    agent.reset()
    agent.assert_not_called()
    assert agent.call_history == []


def test_assertion_helpers_fail_loudly():
    agent = CannedAgent()
    with pytest.raises(AssertionError):
        agent.assert_called()
