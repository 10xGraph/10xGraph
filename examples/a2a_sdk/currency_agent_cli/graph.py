"""
Currency agent graph

Uses the Frankfurter API (free, no key needed) for live exchange rates.
The compiled graph is exposed as ``app`` — referenced in agentflow.json as
``"agent": "graph:app"``.
"""

from __future__ import annotations

import httpx
from dotenv import load_dotenv
from litellm import acompletion

from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState
from tenxgraph.utils.constants import END
from tenxgraph.utils.converter import convert_messages

load_dotenv()

# --------------------------------------------------------------------------- #
#  Tool — Frankfurter API                                                       #
# --------------------------------------------------------------------------- #


async def get_exchange_rate(
    currency_from: str,
    currency_to: str,
    currency_date: str = "latest",
    amount: float = 1.0,
) -> dict:
    """Get exchange rate between two currencies using the Frankfurter API.

    Args:
        currency_from: Source currency code (e.g. USD).
        currency_to:   Target currency code (e.g. INR).
        currency_date: Date in YYYY-MM-DD format or 'latest'.
        amount:        Amount to convert.

    Returns:
        dict with keys: amount, base, date, rates.
    """
    url = f"https://api.frankfurter.app/{currency_date}"
    params = {"from": currency_from, "to": currency_to, "amount": amount}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


tool_node = ToolNode([get_exchange_rate])

# --------------------------------------------------------------------------- #
#  LLM node                                                                     #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are a helpful currency conversion assistant. "
    "Use the get_exchange_rate tool to look up live exchange rates "
    "from the Frankfurter API. Always tell the user the converted "
    "amount and the date of the rate."
    "you can talk other than currency convo for genral queries like 2 + 2 or remembering name  but for currency conversion you must use the tool"
)


async def llm_node(state: AgentState):
    messages = convert_messages(
        system_prompts=[{"role": "system", "content": SYSTEM_PROMPT}],
        state=state,
    )

    # If last message is a tool result, summarise without tools
    if state.context and state.context[-1].role == "tool":
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
        )
    else:
        tools = await tool_node.all_tools()
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            tools=tools,
        )

    return ModelResponseConverter(response, converter="litellm")


# --------------------------------------------------------------------------- #
#  Routing                                                                      #
# --------------------------------------------------------------------------- #


def should_use_tools(state: AgentState) -> str:
    if not state.context:
        return END

    last = state.context[-1]

    if hasattr(last, "tools_calls") and last.tools_calls and last.role == "assistant":
        return "TOOL"

    if last.role == "tool":
        return "MAIN"

    return END


# --------------------------------------------------------------------------- #
#  Graph                                                                        #
# --------------------------------------------------------------------------- #

graph = StateGraph()
graph.add_node("MAIN", llm_node)
graph.add_node("TOOL", tool_node)
graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", "MAIN": "MAIN", END: END})
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile(checkpointer=InMemoryCheckpointer[AgentState]())
