"""Plan -> Act -> Reflect -> Report graph with a different company's model per phase.

`PlanActReflectAgent` runs tools in a plain `ToolNode` and lets you set only the
plan and reflect models, so its ACT phase has no model of its own. To give ACT
its own model (OpenAI) and add a report phase (DeepSeek), this file wires the
same loop by hand with `StateGraph`.
"""

from __future__ import annotations

import os

import prompts
from tools import RESEARCH_TOOLS

from tenxgraph.core.graph import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import END


MAX_ROUNDS = 2  # plan/act/reflect passes before the report is written anyway


class ResearchState(AgentState):
    rounds: int = 0


def stage(agent: Agent, task: str):
    """Run `agent` on a text-only view of the conversation plus a closing instruction.

    Tool calls and tool output are hidden from the view: they are long, and some
    providers reject a tool result they did not request. The real state is
    restored afterwards, so later phases still see everything.
    """

    async def node(state: ResearchState, config: dict):
        original = state.context
        visible = [m for m in original if m.role != "tool" and not m.tools_calls]
        state.context = [*visible, Message.text_message(task)]
        try:
            return await agent.execute(state, config)
        finally:
            state.context = original

    return node


def after_act(state: ResearchState) -> str:
    last = state.context[-1]
    return "TOOL" if last.role == "assistant" and last.tools_calls else "REFLECT"


def after_reflect(state: ResearchState) -> str:
    done = "[done]" in state.context[-1].text().lower()
    return "REPORT" if done or state.rounds + 1 >= MAX_ROUNDS else "NEXT_ROUND"


def next_round(state: ResearchState) -> list[Message]:
    state.rounds += 1
    return []


def build_graph():
    tool_node = ToolNode(RESEARCH_TOOLS)

    planner = Agent(model="claude-sonnet-5-5", system_prompt=prompts.PLAN)
    actor = Agent(model="gpt-5-mini", system_prompt=prompts.ACT, tool_node=tool_node)
    reflector = Agent(model="gemini-2.5-flash", system_prompt=prompts.REFLECT)
    writer = Agent(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        system_prompt=prompts.REPORT,
        reasoning_config=None,  # plain chat model; no reasoning parameters
    )

    graph = StateGraph(state=ResearchState())
    graph.add_node("PLAN", stage(planner, prompts.PLAN_TASK))
    graph.add_node("ACT", actor)
    graph.add_node("TOOL", tool_node)
    graph.add_node("REFLECT", stage(reflector, prompts.REFLECT_TASK))
    graph.add_node("NEXT_ROUND", next_round)
    graph.add_node("REPORT", stage(writer, prompts.REPORT_TASK))

    graph.set_entry_point("PLAN")
    graph.add_edge("PLAN", "ACT")
    graph.add_conditional_edges("ACT", after_act, {"TOOL": "TOOL", "REFLECT": "REFLECT"})
    graph.add_edge("TOOL", "ACT")
    graph.add_conditional_edges(
        "REFLECT", after_reflect, {"REPORT": "REPORT", "NEXT_ROUND": "NEXT_ROUND"}
    )
    graph.add_edge("NEXT_ROUND", "PLAN")
    graph.add_edge("REPORT", END)
    return graph.compile()
