"""SupervisorTeamAgent example — supervisor routes tasks to specialist workers.

The supervisor LLM analyses each message and picks the best worker, or signals
FINISH when the task is complete.  Workers are independently configured Agents.

Run::

    OPENAI_API_KEY=sk-... python examples/supervisor_team/supervisor_team_example.py
"""

from __future__ import annotations

import asyncio

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.prebuilt.agent.supervisor_team import SupervisorTeamAgent, WorkerConfig


# ---------------------------------------------------------------------------
# Domain tools
# ---------------------------------------------------------------------------


def web_search(query: str) -> str:
    """Search the web and return relevant results."""
    return f"[web_search] Results for '{query}': many relevant findings."


def run_python(code: str) -> str:
    """Execute Python code and return the output."""
    return f"[run_python] Output:\n{code[:80]}... (executed)"


def write_report(title: str, content: str) -> str:
    """Format and write a final report."""
    return f"[write_report] Report '{title}':\n{content[:200]}"


# ---------------------------------------------------------------------------
# Scenario 1 — 3-worker research + code + writing team
# ---------------------------------------------------------------------------

RESEARCHER_SYSTEM = [
    {
        "role": "system",
        "content": (
            "You are a research specialist. Use web_search to find relevant, "
            "accurate information. Summarize key findings concisely."
        ),
    }
]

CODER_SYSTEM = [
    {
        "role": "system",
        "content": (
            "You are a Python expert. Write clean, well-documented code. "
            "Use run_python to test your solutions."
        ),
    }
]

WRITER_SYSTEM = [
    {
        "role": "system",
        "content": (
            "You are a technical writer. Produce clear, well-structured reports "
            "from research and code outputs. Use write_report to finalize."
        ),
    }
]


def build_research_team() -> SupervisorTeamAgent:
    return SupervisorTeamAgent(
        supervisor_model="gpt-4o",
        workers={
            "RESEARCHER": WorkerConfig(
                agent=Agent(
                    model="gpt-4o-mini",
                    system_prompt=RESEARCHER_SYSTEM,
                    tool_node=ToolNode([web_search]),
                ),
                description=(
                    "Searches the web for factual information on any topic."
                ),
            ),
            "CODER": WorkerConfig(
                agent=Agent(
                    model="gpt-4o",         # higher-capability model for coding
                    system_prompt=CODER_SYSTEM,
                    tool_node=ToolNode([run_python]),
                    # Could add: retry_config=..., memory=MemoryConfig(...)
                ),
                description=(
                    "Writes and executes Python code to solve computational problems."
                ),
            ),
            "WRITER": WorkerConfig(
                agent=Agent(
                    model="gpt-4o-mini",
                    system_prompt=WRITER_SYSTEM,
                    tool_node=ToolNode([write_report]),
                ),
                description=(
                    "Writes polished technical reports from research and code outputs."
                ),
            ),
        },
        max_rounds=10,
    )


# ---------------------------------------------------------------------------
# Scenario 2 — 2-worker analyst + communicator with custom supervisor prompt
# ---------------------------------------------------------------------------

CUSTOM_SUPERVISOR_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are the project manager of a 2-person team.\n"
            "Workers:\n"
            "- ANALYST: analyses data and produces insights.\n"
            "- WRITER: writes the final deliverable.\n"
            "- FINISH: the deliverable is complete.\n\n"
            "Respond with exactly one word: ANALYST, WRITER, or FINISH."
        ),
    }
]


def build_analyst_writer_team() -> SupervisorTeamAgent:
    return SupervisorTeamAgent(
        supervisor_model="gpt-4o-mini",
        workers={
            "ANALYST": WorkerConfig(
                agent=Agent(
                    model="gpt-4o-mini",
                    system_prompt=[
                        {
                            "role": "system",
                            "content": "Analyse data and produce structured insights.",
                        }
                    ],
                    tool_node=ToolNode([web_search]),
                ),
                description="Analyses data and produces structured insights.",
            ),
            "WRITER": WorkerConfig(
                agent=Agent(
                    model="gpt-4o-mini",
                    system_prompt=[
                        {
                            "role": "system",
                            "content": "Write the final deliverable from analyst outputs.",
                        }
                    ],
                    tool_node=ToolNode([write_report]),
                ),
                description="Writes the final deliverable from analyst outputs.",
            ),
        },
        supervisor_system_prompt=CUSTOM_SUPERVISOR_PROMPT,
        max_rounds=6,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_scenario_1() -> None:
    print("\n--- Scenario 1: Research + Code + Writing Team ---")
    team = build_research_team()
    app = team.compile()

    result = await app.ainvoke(
        {
            "message": (
                "Research the top 3 Python libraries for data visualization, "
                "write a short demo script for each, and produce a report."
            )
        },
        config={"thread_id": "supervisor-1"},
    )
    print("Final messages:")
    for msg in result.get("context", []):
        print(f"  [{msg.get('role', '?')}] {str(msg.get('content', ''))[:120]}")


async def run_scenario_2() -> None:
    print("\n--- Scenario 2: Analyst + Writer Team (custom supervisor prompt) ---")
    team = build_analyst_writer_team()
    app = team.compile()

    result = await app.ainvoke(
        {"message": "Analyse the latest trends in renewable energy and write a summary."},
        config={"thread_id": "supervisor-2"},
    )
    print("Final messages:")
    for msg in result.get("context", []):
        print(f"  [{msg.get('role', '?')}] {str(msg.get('content', ''))[:120]}")


async def main() -> None:
    await run_scenario_1()
    await run_scenario_2()


if __name__ == "__main__":
    asyncio.run(main())
