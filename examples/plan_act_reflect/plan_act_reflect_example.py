"""PlanActReflectAgent — runnable example.

Demonstrates three scenarios:
  1. Basic: single-pass research task without tools (PLAN → REFLECT → [DONE]).
  2. With tools: multi-step research using a web-search helper tool.
  3. Custom prompts: override the default plan / reflect system prompts.

Run with a valid OPENAI_API_KEY (or any OpenAI-compatible key):

    python examples/plan_act_reflect/plan_act_reflect_example.py
"""

import asyncio

from dotenv import load_dotenv

from tenxgraph.prebuilt.agent import PlanActReflectAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()

# ---------------------------------------------------------------------------
# Fake tool for offline demonstration (replace with a real search client)
# ---------------------------------------------------------------------------


def fake_web_search(query: str) -> str:
    """Return a canned research snippet — replace with a real implementation."""
    return (
        f"[Search result for '{query}']\n"
        "AI is transforming climate science through improved weather prediction "
        "models, satellite data processing, and carbon-capture optimisation. "
        "Recent studies show a 35 % accuracy improvement in 10-day forecasts."
    )


# ---------------------------------------------------------------------------
# Example 1 — No tools (pure reasoning loop)
# ---------------------------------------------------------------------------


async def example_no_tools() -> None:
    print("\n" + "=" * 60)
    print("Example 1: No-tool reasoning loop")
    print("=" * 60)

    agent = PlanActReflectAgent(
        model="gpt-4o-mini",
        max_iterations=2,
        provider="openai",
    )

    app = agent.compile(checkpointer=InMemoryCheckpointer())

    result = await app.ainvoke(
        {"message": "Explain the three laws of thermodynamics in simple terms."},
        config={"thread_id": "thermo-1"},
    )

    last = result["context"][-1]
    print(f"Final answer:\n{last.text()[:500]}")


# ---------------------------------------------------------------------------
# Example 2 — With tools (PLAN calls tool → ACT executes → REFLECT evaluates)
# ---------------------------------------------------------------------------


async def example_with_tools() -> None:
    print("\n" + "=" * 60)
    print("Example 2: Research loop with a web-search tool")
    print("=" * 60)

    agent = PlanActReflectAgent(
        model="gpt-4o-mini",
        tools=[fake_web_search],
        max_iterations=3,
        provider="openai",
    )

    app = agent.compile(checkpointer=InMemoryCheckpointer())

    result = await app.ainvoke(
        {"message": "Research the impact of AI on climate science."},
        config={"thread_id": "climate-research-1"},
    )

    last = result["context"][-1]
    print(f"Final answer:\n{last.text()[:500]}")


# ---------------------------------------------------------------------------
# Example 3 — Custom system prompts
# ---------------------------------------------------------------------------


CUSTOM_PLAN_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a Socratic philosopher. Break every question into smaller "
            "sub-questions, then reason through each one systematically."
        ),
    }
]

CUSTOM_REFLECT_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a peer reviewer. Evaluate whether the analysis is thorough. "
            "If it is complete, summarise the key insights and end with [DONE]. "
            "If gaps remain, list them explicitly."
        ),
    }
]


async def example_custom_prompts() -> None:
    print("\n" + "=" * 60)
    print("Example 3: Custom plan / reflect prompts")
    print("=" * 60)

    agent = PlanActReflectAgent(
        model="gpt-4o-mini",
        plan_system_prompt=CUSTOM_PLAN_PROMPT,
        reflect_system_prompt=CUSTOM_REFLECT_PROMPT,
        max_iterations=2,
        provider="openai",
    )

    app = agent.compile(checkpointer=InMemoryCheckpointer())

    result = await app.ainvoke(
        {"message": "Is it ethical to use AI in hiring decisions?"},
        config={"thread_id": "ethics-1"},
    )

    last = result["context"][-1]
    print(f"Final answer:\n{last.text()[:500]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    await example_no_tools()
    await example_with_tools()
    await example_custom_prompts()


if __name__ == "__main__":
    asyncio.run(main())
