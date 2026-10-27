"""QuickEval: evaluate the weather agent in a few lines.

QuickEval builds the dataset, picks a preset EvalConfig, runs the evaluator, and
prints a console report. Good for a first check before writing a full suite.
Run: python examples/evaluation/quick_eval.py   (needs OPENAI_API_KEY)
"""

import asyncio

from dotenv import load_dotenv
from weather_agent import build_eval_app

from tenxgraph.qa.evaluation import EvalPresets, EvalSetBuilder, QuickEval, ToolCall


async def main() -> None:
    load_dotenv()
    app, collector = build_eval_app()

    # 1. One query, one check on the reply text (ROUGE-1, no judge model). Passing
    #    expected_tools here would also enable an LLM response_match, whose default
    #    judge is gemini-2.5-flash, so tool checks live in step 3 instead.
    await QuickEval.check(
        app,
        collector,
        query="What is the weather in London?",
        expected_response_contains="The weather in London is sunny",
    )

    # 2. Many (query, expected response) pairs, scored with ROUGE-1 (no judge model).
    await QuickEval.batch(
        app,
        collector,
        [
            ("What is the weather in Paris?", "The weather in Paris is sunny"),
            ("What is the weather in Rome?", "The weather in Rome is sunny"),
        ],
        threshold=0.5,
    )

    # 3. Tool usage only: (query, expected response, expected tools).
    await QuickEval.tool_usage(
        app,
        collector,
        [
            ("Give me a 3-day forecast for Oslo.", "forecast for Oslo", ["get_forecast"]),
            ("What is the capital of Spain?", "Madrid", []),
        ],
        strict=False,
    )

    # 4. A builder plus combined presets. Neither preset uses a judge model.
    builder = EvalSetBuilder("weather_builder").add_case(
        "What is the weather in Cairo?",
        "The weather in Cairo is sunny",
        expected_tools=[ToolCall(name="get_weather", args={"location": "Cairo"})],
    )
    config = EvalPresets.combine(EvalPresets.quick_check(), EvalPresets.tool_usage(strict=False))
    report = await QuickEval.from_builder(app, collector, builder, config=config)
    print(f"pass rate: {report.summary.pass_rate:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
