"""Entry point for the `10xgraph eval` command.

The CLI discovers files named `*_eval.py` or `eval_*.py`, takes the module-level
`app` (a compiled graph), `get_eval_set()` and `EVAL_CONFIG`, then writes console,
JSON and HTML reports. Run from the repo root: 10xgraph eval examples/evaluation/eval_weather.py
(needs OPENAI_API_KEY)
"""

from pathlib import Path

from dotenv import load_dotenv
from weather_agent import build_agent

from tenxgraph.qa.evaluation import (
    CriteriaConfig,
    CriterionConfig,
    EvalConfig,
    EvalSet,
    MatchType,
)


load_dotenv()

# The CLI attaches its own trajectory collector, so compile without one here.
app = build_agent().compile()

EVAL_CONFIG = EvalConfig(
    criteria=CriteriaConfig(
        tool_name_match=CriterionConfig.tool_name_match(threshold=1.0),
        trajectory=CriterionConfig.trajectory(threshold=1.0, match_type=MatchType.IN_ORDER),
    )
)


def get_eval_set() -> EvalSet:
    return EvalSet.from_file(str(Path(__file__).parent / "weather.evalset.json"))
