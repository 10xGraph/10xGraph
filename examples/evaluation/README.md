---
title: Evaluate a 10xGraph Agent with Trajectory and LLM Judge Criteria
description: Evaluate a 10xGraph Python AI agent with pytest. Score tool trajectories, run LLM-as-judge criteria and user simulation, and write console, JSON, HTML and JUnit reports.
section: Examples
group: "Quality"
order: 230
label: Agent evaluation
updated: "2026-10-28"
faq:
  - question: How do I check that an agent called the right tools?
    answer: Build an EvalSet with expected_tools and enable the tool_name_match and trajectory criteria in an EvalConfig. AgentEvaluator records the real tool calls through a TrajectoryCollector and scores them against the expected list.
  - question: Which model judges the answers?
    answer: Each LLM criterion has a judge_model on its CriterionConfig. The default is gemini-2.5-flash, and you can pass another model name, such as claude-sonnet-5-5, to llm_judge, rubric_based, hallucination, safety or response_match.
  - question: Why does an LLM criterion score exactly 0.5?
    answer: When the judge model cannot be called, the criterion returns 0.5 with the reasoning "No LLM provider available" instead of raising. conftest.py has an assert_judged helper that turns that case into a test failure.
  - question: Can I run evaluations from the command line?
    answer: Yes. "10xgraph eval" discovers files named *_eval.py or eval_*.py, runs the app and eval set they expose, and writes console, JSON and HTML reports. See eval_weather.py.
---

# Evaluate a 10xGraph agent

## What you will build

An evaluation suite for a small weather assistant. The agent runs on OpenAI `gpt-5-mini`, with a `get_weather` and a `get_forecast` tool. The suite checks which tools it calls, how close its answers are to references, how a judge model rates them, and how it behaves in a conversation played by a simulated user.

Everything is a normal pytest file. Tests that call a live model are marked and skipped when the key is missing, so `pytest examples/evaluation` is safe to run without keys. The remaining tests run offline against a scripted agent.

## Key concepts

- `AgentEvaluator(graph, collector, config)`: runs each `EvalCase`, scores it against the enabled criteria and returns an `EvalReport`.
- `TrajectoryCollector` and `make_trajectory_callback`: capture tool calls and node visits during a run. Compile the graph with the callback manager.
- `EvalSet`, `EvalCase`, `EvalSetBuilder`: the dataset, with expected tools and expected responses. Sets save to and load from JSON.
- `EvalConfig` with `CriteriaConfig` and `CriterionConfig`: choose criteria and thresholds. `MatchType` sets trajectory strictness (`EXACT`, `IN_ORDER`, `ANY_ORDER`).
- Non-LLM criteria: `tool_name_match`, `trajectory`, `node_order`, `rouge_match`, `contains_keywords`.
- LLM criteria: `llm_judge`, `response_match`, `rubric_based` (with `Rubric`), `hallucination`, `safety`, plus `simulation_goals` for simulations.
- `UserSimulator`, `BatchSimulator`, `ConversationScenario`: a model plays the user against your graph.
- `QuickEval` and `EvalPresets`: shorter entry points with ready-made configs.
- Reporters: `ConsoleReporter`, `JSONReporter`, `HTMLReporter`, `JUnitXMLReporter`, and `ReporterManager`.
- Pytest helpers: `assert_eval_passed`, `assert_criterion_passed`, `run_eval`, `parametrize_eval_cases`, `create_eval_app`, `create_simple_eval_set`.

## Prerequisites

```bash
pip install "10xgraph[openai,google-genai,anthropic]" pytest pytest-asyncio python-dotenv
pip install 10xgraph-api       # only for the 10xgraph eval command
```

| Variable | Needed by |
|---|---|
| `OPENAI_API_KEY` | the agent under test (`gpt-5-mini`), all live tests and scripts |
| `ANTHROPIC_API_KEY` | the judge in `test_llm_judge.py` (`claude-sonnet-5-5`) |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | the simulated user in `test_user_simulator.py` (`gemini-2.5-flash`) |

Put them in a `.env` file at the repo root. `conftest.py` loads it.

## Run it

From the repository root:

```bash
pytest examples/evaluation -v -c examples/evaluation/pytest.ini     # live tests skip without keys
python examples/evaluation/reporters.py                             # no key needed
python examples/evaluation/quick_eval.py                            # needs OPENAI_API_KEY
10xgraph eval examples/evaluation/eval_weather.py --threshold 0.8   # needs OPENAI_API_KEY
```

| File | Shows |
|---|---|
| `weather_agent.py` | the agent under test and `build_eval_app()` |
| `conftest.py` | shared app fixture, key-based skipping, `assert_judged` |
| `test_trajectory.py` | datasets, tool trajectory and response criteria, pytest helpers |
| `test_llm_judge.py` | LLM criteria with `claude-sonnet-5-5` as the judge |
| `test_user_simulator.py` | `UserSimulator` on `gemini-2.5-flash` |
| `quick_eval.py` | `QuickEval` one-liners |
| `reporters.py` | all four reporters, with a hand-built report |
| `eval_weather.py` | the file shape that `10xgraph eval` discovers |
| `weather.evalset.json` | the dataset as JSON, used by the CLI and `run_eval` |

## How it works

**Three companies, one suite.** The agent under test uses OpenAI, the judge uses Anthropic and the simulated user uses Google. Keeping them separate means the judge is not grading its own output, and each model string is visible where it is used.

**Collecting the trajectory.** `weather_agent.py` compiles the agent with a collector so the evaluator can see tool calls:

```python
collector = TrajectoryCollector(capture_all_events=True)
_, callback_manager = make_trajectory_callback(collector)
app = build_agent().compile(
    checkpointer=InMemoryCheckpointer(),
    callback_manager=callback_manager,
)
```

If you already have an uncompiled `StateGraph`, `create_eval_app(graph)` does the same in one call.

**Datasets and criteria.** `test_trajectory.py` builds the set with `EvalSetBuilder`, then picks criteria:

```python
config = EvalConfig(
    criteria=CriteriaConfig(
        tool_name_match=CriterionConfig.tool_name_match(threshold=1.0),
        trajectory=CriterionConfig.trajectory(threshold=1.0, match_type=MatchType.IN_ORDER),
    ),
    reporter=ReporterConfig(enabled=False),   # reporters write files by default
)
report = await AgentEvaluator(app, collector, config=config).evaluate(eval_set)
assert_eval_passed(report)
```

An expected `ToolCall` with empty `args` accepts any arguments; give `args` and set `check_args=True` to compare them. `expected_tools=[]` asserts that no tool was called. The collector lists only the nodes that call a model, so the expected `node_order` for a tool run is `["__start__", "MAIN", "MAIN"]`, with no tool node in it.

The offline tests in the same file run the evaluator against a `TestAgent` graph, which checks the wiring without a key.

**The judge model.** `judge_model` is a string on every LLM criterion:

```python
CriterionConfig.llm_judge(threshold=0.6, judge_model="claude-sonnet-5-5", num_samples=1)
```

Each criterion samples the judge `num_samples` times and averages the scores.

**Known gap in the judge.** In this version of the library, `LLMCallerMixin._call_llm_json` in `tenxgraph/qa/evaluation/criteria/llm_utils.py` only has branches for Google and OpenAI models. `detect_provider("claude-sonnet-5-5")` returns `"anthropic"`, so an Anthropic judge falls through to the fallback: the criterion scores `0.5` with the reasoning "No LLM provider available" and does not raise. `assert_judged` in `conftest.py` fails the test in that case, so `test_llm_judge.py` will fail against an unpatched library until the Anthropic branch is added (it can call `call_llm`, which already supports Anthropic). `UserSimulator` and `SimulationGoalsCriterion` are not affected, because they call `call_llm` directly or use the Gemini default.

**Simulated users.** `UserSimulator(model="gemini-2.5-flash", criteria=[...])` writes the user turns, checks goals after each reply, and returns a `SimulationResult` with the conversation and `criterion_scores`. `SimulationGoalsCriterion` reads the whole transcript, so it works only inside a simulator.

**Reports.** `reporters.py` builds an `EvalReport` by hand and renders it four ways. With a real run, set `EvalConfig(reporter=ReporterConfig(...))` and `evaluate()` writes the files for you into `eval_reports/`.

**Command line.** `10xgraph eval` finds files named `*_eval.py` or `eval_*.py`. A file exposes `app` (a compiled graph), `get_eval_set()`, and optionally `EVAL_CONFIG`. For simulations, it exposes `SCENARIOS` instead. Useful flags are `--threshold`, `--parallel`, `--max-concurrency`, `--output` and `--open`. `quick_eval.py` also matches `*_eval.py` but has no eval entry point, so the CLI would skip it with a warning. Point the command at `eval_weather.py` instead of the whole folder.

## Expected output

Illustrative:

```text
$ pytest examples/evaluation -v -c examples/evaluation/pytest.ini
test_trajectory.py::test_eval_set_round_trips_through_json PASSED
test_trajectory.py::test_non_llm_criteria_on_scripted_agent PASSED
test_trajectory.py::test_tool_trajectory_for_weather_agent SKIPPED (OPENAI_API_KEY is not set)
...
7 passed, 12 skipped
```

With `OPENAI_API_KEY` set, `reporters.py` prints a report like this:

```text
Evaluation Report: Weather demo
  Total Cases:  2
  Passed:       1 (50.0%)
  tool_name_match_score: 1/2 passed, avg score: 0.50
  PASS NYC weather (2.40s)
  FAIL Capital, no tool (1.10s)
```

## Using a different model

Change the string where it is used: `model="gpt-5-mini"` in `weather_agent.py`, `judge_model=` in `test_llm_judge.py`, and `model=` in `UserSimulator`. Install the matching extra (`openai`, `google-genai`, `anthropic`) and set its key. Swap the markers on the tests to match the keys they need.

## Next steps

- [Testing](../testing/) runs the same graphs offline with `TestAgent` and mock tools.
- [Serving](../serving/) puts an agent behind the API that `10xgraph eval` can also load from `10xgraph.json`.
