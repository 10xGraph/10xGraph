---
title: Structured Output with Pydantic in 10xGraph Agents
description: Get typed Pydantic output from a 10xGraph Python AI agent with output_schema, or use StructuredOutputAgent to validate replies and retry with repair prompts.
section: Examples
group: "Agent patterns"
order: 170
label: Structured output
updated: "2026-10-28"
faq:
  - question: What is the difference between output_schema and StructuredOutputAgent?
    answer: output_schema asks the provider to answer in a schema in one call. StructuredOutputAgent adds a validation step after generation and, if the reply does not match, sends the error back to the model and retries up to max_attempts times.
  - question: What happens when max_attempts is reached?
    answer: The graph ends and returns the last reply even if it is invalid. Validate the text yourself, as structured_output_agent.py does with model_validate_json.
  - question: Can structured output be combined with tools?
    answer: Yes. StructuredOutputAgent routes tool calls to a ToolNode and back to generation before it validates the final reply.
  - question: Does output_schema work with image or audio output?
    answer: No. output_schema requires text output, and Agent raises a ValueError for output_type image, video or audio.
---

# Structured output with Pydantic

## What you will build

Two scripts that return typed data instead of free text. The first uses `output_schema` on a `ReactAgent` to get a `MovieReview` object from one model call. The second uses `StructuredOutputAgent` to fetch weather with a tool and return a validated `WeatherReport`, retrying with a repair prompt when the reply does not match the schema.

## Key concepts

- `output_schema`: a Pydantic model (or schema dict) passed to `Agent` or `ReactAgent`. The provider is asked to answer in that shape.
- `Message.parsed_content`: the parsed object when the provider returns one. `Message.text()` always has the JSON text.
- `StructuredOutputAgent`: prebuilt GENERATE, validate, REPAIR loop with an optional tool node.
- `max_attempts`: how many repair retries are allowed per turn.
- `repair_system_prompt`: optional. If set, repair uses a dedicated model call instead of an injected correction message.

## Prerequisites

```bash
pip install "10xgraph[openai]" python-dotenv
export OPENAI_API_KEY="your-key"
```

Both scripts use `gpt-5-mini`.

## Run it

```bash
python examples/structured_output/agent_output_schema.py
python examples/structured_output/structured_output_agent.py
```

## How it works

### One call with `output_schema`

```python
agent = ReactAgent(
    model="gpt-5-mini",
    output_schema=MovieReview,
    system_prompt=[{"role": "system", "content": "You are a film critic."}],
)
```

The last message holds the result. The script uses `parsed_content` when it is a `MovieReview` and otherwise validates the JSON text, so the same code works on providers that return text only.

### Validate and repair with `StructuredOutputAgent`

```python
agent = StructuredOutputAgent(
    model="gpt-5-mini",
    output_schema=WeatherReport,
    tools=[get_current_weather],
    max_attempts=3,
)
```

The graph is `GENERATE`, then a routing check. A tool call goes to `TOOL` and back. A valid reply ends the run. An invalid reply goes to `REPAIR`, which adds a user message with the validation error and the target JSON Schema, then back to `GENERATE`. `WeatherReport.advice` is a `Literal`, so a reply with any other value fails validation and triggers a repair.

After `max_attempts` repairs the agent returns what it has. The script therefore ends with `WeatherReport.model_validate_json(last.text())` and handles the failure case.

## Expected output

Illustrative output of `structured_output_agent.py`:

```text
Paris: 22.5 C, partly cloudy
humidity 68%, advice: sunscreen
```

## Using a different model

Change the model string and install the matching extra. For Gemini use `model="gemini-2.5-flash"` with `pip install "10xgraph[google-genai]"` and `GEMINI_API_KEY`. For Anthropic use `model="claude-sonnet-5-5"` with `pip install "10xgraph[anthropic]"` and `ANTHROPIC_API_KEY`.

## Next steps

- [Multi-agent](../multi_agent/) combines agents from different providers in one graph.
- [Callbacks](../callbacks/) validates input before it reaches the model.
- [Evaluation](../evaluation/) scores structured answers against expected values.
