---
title: Agent Skills in 10xGraph - On-Demand and Per-Tenant Session Mode
description: Load SKILL.md instructions into a 10xGraph Python agent on demand with activate_skill, or preload a per-tenant persona with session mode. Includes bundled resources and validation.
section: Examples
group: "Agent patterns"
order: 190
label: Skills
updated: "2026-10-28"
faq:
  - question: What is the difference between on-demand and session mode?
    answer: In on-demand mode the system prompt lists each skill's name and description and the model calls activate_skill when a request matches. In session mode the framework preloads one skill named by a state field before every call, with no catalog and no extra tool round trip.
  - question: How does the agent read files bundled with a skill?
    answer: When a skill folder contains extra files such as references/ or scripts/, the agent gets a read_skill_resource tool. The SKILL.md body tells the model which file to open, and reads are capped by max_resource_bytes.
  - question: How do I check that a skill follows the spec?
    answer: Call validate_skill(path) from tenxgraph.core.skills, which returns diagnostics, or run 10xgraph skills --validate. The format follows the Agent Skills specification at agentskills.io.
---

## What you will build

Two small agents that use skills. `on_demand.py` has three general skills (code review, data analysis, writing help) and lets the model decide when to load one. `session_mode.py` serves two tenants from one agent, where each tenant's support persona is a skill chosen by a state field.

## Key concepts

- `SkillConfig(skills_dir=...)`: discovers `<dir>/<name>/SKILL.md`. The folder name must match the `name` in the frontmatter.
- On-demand mode (default): an `<available_skills>` catalog goes into the system prompt and the model calls `activate_skill`.
- `read_skill_resource`: registered when a skill bundles files. The `code-review` skill ships `references/severity-guide.md`.
- `mode="session"` with `preload_from`: reads the named `AgentState` field on every call and injects that skill.
- `hot_reload`: re-reads a SKILL.md when it changes on disk.
- `validate_skill(path)` and `10xgraph skills --validate`: check skills against the Agent Skills spec.

## Prerequisites

```bash
pip install "10xgraph[anthropic]"
export ANTHROPIC_API_KEY=...
```

## Run it

```bash
python examples/skills/on_demand.py          # two scripted questions
python examples/skills/on_demand.py --chat   # interactive prompt, type quit to stop
python examples/skills/session_mode.py
```

## How it works

Layout:

```text
skills/
  code-review/SKILL.md
  code-review/references/severity-guide.md
  data-analysis/SKILL.md
  writing-assistant/SKILL.md
personas/
  acme-support/SKILL.md
  globex-support/SKILL.md
```

On-demand setup. Ordinary tools sit next to the skill tools:

```python
agent = ReactAgent(
    model="claude-sonnet-5-5",
    tools=[get_weather],
    skills=SkillConfig(skills_dir=str(SKILLS_DIR), hot_reload=True),
)
```

The script prints a `skill loaded` line whenever a tool message starts with `<skill_content name="...">`. A weather question loads nothing. The code-review skill tells the model to read its severity guide with `read_skill_resource`.

Session mode needs a state field that holds the skill name:

```python
class TenantState(AgentState):
    persona: str = ""

skills=SkillConfig(skills_dir=..., mode="session", preload_from="persona")
```

Each request sets the field through the partial `state` input:

```python
{"messages": [...], "state": {"persona": "acme-support"}}
```

Set that field from the authenticated tenant on the server, not from client input, or a user could switch to another tenant's persona.

Before shipping, validate:

```bash
10xgraph skills --validate
```

`on_demand.py` also calls `validate_skill` on each folder and prints the result.

## Expected output

Illustrative output.

```text
code-review: ok
...
  skill loaded: code-review
assistant: Summary: Significant issues. avg() raises ZeroDivisionError on an empty list...

assistant: Paris: 18 C, light rain.
```

Session mode:

```text
[acme] assistant: To request a refund, please return the item within 30 days with your receipt.
[globex] assistant: Easy. Open Settings > Billing in the app and ask for a refund within 14 days.
```

## Using a different model

Change `model="claude-sonnet-5-5"` to `"gpt-5-mini"` (extra `openai`) or `"gemini-2.5-flash"` (extra `google-genai`). Skills are plain prompt text and tools, so they work with any provider.

## Next steps

- [Long-term memory](../long_term_memory/)
- [Tool decorator](../tools/)
- [Prebuilt ReAct agent](../quickstart/)
