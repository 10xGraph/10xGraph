---
title: Multi-Agent Patterns in Python with 10xGraph
description: Build supervisor, swarm, manual handoff and plan-act-reflect multi-agent systems in 10xGraph where each agent runs on a different company's LLM.
section: Examples
group: "Agent patterns"
order: 180
label: Multi-agent
updated: "2026-10-28"
faq:
  - question: Can agents in the same graph use different LLM providers?
    answer: Yes. Each worker or member is a separate Agent with its own model string, so one graph can mix OpenAI, Anthropic, Google and OpenAI-compatible endpoints such as DeepSeek.
  - question: What is the difference between SupervisorTeamAgent and SwarmAgent?
    answer: In a supervisor team one model decides which worker runs next and control returns to it after each worker. In a swarm the agents hand work directly to each other with transfer_to tools and no central router.
  - question: How do I give one agent a custom base_url without affecting the others?
    answer: Set base_url and api_key on that Agent when you build it. SupervisorTeamAgent and SwarmAgent take pre-built Agent instances, so settings stay per agent. PlanActReflectAgent shares extra keyword arguments across its phases.
  - question: Does PlanActReflectAgent call a model in the act step?
    answer: No. ACT is a ToolNode that runs the tools the planner chose. Only PLAN and REFLECT call a model, so plan_model and reflect_model are the two per-phase settings.
---

# Multi-agent patterns across model providers

## What you will build

Four multi-agent systems, each with a different shape of control flow. A supervisor routes work to specialists. A swarm lets peers hand work to each other. A manual graph shows how to wire handoffs yourself. A plan, act and reflect loop separates planning from review.

In every file the agents run on models from different companies. That is possible because each agent is its own `Agent` with its own model string, and 10xGraph converts messages for each provider.

## Which pattern when

| Pattern | File | Use it when | Control flow |
|---|---|---|---|
| Supervisor team | `supervisor.py` | One coordinator should decide who works next and when to stop | Supervisor, worker, supervisor, ... until `FINISH` |
| Swarm | `swarm.py` | Specialists know best who should take over, and you want no central router | Agent hands off directly to a peer |
| Manual handoff | `handoff_manual.py` | You need custom edges, tool nodes or state beyond what `SwarmAgent` wires | `create_handoff_tool` on your own `StateGraph` |
| Plan, act, reflect | `plan_act_reflect.py` | A task needs tool use followed by a critique pass and possible retry | PLAN, ACT, REFLECT, repeat until `[DONE]` |

## Which agent uses which model

| File | Agent | Company | Model |
|---|---|---|---|
| `supervisor.py` | Supervisor | Anthropic | `claude-sonnet-5-5` |
| | RESEARCHER | OpenAI | `gpt-5-mini` |
| | ANALYST | Google | `gemini-2.5-flash` |
| | WRITER | DeepSeek | `deepseek-chat` |
| `swarm.py` | TRIAGE | OpenAI | `gpt-5-mini` |
| | RESEARCHER | Anthropic | `claude-sonnet-5-5` |
| | WRITER | Google | `gemini-2.5-flash` |
| `handoff_manual.py` | coordinator | Google | `gemini-2.5-flash` |
| | researcher | DeepSeek | `deepseek-chat` |
| | writer | Anthropic | `claude-sonnet-5-5` |
| `plan_act_reflect.py` | PLAN | Anthropic | `claude-sonnet-5-5` |
| | ACT | none (runs tools) | not applicable |
| | REFLECT | Google | `gemini-2.5-flash` |

DeepSeek is reached through its OpenAI-compatible endpoint. 10xGraph treats an unknown model name as an OpenAI-style model, so `base_url` and `api_key` on the `Agent` are all it needs.

## Key concepts

- `SupervisorTeamAgent(supervisor_model=..., workers={...})`: builds the supervisor from a model string and accepts pre-built worker agents.
- `WorkerConfig(agent=..., description=...)`: a worker and the description the supervisor reads when choosing.
- `SwarmAgent(members={...}, entry=...)`: wires peer handoffs and injects a `transfer_to_<member>` tool into each member.
- `SwarmMemberConfig(agent=..., can_handoff_to=[...], description=...)`: one member and its allowed targets.
- `create_handoff_tool(name, description)`: a tool that moves control to the node `name` without running.
- `PlanActReflectAgent(model, plan_model=..., reflect_model=..., tools=[...], max_iterations=...)`: planner, tool step and reviewer loop.
- `Agent(model=..., base_url=..., api_key=...)`: per-agent provider settings.

## Prerequisites

```bash
pip install "10xgraph[openai,anthropic,google-genai]" python-dotenv
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

The `openai` extra also covers DeepSeek. Each script needs only the keys for the models it uses, as listed in its docstring.

## Run it

```bash
python examples/multi_agent/supervisor.py
python examples/multi_agent/swarm.py
python examples/multi_agent/handoff_manual.py
python examples/multi_agent/plan_act_reflect.py
```

Each script prints every assistant reply with the model that produced it.

## How it works

### Supervisor

Workers are normal agents. Only the supervisor is built from a string:

```python
writer = Agent(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    ...
)

team = SupervisorTeamAgent(
    supervisor_model="claude-sonnet-5-5",
    workers={"RESEARCHER": WorkerConfig(agent=researcher, description="..."), ...},
    max_rounds=6,
)
```

The supervisor replies with one worker name or `FINISH`, and its choice is not added to the conversation. A worker that has tools gets its own tool loop. `max_rounds` caps the hops per turn.

### Swarm

Members are pre-built agents too. `can_handoff_to` limits who each member can pass work to, and `None` means everyone else. Do not add handoff tools yourself:

```python
swarm = SwarmAgent(
    members={
        "TRIAGE": SwarmMemberConfig(agent=triage, can_handoff_to=["RESEARCHER", "WRITER"], ...),
        "RESEARCHER": SwarmMemberConfig(agent=researcher, can_handoff_to=["WRITER"], ...),
        "WRITER": SwarmMemberConfig(agent=writer, ...),
    },
    entry="TRIAGE",
)
```

A member that replies without a handoff or tool call ends the run.

### Manual handoff

Each agent has a `ToolNode` holding its own tools and transfer tools. The tool node sees a `transfer_to_<name>` call and returns a jump to that node, so the transfer tool itself never runs:

```python
graph.add_node(name, agent)
graph.add_node(f"{name}_tools", ToolNode(tools))
graph.add_conditional_edges(name, route_to(f"{name}_tools"), {f"{name}_tools": f"{name}_tools", END: END})
graph.add_edge(f"{name}_tools", name)
```

The target passed to `create_handoff_tool` is used as the node name exactly as written, so the nodes in this file are lowercase. The `{name}_tools -> {name}` edge returns control after ordinary tools.

### Plan, act, reflect

```python
agent = PlanActReflectAgent(
    model="claude-sonnet-5-5",
    plan_model="claude-sonnet-5-5",
    reflect_model="gemini-2.5-flash",
    tools=[search_flights, search_hotels],
    max_iterations=3,
)
```

`model` is the fallback for a phase with no override. PLAN calls tools, ACT is the `ToolNode` that executes them, and REFLECT ends the loop by including `[DONE]` or sends it back to PLAN until `max_iterations`. Extra keyword arguments such as `base_url` apply to both model phases, so pick providers that can share them, or use `supervisor.py` when each agent needs its own endpoint.

## Expected output

Illustrative output of `supervisor.py`:

```text
[gpt-5-mini-2025-08-07] Findings: global EV sales grew about 25% year over year, with China at roughly 60%...

[gemini-2.5-flash] Growth of 25% on a base of 14M units implies about 3.5M additional vehicles...

[deepseek-chat] Electric vehicle sales are growing quickly, led by China. Battery costs fell 14%...
```

## Using a different model

Change the model string on any agent and install the matching extra. Swapping the supervisor to OpenAI is `supervisor_model="gpt-5-mini"` with `pip install "10xgraph[openai]"`. Swapping a worker to Gemini is `model="gemini-2.5-flash"` with `pip install "10xgraph[google-genai]"`. The rule that agents use different companies is a choice made in these examples, not a requirement.

## Next steps

- [Structured output](../structured_output/) makes a worker return typed data.
- [Observability](../observability/) shows which agent and model handled each step.
- [Callbacks](../callbacks/) adds validation and cost tracking around every model call.
