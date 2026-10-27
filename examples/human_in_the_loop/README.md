---
title: Human-in-the-Loop Approval with interrupt() in a Python AI Agent (10xGraph)
description: Pause a 10xGraph agent mid-run to ask a person for approval, validate the answer with a JSON schema, and resume the same thread in Python.
section: Examples
group: "Production"
order: 100
label: Human in the loop
updated: "2026-10-28"
faq:
  - question: Does interrupt() need a checkpointer?
    answer: Yes. The paused thread is saved by the checkpointer so it can be resumed later, even from another process. interrupt_before and interrupt_after need one too.
  - question: How do I resume a paused run?
    answer: Call invoke or ainvoke on the same thread_id with {"resume": value}. Add "resume_interrupt_id" with the pending interrupt's id so a stale or repeated answer cannot approve a later prompt.
  - question: Why does my tool code run twice?
    answer: On resume the interrupted tool or node runs again from the start, and interrupt() returns the resume value instead of pausing. Put side effects such as payments after the interrupt() call.
  - question: What is the difference between interrupt() and interrupt_before?
    answer: interrupt() pauses from inside code you write and carries a payload, message and optional response schema. interrupt_before pauses the graph before a named node runs, with no code change in that node.
---

## What you will build

Two ways to put a person in charge of an agent. In `approval_interrupt.py`, a refund tool calls `interrupt()` with a JSON schema. The run pauses, the script reads the pending interrupt, and resumes once with an approval and once with a rejection. In `interrupt_before.py`, the graph is compiled to stop before the tool node, so a reviewer can inspect the tool calls and then continue.

Both use OpenAI `gpt-5-mini` through the prebuilt `ReactAgent`, and both need a checkpointer.

## Key concepts

- `interrupt(value, message=, reason=, response_schema=)`: pauses the graph from a node or tool and returns the resume value when the thread is resumed.
- `pending_interrupt(state)`: returns the `Interrupt` a paused thread is waiting on (`id`, `node`, `value`, `message`, `response_schema`, `tool_call_id`), or `None`.
- `ResponseGranularity.FULL`: makes `ainvoke` return the whole `state`, which you pass to `pending_interrupt`.
- `{"resume": value, "resume_interrupt_id": id}`: the input that resumes a paused thread.
- `ResumeValidationError` / `InterruptMismatchError`: raised when the answer breaks the schema or names the wrong interrupt. Nothing is recorded and the thread stays paused.
- `compile(interrupt_before=[...])`: pauses before the named nodes. `ReactAgent` names its tool node `"TOOL"`.
- `InMemoryCheckpointer`: stores the paused thread. Use a database checkpointer when the approver is a different process.

## Prerequisites

```bash
pip install "10xgraph[openai]" python-dotenv
export OPENAI_API_KEY="..."
```

## Run it

```bash
python examples/human_in_the_loop/approval_interrupt.py
python examples/human_in_the_loop/interrupt_before.py
```

## How it works

### A tool that asks for approval

```python
decision = interrupt(
    {"order_id": order_id, "amount": amount},
    message=f"Approve a ${amount:.2f} refund for order {order_id}?",
    reason="tool_approval",
    response_schema=DECISION_SCHEMA,
)
if not decision or not decision.get("approved"):
    return f"Refund rejected. Reviewer note: ..."
return f"Refund of ${amount:.2f} issued for order {order_id}."
```

`interrupt()` raises `GraphInterrupt`, a `BaseException`, so the tool error handling does not turn the pause into a failed tool result. The graph saves the thread and `ainvoke` returns without an assistant answer. The schema supports `type`, `enum`, `properties`, `required` and `items`. If the client cancels, the tool receives `None`, which the code above treats as a rejection.

### Read the pending interrupt

```python
result = await graph.ainvoke(inputs, config, response_granularity=ResponseGranularity.FULL)
request = pending_interrupt(result["state"])
```

`request.value` is the payload for your UI, `request.message` is the prompt, and `request.id` identifies this pause. With streaming, the same data arrives as an update event with `status="interrupted"`.

### Resume: approve or reject

```python
final = await graph.ainvoke(
    {"resume": decision, "resume_interrupt_id": request.id},
    config,
)
```

`decision` is `{"approved": True, ...}` on one thread and `{"approved": False, ...}` on the other. The tool runs again, `interrupt()` returns the decision, and the agent writes the final message. Sending `{"approved": "yes"}` first shows the schema check: `ResumeValidationError` is raised and the thread stays paused.

### Review tool calls with `interrupt_before`

```python
graph = agent.compile(checkpointer=InMemoryCheckpointer(), interrupt_before=["TOOL"])
...
for call in state.context[-1].tools_calls or []:
    print(call["function"]["name"], json.loads(call["function"]["arguments"]))
final = await graph.ainvoke({"messages": []}, config)
```

The model has decided to call `delete_account`, but the tool has not run. The last message in `state.context` holds the proposed calls in OpenAI format. Invoking the same thread again with an empty message list continues from the saved node. This form approves or inspects. To reject with a reason, use `interrupt()` inside the tool as in the first file.

## Expected output

Illustrative. Wording from the model and interrupt ids will differ.

```text
[refund-approve] paused: Approve a $42.50 refund for order A-1001?
[refund-approve] payload: {'order_id': 'A-1001', 'amount': 42.5} | interrupt id: int_3f9a...
[refund-approve] bad resume refused: ['$.approved: expected boolean, got str']
[refund-approve] assistant: The refund of $42.50 for order A-1001 has been issued.
[refund-reject] paused: Approve a $42.50 refund for order A-1001?
[refund-reject] assistant: I could not issue the refund. The reviewer noted: outside return window.
```

## Using a different model

Change `model="gpt-5-mini"` in `ReactAgent(...)` and install the matching extra, for example `model="claude-sonnet-5-5"` with `pip install "10xgraph[anthropic]"` and `ANTHROPIC_API_KEY`, or `model="gemini-2.5-flash"` with `pip install "10xgraph[google-genai]"`.

## Next steps

- [Persistence](../persistence/): keep paused threads in SQLite or Postgres so another process can resume them.
- [Resilience](../resilience/): retries, fallbacks and recursion limits.
- [Graceful shutdown](../graceful_shutdown/): flush pending checkpoint writes before the process exits.
