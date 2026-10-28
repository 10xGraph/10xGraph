"""Main assistant (OpenAI gpt-5-mini): a task-board agent served over AG-UI.

Loaded by 10xgraph.json ("agent": "graph.agent:app"). The tools change `state`, and the API
server sends the changed fields to CopilotKit as STATE_SNAPSHOT events.
"""

from dotenv import load_dotenv

from graph.planner import plan_steps
from graph.state import BoardState
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import interrupt


load_dotenv()

_STATUSES = ("todo", "doing", "done")


def _new_task(state: BoardState, title: str, priority: str) -> dict:
    task = {"id": state.next_id, "title": title, "status": "todo", "priority": priority}
    state.next_id += 1
    state.tasks.append(task)
    return task


def add_task(title: str, priority: str = "medium", state: BoardState | None = None) -> str:
    """Add one task to the board. priority is low, medium or high."""
    if state is None:
        return "No board state available."
    task = _new_task(state, title, priority)
    return f"Added task #{task['id']}: {title}"


def set_task_status(task_id: int, status: str, state: BoardState | None = None) -> str:
    """Move a task to todo, doing or done."""
    if state is None:
        return "No board state available."
    if status not in _STATUSES:
        return f"Status must be one of {', '.join(_STATUSES)}."
    for task in state.tasks:
        if task["id"] == task_id:
            task["status"] = status
            return f"Task #{task_id} is now {status}."
    return f"No task with id {task_id}."


async def plan_goal(goal: str, state: BoardState | None = None) -> str:
    """Ask the planner agent to split a goal into steps and add them as tasks."""
    if state is None:
        return "No board state available."
    steps = await plan_steps(goal)
    for step in steps:
        _new_task(state, step, "medium")
    return f"Planner added {len(steps)} tasks: " + "; ".join(steps)


def clear_done_tasks(state: BoardState | None = None) -> str:
    """Delete every task marked done. Asks the user for approval first."""
    if state is None:
        return "No board state available."
    done = [t for t in state.tasks if t["status"] == "done"]
    if not done:
        return "There are no finished tasks to clear."
    # interrupt() pauses the run. The AG-UI endpoint ends it with an interrupt outcome, and
    # CopilotKit's useInterrupt renders the approval. The tool runs again on resume, so nothing
    # with side effects may come before this call.
    decision = interrupt(
        value={"count": len(done), "titles": [t["title"] for t in done]},
        message=f"Delete {len(done)} finished task(s)?",
        reason="clear_done_approval",
        response_schema={
            "type": "object",
            "properties": {"approved": {"type": "boolean"}},
            "required": ["approved"],
        },
    )
    if not decision or not decision.get("approved"):
        return "The user declined. No tasks were deleted."
    state.tasks = [t for t in state.tasks if t["status"] != "done"]
    return f"Deleted {len(done)} finished task(s)."


react_agent = ReactAgent(
    model="gpt-5-mini",
    state=BoardState(),
    system_prompt=[
        {
            "role": "system",
            "content": (
                "You manage a task board shown next to the chat. Use add_task for a single "
                "task and plan_goal when the user states a larger goal. Use set_task_status "
                "to move tasks and clear_done_tasks to delete finished ones. After you add or "
                "change a task, call the browser tool focus_task with its id so the user sees "
                "it. Keep replies short."
            ),
        }
    ],
    tools=[add_task, set_task_status, plan_goal, clear_done_tasks],
)

app = react_agent.compile(checkpointer=InMemoryCheckpointer())
