"""A call with no new input after a cancel inside a ToolNode resumes (decision D1).

Once the repair has answered every tool call of the saved tool message (from the ledger, or
with a synthetic result), the tool node has nothing left to run. The resume must continue at
the node after it, not fail with NODE_002 "No tool calls to execute".
"""

import pytest

from tenxgraph.core.state import ExecutionStatus
from tests.graph.test_cancel_safety import Harness, assert_valid_history


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_call_without_input_after_cancel_in_a_tool_node_resumes_after_it(mode):
    h = Harness(mode)
    await h.cancel_run("use tool", f"t-toolresume-{mode}")

    h.histories.clear()
    config = {"thread_id": f"t-toolresume-{mode}"}
    if mode == "ainvoke":
        await h.app.ainvoke({}, config)
    else:
        _ = [c async for c in h.app.astream({}, config)]

    assert h.histories, "the model ran after the tool node"
    assert_valid_history(h.histories[0])
    assert h.histories[0][-1].role == "tool"
    state = await h.checkpointer.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert state.context[-1].text() == "finished"
