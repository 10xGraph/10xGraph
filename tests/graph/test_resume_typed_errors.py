"""Resume mistakes raise typed errors (still ValueError) so the API can answer 4xx."""

import pytest

from tenxgraph.core.exceptions import GraphError, InterruptPendingError, NothingToResumeError
from tenxgraph.core.graph.utils.handler_utils import check_interrupted
from tenxgraph.core.state import AgentState
from tenxgraph.utils.interrupt import Interrupt, pause_at, record_resume


@pytest.mark.asyncio
async def test_resume_on_unpaused_thread_is_typed():
    with pytest.raises(NothingToResumeError) as info:
        await check_interrupted(AgentState(), {"resume": 1}, {})
    assert isinstance(info.value, ValueError)
    assert isinstance(info.value, GraphError)
    assert "Nothing to resume" in str(info.value)


def test_record_resume_on_unpaused_state_is_typed():
    with pytest.raises(NothingToResumeError):
        record_resume(AgentState(), 1)


@pytest.mark.asyncio
async def test_plain_message_on_paused_thread_is_typed():
    state = AgentState()
    pause_at(state, Interrupt(id="i1", key="k", node="ASK", value=1))
    with pytest.raises(InterruptPendingError) as info:
        await check_interrupted(state, {"messages": []}, {})
    assert isinstance(info.value, ValueError)
    assert "i1" in str(info.value)
