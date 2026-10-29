"""G-05: the reflector must never see an assistant tool call without its result."""

from __future__ import annotations

import pytest

from tenxgraph.core.state import AgentState, Message
from tenxgraph.prebuilt.agent.plan_act_reflect import _make_reflect_node
from tenxgraph.utils import END
from tenxgraph.utils.converter import convert_messages


class _CapturingReflector:
    def __init__(self):
        self.seen = None

    async def execute(self, state, config):
        self.seen = list(state.context)
        return []


@pytest.mark.asyncio
async def test_reflector_context_has_no_unanswered_tool_calls():
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    assistant = Message.text_message("plan", role="assistant")
    assistant.tools_calls = [call]
    tool = Message.tool_message(content=[], message_id="m")
    state = AgentState()
    state.context = [Message.text_message("task", role="user"), assistant, tool]
    original = list(state.context)

    reflector = _CapturingReflector()
    node = _make_reflect_node(reflector)
    await node(state, {})

    out = convert_messages(system_prompts=[], state=AgentState(context=reflector.seen))
    assert not any(m.get("tool_calls") for m in out)
    assert not any(m["role"] == "tool" for m in out)
    # shared state is restored untouched
    assert state.context == original
    assert state.context[1].tools_calls == [call]


def test_structured_output_marks_reply_invalid_after_max_attempts():
    from pydantic import BaseModel

    from tenxgraph.prebuilt.agent.structured_output import _ATTEMPTS_KEY, _make_route_fn

    class Out(BaseModel):
        n: int

    route = _make_route_fn(Out, 1, has_tools=False)
    reply = Message.text_message("not json", role="assistant")
    state = AgentState()
    state.context = [reply]
    state.execution_meta.internal_data[_ATTEMPTS_KEY] = 1
    assert route(state) == END
    assert reply.metadata["structured_output_valid"] is False
    assert reply.metadata["structured_output_error"]
