"""Tool arguments are checked against the annotations shown to the model (G-49)."""

import pytest

from tenxgraph.core.graph import ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.utils import CallbackManager


async def _call(fn, args):
    node = ToolNode([fn])
    result = await node.invoke(
        name=fn.__name__,
        args=args,
        tool_call_id="c1",
        config={},
        state=AgentState(),
        callback_manager=CallbackManager(),
    )
    assert isinstance(result, Message)
    return next(b for b in result.content if isinstance(b, ToolResultBlock))


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["not a number", [1, 2], {"a": 1}])
async def test_wrong_type_is_a_failed_tool_result_and_tool_does_not_run(bad):
    ran = []

    def add(a: int, b: int) -> int:
        ran.append(a)
        return a + b

    block = await _call(add, {"a": bad, "b": 2})
    assert block.is_error is True
    assert "Invalid argument 'a'" in str(block.output)
    assert ran == []


@pytest.mark.asyncio
async def test_numeric_string_is_converted():
    def add(a: int, b: int) -> int:
        return a + b

    block = await _call(add, {"a": "3", "b": 2})
    assert block.is_error is False
    assert "5" in str(block.output)


@pytest.mark.asyncio
async def test_number_for_string_parameter_is_accepted():
    def echo(code: str) -> str:
        return f"code={code}"

    block = await _call(echo, {"code": 123})
    assert block.is_error is False
    assert "code=123" in str(block.output)


@pytest.mark.asyncio
async def test_list_and_optional_annotations_are_checked():
    def total(xs: list[int], note: str | None = None) -> int:
        return sum(xs)

    ok = await _call(total, {"xs": [1, 2, 3]})
    assert ok.is_error is False
    bad = await _call(total, {"xs": "nope"})
    assert bad.is_error is True
