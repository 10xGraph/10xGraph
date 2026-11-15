"""A tool's returned dict reaches the model unchanged; status/error keys only decide is_error."""

import pytest

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.state import AgentState
from tenxgraph.core.state.message import ToolResultBlock
from tenxgraph.utils import CallbackManager


ORDER = {"order_id": "1042", "status": "delivered", "total": 59.0}


def lookup_order() -> dict:
    """Business data with a status field (sync tool)."""
    return dict(ORDER)


async def lookup_order_async() -> dict:
    """Business data with a status field (async tool)."""
    return dict(ORDER)


async def _block(fn, name: str) -> ToolResultBlock:
    msg = await ToolNode([fn]).invoke(
        name,
        {},
        tool_call_id="c1",
        config={},
        state=AgentState(),
        callback_manager=CallbackManager(),
    )
    blocks = [b for b in msg.content if isinstance(b, ToolResultBlock)]
    assert len(blocks) == 1
    return blocks[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("fn", [lookup_order, lookup_order_async])
async def test_status_key_reaches_model_and_is_not_an_error(fn):
    block = await _block(fn, fn.__name__)
    assert block.output == [ORDER]
    assert block.is_error is False
    assert block.status == "completed"


CASES = [
    ({"error": None, "x": 1}, False),
    ({"error": "", "x": 1}, False),
    ({"error": "card declined", "x": 1}, False),
    ({"error": "true", "x": 1}, True),
    ({"is_error": True, "x": 1}, True),
    ({"success": True, "x": 1}, False),
    ({"success": False, "x": 1}, True),
    ({"status": "failed", "x": 1}, True),
    ({"status": "failed_delivery", "x": 1}, False),
    ({"status": "pending", "x": 1}, False),
    ({"status": "cancelled", "x": 1}, False),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("payload", "expected_error"), CASES)
async def test_keys_kept_and_is_error_derived(payload, expected_error):
    def tool() -> dict:
        """Return the payload."""
        return dict(payload)

    block = await _block(tool, "tool")
    assert block.output == [payload]
    assert block.is_error is expected_error
    assert block.status == ("failed" if expected_error else "completed")
