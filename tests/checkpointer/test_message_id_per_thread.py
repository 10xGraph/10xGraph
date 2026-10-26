"""Two threads may store the same message id without clashing (SQLite and in-memory)."""

import pytest

from tenxgraph.core.state import Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer


@pytest.fixture(params=["memory", "sqlite"])
def cp(request):
    return InMemoryCheckpointer() if request.param == "memory" else SqliteCheckpointer(":memory:")


@pytest.mark.asyncio
async def test_same_message_id_in_two_threads(cp):
    a, b = {"thread_id": "a", "user_id": "u1"}, {"thread_id": "b", "user_id": "u2"}
    await cp.aput_messages(a, [Message.text_message("from a", message_id="m1")])
    await cp.aput_messages(b, [Message.text_message("from b", message_id="m1")])

    assert (await cp.aget_message(a, "m1")).text() == "from a"
    assert (await cp.aget_message(b, "m1")).text() == "from b"

    await cp.adelete_message(a, "m1")
    with pytest.raises(IndexError):
        await cp.aget_message(a, "m1")
    assert (await cp.aget_message(b, "m1")).text() == "from b"
