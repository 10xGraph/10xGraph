"""Per-tool human approval with ToolGuard(approval=...): pause, approve once, deny, parallel."""

import pytest

from tenxgraph.core.graph import ToolGuard, ToolNode
from tenxgraph.core.state import AgentState, StreamEvent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import CallbackManager, interrupt, pending_interrupt

from .tool_guard_helpers import MODES, FakeMCPClient, build, run, tool_results


def _guarded(node: ToolNode, calls, *, args=None, **guard_kwargs):
    manager = CallbackManager()
    ToolGuard(**guard_kwargs).register(manager)
    return build(node, calls, args=args, checkpointer=InMemoryCheckpointer(), manager=manager)


def _stream_interrupt(chunks):
    found = [
        c.data["interrupt"]
        for c in chunks
        if c.event == StreamEvent.UPDATES and (c.data or {}).get("status") == "interrupted"
    ]
    assert len(found) == 1
    return found[0]


async def _pending(app, mode, state, chunks, config):
    if mode == "astream":
        return _stream_interrupt(chunks)
    return pending_interrupt(state).model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("answer", [True, {"approved": True}])
async def test_pause_payload_then_approve_runs_once(mode, answer):
    ran: list[int] = []

    def refund(amount: int = 0) -> str:
        ran.append(amount)
        return f"refunded {amount}"

    app = _guarded(ToolNode([refund]), [("c1", "refund")], args={"amount": 40}, approval={"refund"})
    config = {"thread_id": f"appr-{mode}-{answer!s}"}

    state, chunks = await run(app, mode, config)
    assert ran == []
    request = await _pending(app, mode, state, chunks, config)
    assert request["reason"] == "tool_approval"
    assert request["tool_call_id"] == "c1"
    assert request["value"] == {
        "tool_name": "refund",
        "args": {"amount": 40},
        "tool_call_id": "c1",
        "action": "Run tool 'refund'",
    }
    assert request["message"] == "Approve: Run tool 'refund'?"

    state, _ = await run(app, mode, config, {"resume": answer})
    assert ran == [40]
    assert tool_results(state)["c1"].output == "refunded 40"
    assert state.context[-1].text() == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("answer", [None, False, {"approved": False}])
async def test_deny_returns_denied_result(mode, answer):
    ran: list[str] = []

    def refund() -> str:
        ran.append("refund")
        return "refunded"

    app = _guarded(ToolNode([refund]), [("c1", "refund")], approval={"refund"})
    config = {"thread_id": f"appr-deny-{mode}-{answer!s}"}
    await run(app, mode, config)
    state, _ = await run(app, mode, config, {"resume": answer})

    assert ran == []
    block = tool_results(state)["c1"]
    assert block.output == ("Denied by policy: refund. A human reviewer did not approve this call.")
    assert block.is_error is True


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_parallel_calls_each_get_their_own_approval(mode):
    ran: list[str] = []

    def refund() -> str:
        ran.append("refund")
        return "refunded"

    def wire() -> str:
        ran.append("wire")
        return "wired"

    def lookup() -> str:
        ran.append("lookup")
        return "found"

    app = _guarded(
        ToolNode([refund, wire, lookup]),
        [("a", "refund"), ("b", "wire"), ("c", "lookup")],
        approval={"refund", "wire"},
    )
    config = {"thread_id": f"appr-par-{mode}"}

    state, chunks = await run(app, mode, config)
    first = await _pending(app, mode, state, chunks, config)
    assert ran == ["lookup"]  # the unguarded sibling ran and is in the ledger

    resume = {"resume": True, "resume_interrupt_id": first["id"]}
    state, chunks = await run(app, mode, config, resume)
    second = await _pending(app, mode, state, chunks, config)
    assert {first["tool_call_id"], second["tool_call_id"]} == {"a", "b"}

    state, _ = await run(app, mode, config, {"resume": False})
    results = tool_results(state)
    approved = first["value"]["tool_name"]
    denied = second["value"]["tool_name"]
    # Each tool ran at most once, whatever the order of the pauses.
    assert sorted(ran) == sorted(["lookup", approved])
    by_name = {"refund": results["a"], "wire": results["b"]}
    assert by_name[approved].is_error is False
    assert by_name[denied].output.startswith(f"Denied by policy: {denied}.")
    assert results["c"].output == "found"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_mcp_tool_approval(mode):
    client = FakeMCPClient("delete_repo")
    node = ToolNode([], client=client)
    await node.all_tools()
    app = _guarded(node, [("m1", "delete_repo")], approval={"delete_*"})
    config = {"thread_id": f"appr-mcp-{mode}"}

    state, chunks = await run(app, mode, config)
    assert client.calls == []
    request = await _pending(app, mode, state, chunks, config)
    assert request["value"]["tool_name"] == "delete_repo"

    state, _ = await run(app, mode, config, {"resume": True})
    assert client.calls == [("delete_repo", {})]
    assert tool_results(state)["m1"].is_error is False


@pytest.mark.asyncio
async def test_tool_interrupt_after_approval_is_answered_separately():
    """The guard's pause and an interrupt() inside the tool are two keys of the same call."""
    answers: list = []

    def pick() -> str:
        answers.append(interrupt("which size?"))
        return f"picked {answers[-1]}"

    app = _guarded(ToolNode([pick]), [("p", "pick")], approval={"pick"})
    config = {"thread_id": "appr-nested"}
    state, _ = await run(app, "ainvoke", config)
    assert pending_interrupt(state).reason == "tool_approval"
    state, _ = await run(app, "ainvoke", config, {"resume": True})
    assert pending_interrupt(state).value == "which size?"
    state, _ = await run(app, "ainvoke", config, {"resume": "L"})
    assert tool_results(state)["p"].output == "picked L"
    assert answers == ["L"]


@pytest.mark.asyncio
async def test_approval_after_failed_rules_is_not_asked():
    def refund() -> str:
        return "refunded"

    app = _guarded(
        ToolNode([refund]),
        [("c1", "refund")],
        approval={"refund"},
        permissions={"refund": "tools:refund"},
        strict=True,
    )
    state, _ = await run(app, "ainvoke", {"thread_id": "appr-rules-first"})
    assert pending_interrupt(state) is None
    assert tool_results(state)["c1"].output.startswith("Denied by policy: refund.")


@pytest.mark.asyncio
async def test_direct_invoke_without_a_graph_denies():
    def refund() -> str:
        return "refunded"

    manager = CallbackManager()
    ToolGuard(approval={"refund"}).register(manager)
    res = await ToolNode([refund]).invoke(
        "refund", {}, "c1", {"thread_id": "x"}, AgentState(), callback_manager=manager
    )
    assert res.content[0].output == (
        "Denied by policy: refund. This call needs approval, but the run cannot pause."
    )
