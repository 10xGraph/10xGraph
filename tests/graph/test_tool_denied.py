"""``ToolDenied`` blocks a tool call: the model gets a denied result, recovery cannot undo it."""

import asyncio

import pytest

from tenxgraph.core.exceptions import ToolDenied
from tenxgraph.core.graph import ToolNode
from tenxgraph.core.graph.tool_node.denial import TOOL_DENIED_CODE, TOOL_DENIED_META
from tenxgraph.core.graph.utils.invoke_node_handler import InvokeNodeHandler
from tenxgraph.core.state import AgentState, Message, StreamEvent
from tenxgraph.runtime.publisher.events import Event, EventType
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import CallbackManager, InvocationType

from .tool_guard_helpers import (
    MODES,
    FakeMCPClient,
    RecordingPublisher,
    build,
    error_codes,
    run,
    tool_messages,
    tool_results,
)


def _deny(names: set[str], reason: str = "Not for you."):
    async def before(context, input_data):
        if context.function_name in names:
            raise ToolDenied(reason)
        return input_data

    return before


def _recovery_manager(*types: InvocationType, deny: set[str]) -> tuple[CallbackManager, list]:
    """A manager whose on_error always tries to recover with a fake success."""
    seen: list = []

    async def recover(context, input_data, error):
        seen.append(error)
        return Message.text_message("recovered: success", role="tool")

    manager = CallbackManager()
    for kind in types:
        manager.register_before_invoke(kind, _deny(deny))
        manager.register_on_error(kind, recover)
    return manager, seen


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_local_tool_denied_model_sees_denied_text_and_recovery_is_skipped(mode):
    ran: list[str] = []

    def refund(amount: int = 0) -> str:
        ran.append("refund")
        return "refunded"

    manager, recovered = _recovery_manager(InvocationType.TOOL, deny={"refund"})
    app = build(
        ToolNode([refund]),
        [("c1", "refund")],
        checkpointer=InMemoryCheckpointer(),
        manager=manager,
    )
    state, chunks = await run(app, mode, {"thread_id": f"deny-local-{mode}"})

    assert ran == []
    assert recovered == []  # on_error never ran, so it could not recover
    block = tool_results(state)["c1"]
    assert block.output == "Denied by policy: refund. Not for you."
    assert block.is_error is True
    assert block.status == "failed"
    msg = tool_messages(state)[0]
    assert error_codes(msg) == [TOOL_DENIED_CODE]
    assert msg.metadata[TOOL_DENIED_META] is True
    assert state.context[-1].text() == "done"  # the run went on to the model
    if mode == "astream":
        denied = [c for c in chunks if (c.data or {}).get("status") == "tool_denied"]
        assert len(denied) == 1
        assert denied[0].event == StreamEvent.ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_mcp_tool_denied(mode):
    client = FakeMCPClient("delete_repo")
    node = ToolNode([], client=client)
    await node.all_tools()
    manager, recovered = _recovery_manager(InvocationType.MCP, deny={"delete_repo"})
    app = build(node, [("m1", "delete_repo")], checkpointer=InMemoryCheckpointer(), manager=manager)
    state, _ = await run(app, mode, {"thread_id": f"deny-mcp-{mode}"})

    assert client.calls == []
    assert recovered == []
    block = tool_results(state)["m1"]
    assert block.output == "Denied by policy: delete_repo. Not for you."
    assert block.is_error is True


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_parallel_calls_one_denied_one_runs(mode):
    ran: list[str] = []

    def lookup() -> str:
        ran.append("lookup")
        return "found"

    def refund() -> str:
        ran.append("refund")
        return "refunded"

    manager = CallbackManager()
    manager.register_before_invoke(InvocationType.TOOL, _deny({"refund"}))
    app = build(
        ToolNode([lookup, refund]),
        [("a", "lookup"), ("b", "refund")],
        checkpointer=InMemoryCheckpointer(),
        manager=manager,
    )
    state, _ = await run(app, mode, {"thread_id": f"deny-par-{mode}"})

    assert ran == ["lookup"]
    results = tool_results(state)
    assert results["a"].output == "found"
    assert results["a"].is_error is False
    assert results["b"].output.startswith("Denied by policy: refund.")


@pytest.mark.asyncio
async def test_tool_raising_tool_denied_is_a_denial():
    def wire(amount: int = 0) -> str:
        raise ToolDenied("Wires are off today.")

    manager, recovered = _recovery_manager(InvocationType.TOOL, deny=set())
    app = build(ToolNode([wire]), [("w", "wire")], manager=manager)
    state, _ = await run(app, "ainvoke", {"thread_id": "deny-in-tool"})

    assert recovered == []
    assert tool_results(state)["w"].output == "Denied by policy: wire. Wires are off today."


@pytest.mark.asyncio
async def test_ordinary_errors_still_use_recovery():
    def broken() -> str:
        raise RuntimeError("boom")

    manager, recovered = _recovery_manager(InvocationType.TOOL, deny=set())
    app = build(ToolNode([broken]), [("x", "broken")], manager=manager)
    state, _ = await run(app, "ainvoke", {"thread_id": "deny-normal-error"})

    assert len(recovered) == 1
    assert tool_messages(state)[0].text() == "recovered: success"


@pytest.mark.asyncio
async def test_denied_event_and_trace_flags():
    def refund() -> str:
        return "refunded"

    publisher = RecordingPublisher()
    manager = CallbackManager()
    manager.register_before_invoke(InvocationType.TOOL, _deny({"refund"}))
    app = build(ToolNode([refund]), [("c1", "refund")], manager=manager, publisher=publisher)
    await run(app, "ainvoke", {"thread_id": "deny-event"})
    await asyncio.sleep(0.05)

    denied = [
        e
        for e in publisher.events
        if e.event == Event.TOOL_EXECUTION and e.metadata.get("status") == "denied"
    ]
    assert len(denied) == 1
    assert denied[0].event_type == EventType.END
    assert denied[0].metadata["denied"] is True
    assert denied[0].data["error"].startswith("Denied by policy: refund.")


@pytest.mark.asyncio
async def test_denied_result_replays_from_the_ledger_as_denied():
    """A denial is recorded as a failed result: replaying the node keeps it denied and never
    runs the tool, even if the policy would now allow it."""
    ran: list[str] = []

    def refund() -> str:
        ran.append("refund")
        return "refunded"

    cp = InMemoryCheckpointer()
    manager = CallbackManager()
    manager.register_before_invoke(InvocationType.TOOL, _deny({"refund"}))
    tool_node = ToolNode([refund])
    # Compiling binds the manager in the container the handler resolves it from.
    build(tool_node, [("c1", "refund")], checkpointer=cp, manager=manager)
    config = {"thread_id": "deny-ledger", "user_id": "u1"}

    # The node's single call through the handler, twice, like a replay after a crash.
    handler = InvokeNodeHandler("TOOL", tool_node)
    call = {"id": "c1", "type": "function", "function": {"name": "refund", "arguments": "{}"}}
    first = await handler._handle_single_tool(
        call, AgentState(), config, origin_message_id="asst-1", checkpointer=cp
    )
    recorded = await cp.aget_tool_result(config, "asst-1:c1")
    assert recorded is not None
    assert recorded["value"]["content"][0]["is_error"] is True

    manager._before_callbacks[InvocationType.TOOL].clear()  # the policy now allows it
    second = await handler._handle_single_tool(
        call, AgentState(), config, origin_message_id="asst-1", checkpointer=cp
    )
    assert ran == []
    assert second.content[0].output == first.content[0].output
    assert second.content[0].output.startswith("Denied by policy")


@pytest.mark.asyncio
async def test_direct_tool_node_invoke_denied():
    def refund() -> str:
        return "refunded"

    manager = CallbackManager()
    manager.register_before_invoke(InvocationType.TOOL, _deny({"refund"}, "Nope."))
    res = await ToolNode([refund]).invoke(
        "refund", {}, "c9", {"thread_id": "x"}, AgentState(), callback_manager=manager
    )
    assert res.content[0].output == "Denied by policy: refund. Nope."


def test_tool_denied_defaults():
    err = ToolDenied()
    assert err.reason == "This tool call is not allowed."
    assert err.tool_name is None
    assert str(err) == err.reason
    assert ToolDenied("x", tool_name="t").tool_name == "t"
