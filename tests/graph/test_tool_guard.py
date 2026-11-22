"""ToolGuard rules: allow list, permissions via Identity, strict mode, argument checks."""

import logging

import pytest

from tenxgraph.core.authz import Identity
from tenxgraph.core.exceptions import ToolDenied
from tenxgraph.core.graph import ToolGuard, ToolNode
from tenxgraph.core.state import AgentState
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import CallbackManager, InvocationType

from .tool_guard_helpers import MODES, FakeMCPClient, build, run, tool_results


RAN: list[str] = []


def search_web(query: str = "") -> str:
    RAN.append("search_web")
    return "results"


def refund(amount: int = 0) -> str:
    RAN.append("refund")
    return "refunded"


def delete_user(user: str = "") -> str:
    RAN.append("delete_user")
    return "deleted"


@pytest.fixture(autouse=True)
def _clear():
    RAN.clear()


def _user(permissions=None) -> Identity:
    return Identity(
        user_id="u1", permissions=frozenset(permissions) if permissions is not None else None
    )


async def _call(guard: ToolGuard, name: str, config: dict, args: dict | None = None):
    """One call through a real ToolNode with the guard registered."""
    manager = CallbackManager()
    guard.register(manager)
    res = await ToolNode([search_web, refund, delete_user]).invoke(
        name, args or {}, f"call-{name}", config, AgentState(), callback_manager=manager
    )
    return res.content[0]


def test_register_covers_tool_and_mcp():
    manager = CallbackManager()
    guard = ToolGuard().register(manager)
    assert manager._before_callbacks[InvocationType.TOOL] == [guard]
    assert manager._before_callbacks[InvocationType.MCP] == [guard]


@pytest.mark.asyncio
async def test_allow_list_with_patterns_denies_everything_else():
    guard = ToolGuard(allow={"search_*", "refund"})
    assert (await _call(guard, "search_web", {})).output == "results"
    assert (await _call(guard, "refund", {})).output == "refunded"
    denied = await _call(guard, "delete_user", {})
    assert denied.output == "Denied by policy: delete_user. This tool is not on the allow list."
    assert RAN == ["search_web", "refund"]


@pytest.mark.asyncio
async def test_no_rule_runs_as_before():
    guard = ToolGuard(permissions={"refund": "tools:refund"})
    assert (await _call(guard, "search_web", {"user": _user(set())})).output == "results"


@pytest.mark.asyncio
async def test_permission_allowed_and_denied():
    guard = ToolGuard(permissions={"refund": "tools:refund", "delete_*": {"tools:delete", "admin"}})
    assert (await _call(guard, "refund", {"user": _user({"tools:refund"})})).output == "refunded"
    assert (await _call(guard, "refund", {"user": _user({"tools:*"})})).output == "refunded"
    assert (await _call(guard, "refund", {"user": _user({"*"})})).output == "refunded"

    denied = await _call(guard, "refund", {"user": _user({"threads:read"})})
    assert denied.output == "Denied by policy: refund. Missing permission: tools:refund."
    # Every required permission is needed.
    partial = await _call(guard, "delete_user", {"user": _user({"tools:delete"})})
    assert partial.output == "Denied by policy: delete_user. Missing permission: admin."
    assert RAN == ["refund", "refund", "refund"]


@pytest.mark.asyncio
async def test_identity_as_dict_is_read():
    guard = ToolGuard(permissions={"refund": "tools:refund"})
    config = {"user": {"user_id": "u1", "permissions": ["tools:other"]}}
    assert (await _call(guard, "refund", config)).output.startswith("Denied by policy")


@pytest.mark.asyncio
async def test_malformed_identity_is_denied():
    guard = ToolGuard(permissions={"refund": "tools:refund"})
    res = await _call(guard, "refund", {"user": {"user_id": ""}})
    assert res.output == "Denied by policy: refund. The caller identity is not valid."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "strict", "allowed"),
    [
        ({"user": _user(None)}, False, True),  # permissions None: unrestricted
        ({"user": _user(None)}, True, False),
        ({}, False, True),  # no identity at all
        ({}, True, False),
        ({"user_id": "u1"}, True, False),  # a bare user_id is not an identity
    ],
)
async def test_unrestricted_and_missing_identity(config, strict, allowed):
    guard = ToolGuard(permissions={"refund": "tools:refund"}, strict=strict)
    res = await _call(guard, "refund", config)
    if allowed:
        assert res.output == "refunded"
    else:
        assert res.output == (
            "Denied by policy: refund. The caller has no permissions for this tool."
        )


@pytest.mark.asyncio
async def test_strict_only_affects_tools_with_rules():
    guard = ToolGuard(permissions={"refund": "tools:refund"}, strict=True)
    assert (await _call(guard, "search_web", {})).output == "results"


@pytest.mark.asyncio
async def test_check_sync_returning_reason():
    seen: list = []

    def check(name, args, identity):
        seen.append((name, args, identity.user_id if identity else None))
        if name == "refund" and args.get("amount", 0) > 500:
            return "Refunds above 500 need a manager."
        return None

    guard = ToolGuard(check=check)
    config = {"user": _user({"*"})}
    assert (await _call(guard, "refund", config, {"amount": 10})).output == "refunded"
    res = await _call(guard, "refund", config, {"amount": 900})
    assert res.output == "Denied by policy: refund. Refunds above 500 need a manager."
    assert seen[1] == ("refund", {"amount": 900}, "u1")


@pytest.mark.asyncio
async def test_check_async_raising_tool_denied():
    async def check(name, args, identity):
        if args.get("user") == "root":
            raise ToolDenied("Root cannot be deleted.")

    guard = ToolGuard(check=check)
    assert (await _call(guard, "delete_user", {}, {"user": "bob"})).output == "deleted"
    res = await _call(guard, "delete_user", {}, {"user": "root"})
    assert res.output == "Denied by policy: delete_user. Root cannot be deleted."


@pytest.mark.asyncio
async def test_crashing_check_fails_closed():
    def check(name, args, identity):
        raise KeyError("boom")

    res = await _call(ToolGuard(check=check), "refund", {})
    assert res.output == "Denied by policy: refund. The argument check failed."
    assert RAN == []


@pytest.mark.asyncio
async def test_denial_log_has_no_argument_values(caplog):
    caplog.set_level(logging.INFO)
    secret = "4111-1111-1111-1111"  # noqa: S105 - a test card number, not a password

    def check(name, args, identity):
        raise ValueError(f"bad card {args['card']}")

    guard = ToolGuard(permissions={"refund": "tools:refund"}, check=check)
    await _call(guard, "refund", {"user": _user({"tools:refund"})}, {"card": secret})
    await _call(
        ToolGuard(permissions={"refund": "tools:refund"}),
        "refund",
        {"user": _user(set())},
        {"card": secret},
    )
    warnings = [r for r in caplog.records if "denied by policy" in r.getMessage()]
    assert len(warnings) == 2
    assert all(r.levelno == logging.WARNING for r in warnings)
    assert "u1" in warnings[0].getMessage()
    assert not any(secret in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_guard_in_a_graph_with_local_and_mcp_tools(mode):
    client = FakeMCPClient("gh_delete_repo", "gh_list_repos")
    node = ToolNode([search_web], client=client)
    await node.all_tools()
    manager = CallbackManager()
    ToolGuard(allow={"search_*", "gh_list_*"}).register(manager)
    app = build(
        node,
        [("a", "search_web"), ("b", "gh_delete_repo"), ("c", "gh_list_repos")],
        checkpointer=InMemoryCheckpointer(),
        manager=manager,
    )
    state, _ = await run(app, mode, {"thread_id": f"guard-graph-{mode}"})

    results = tool_results(state)
    assert results["a"].output == "results"
    assert results["b"].output.startswith("Denied by policy: gh_delete_repo.")
    assert results["c"].is_error is False
    assert client.calls == [("gh_list_repos", {})]
