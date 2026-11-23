"""ToolGuard: allow lists, permissions, argument checks and human approval for tool calls."""

from __future__ import annotations

import inspect
import logging
import typing as t
from collections.abc import Callable, Iterable
from fnmatch import fnmatchcase

from pydantic import ValidationError

from tenxgraph.core.authz import Identity, get_identity
from tenxgraph.core.exceptions import ToolDenied
from tenxgraph.utils.callbacks import (
    BeforeInvokeCallback,
    CallbackContext,
    CallbackManager,
    InvocationType,
)
from tenxgraph.utils.interrupt import interrupt
from tenxgraph.utils.log_safety import log_safe


logger = logging.getLogger("tenxgraph.graph.tool_node")

# ``Interrupt.reason`` of an approval pause.
APPROVAL_REASON = "tool_approval"

_APPROVAL_SCHEMA: dict[str, t.Any] = {
    "type": ["boolean", "object"],
    "properties": {"approved": {"type": "boolean"}},
}

ArgumentCheck = Callable[[str, dict[str, t.Any], Identity | None], t.Any]


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def is_approved(value: t.Any) -> bool:
    """Whether a resume value approves the call: ``True`` or ``{"approved": True}``."""
    if isinstance(value, dict):
        return value.get("approved") is True
    return value is True


class ToolGuard(BeforeInvokeCallback[dict[str, t.Any], dict[str, t.Any]]):
    """A ready-made tool policy, run before every local and MCP tool call.

    The framework gives the controls; the policy (which tools, which permissions, who
    approves) is yours. Rules, checked in this order for each call:

    1. ``allow``: tool name patterns (``fnmatch``, case-sensitive, so ``"github_*"`` works).
       When set, a tool that matches none of them is denied. ``None`` means no allow list.
    2. ``permissions``: ``{pattern: permission or set of permissions}``. A call needs every
       permission of every matching pattern, checked with ``Identity.can`` on the identity in
       ``config["user"]``. With ``strict=False`` (default) there is no check when the run has
       no identity or ``identity.permissions`` is ``None`` (unrestricted: direct SDK, auth off),
       as ``Identity`` defines. With ``strict=True`` both cases are denied for a tool that has
       a permission rule.
    3. ``check``: ``check(tool_name, args, identity)``, sync or async. Return ``None`` to allow,
       return a string (the reason) or raise ``ToolDenied`` to deny. Any other exception
       denies too (fail closed) and is logged. ``args`` are the model's raw arguments.
    4. ``approval``: tool name patterns that need a human yes before they run. The run pauses
       with ``interrupt()`` (reason ``"tool_approval"``) keyed to the tool call, so parallel
       calls are approved one by one. The interrupt ``value`` is
       ``{"tool_name", "args", "tool_call_id", "action"}``. Resume with ``True`` or
       ``{"approved": True}`` to run the tool; ``None`` (a cancel), ``False``,
       ``{"approved": False}`` or anything else denies it. The TS client's
       ``approveInterrupt`` / ``rejectInterrupt`` send ``true`` / ``null``.

    A denied call never runs: the model gets ``"Denied by policy: <tool>. <reason>"`` as a
    failed tool result (see ``ToolDenied``). Tools that match no rule run as before.
    Approval needs a checkpointer so the pause can be resumed. An approved call runs once: its
    result goes to the tool ledger, so a later replay of the node does not run it again.

    Not covered: client tools (``remote_tools`` and ``attach_remote_tools``) run in the
    frontend, not in the ToolNode, so before_invoke callbacks never see them. Skill tools
    (``activate_skill``) report as ``InvocationType.SKILL`` and are not guarded.
    Hiding a tool from the model is not enforcement; this guard is.

    Example:
        ```python
        from tenxgraph.core.graph import ToolGuard
        from tenxgraph.utils import CallbackManager

        manager = CallbackManager()
        ToolGuard(
            allow={"search_*", "get_order", "refund"},
            permissions={"refund": "tools:refund"},
            approval={"refund"},
            check=lambda name, args, identity: (
                "Refunds above 500 need a manager." if args.get("amount", 0) > 500 else None
            ),
        ).register(manager)
        app = graph.compile(checkpointer=InMemoryCheckpointer(), callback_manager=manager)

        result = await app.ainvoke({"messages": [...]}, config)  # pauses before refund
        await app.ainvoke({"resume": {"approved": True}}, config)  # refund runs once
        ```
    """

    def __init__(
        self,
        allow: set[str] | None = None,
        permissions: dict[str, str | set[str]] | None = None,
        approval: set[str] | None = None,
        check: ArgumentCheck | None = None,
        strict: bool = False,
    ) -> None:
        self.allow = frozenset(allow) if allow is not None else None
        self.permissions = {
            pattern: frozenset([required] if isinstance(required, str) else required)
            for pattern, required in (permissions or {}).items()
        }
        self.approval = frozenset(approval or ())
        self.check = check
        self.strict = strict

    def register(self, callback_manager: CallbackManager) -> ToolGuard:
        """Register the guard as a before_invoke callback for local and MCP tools.

        Register it on the ``CallbackManager`` you pass to ``graph.compile(callback_manager=)``.
        Each graph uses only the manager it was compiled with, so the guard applies to that
        graph alone.
        """
        callback_manager.register_before_invoke(InvocationType.TOOL, self)
        callback_manager.register_before_invoke(InvocationType.MCP, self)
        return self

    async def __call__(
        self, context: CallbackContext, input_data: dict[str, t.Any]
    ) -> dict[str, t.Any]:
        name = context.function_name or ""
        metadata = context.metadata or {}
        args = metadata.get("args") or {}
        config = metadata.get("config") or {}

        if self.allow is not None and not _matches(name, self.allow):
            raise ToolDenied("This tool is not on the allow list.", tool_name=name)

        identity = self._identity(name, config)
        self._check_permissions(name, identity)
        await self._run_check(name, args, identity)

        if _matches(name, self.approval):
            self._ask_approval(name, args, metadata.get("tool_call_id"))
        return input_data

    def _identity(self, name: str, config: dict[str, t.Any]) -> Identity | None:
        try:
            return get_identity(config)
        except (ValidationError, ValueError) as exc:
            # A malformed identity never means "no checks".
            raise ToolDenied("The caller identity is not valid.", tool_name=name) from exc

    def _required(self, name: str) -> set[str]:
        required: set[str] = set()
        for pattern, perms in self.permissions.items():
            if fnmatchcase(name, pattern):
                required |= perms
        return required

    def _check_permissions(self, name: str, identity: Identity | None) -> None:
        required = self._required(name)
        if not required:
            return
        if identity is None or identity.permissions is None:
            if self.strict:
                raise ToolDenied("The caller has no permissions for this tool.", tool_name=name)
            return
        missing = sorted(p for p in required if not identity.can(p))
        if missing:
            raise ToolDenied(f"Missing permission: {', '.join(missing)}.", tool_name=name)

    async def _run_check(
        self, name: str, args: dict[str, t.Any], identity: Identity | None
    ) -> None:
        if self.check is None:
            return
        try:
            reason = self.check(name, args, identity)
            if inspect.isawaitable(reason):
                reason = await reason
        except ToolDenied as denied:
            denied.tool_name = denied.tool_name or name
            raise
        except Exception as exc:
            # Fail closed; the exception text may hold argument values, so it is not logged.
            logger.error(
                "ToolGuard check for tool '%s' failed (%s)", log_safe(name), type(exc).__name__
            )
            raise ToolDenied("The argument check failed.", tool_name=name) from exc
        if reason:
            raise ToolDenied(str(reason), tool_name=name)

    def _ask_approval(self, name: str, args: dict[str, t.Any], tool_call_id: t.Any) -> None:
        action = f"Run tool '{name}'"
        try:
            answer = interrupt(
                {"tool_name": name, "args": args, "tool_call_id": tool_call_id, "action": action},
                message=f"Approve: {action}?",
                reason=APPROVAL_REASON,
                response_schema=_APPROVAL_SCHEMA,
            )
        except RuntimeError as exc:
            # Called outside a running graph (a direct ToolNode.invoke): nobody can approve.
            raise ToolDenied("This call needs approval, but the run cannot pause.") from exc
        if not is_approved(answer):
            raise ToolDenied("A human reviewer did not approve this call.", tool_name=name)
