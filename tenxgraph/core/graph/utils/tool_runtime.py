"""Tool-call machinery: the ledger, branch merge and tool-call helpers.

A durable ledger of finished calls (so a replayed node does not fire a tool twice), isolated
state per parallel branch, and a deterministic merge of the branches back into the shared
state. ``ToolRunner`` runs a ToolNode step with it; ``LiveAgent`` uses the ledger helpers and
:meth:`ToolRuntimeMixin._invoke_tool_metered` for realtime tool calls.
"""

import logging
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any

from tenxgraph.core.exceptions import NodeError, NodeTimeoutError
from tenxgraph.core.graph.tool_node._helpers import generic_tool_failure
from tenxgraph.core.graph.utils.guards import execute_with_guards, resolve_timeout
from tenxgraph.core.graph.utils.tool_dispatch import (
    allowed_tool_names,
    handoff_target,
    handoff_tool_messages,
    parse_call_arguments,
)
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.core.state.stream_emitter import StreamEmitter
from tenxgraph.storage.checkpointer import BaseCheckpointer
from tenxgraph.utils import metrics
from tenxgraph.utils.constants import DEFAULT_TOOL_TIMEOUT_SECONDS
from tenxgraph.utils.log_safety import log_safe


logger = logging.getLogger("tenxgraph.graph")


async def _last_item(items: AsyncIterator[Any]) -> Any:
    """The last item of an async iterator (``ToolNode.stream`` yields its result once)."""
    last = None
    async for last in items:  # noqa: B007
        pass
    return last


class ToolRuntimeMixin:
    """Ledger and parallel-merge helpers. The host provides ``name`` and ``func``."""

    name: str
    func: Any

    def _find_handoff(self, last_message: Message) -> tuple[str, int, list[dict]] | None:
        """Detect a handoff in ``last_message``.

        Returns ``(target_agent, winner, regular_calls)`` or ``None`` when there is no
        handoff. ``winner`` is the index of the first registered handoff call (the first
        handoff wins); ``regular_calls`` are the other offered calls that are not handoffs,
        which run before control moves on (a model that asks for ``get_weather`` and
        ``transfer_to_planner`` together wants both). Answer the rest with
        :meth:`_handoff_answers`.
        """
        calls = last_message.tools_calls or []
        allowed = allowed_tool_names(last_message)
        target: str | None = None
        winner = -1
        regular: list[dict] = []
        for index, call in enumerate(calls):
            name = call.get("function", {}).get("name", "")
            if allowed is not None and name not in allowed:
                continue
            call_target = handoff_target(self.func, name)
            if call_target and target is None:
                target, winner = call_target, index
            elif not call_target:
                regular.append(call)
        if not target:
            return None
        logger.info(
            "Handoff detected in node '%s': tool '%s' -> agent '%s'",
            self.name,
            calls[winner].get("function", {}).get("name", ""),
            target,
        )
        return target, winner, regular

    @staticmethod
    def _handoff_answers(
        last_message: Message, winner: int, target: str, regular: list[dict]
    ) -> list[Message]:
        """Tool results for the calls of a handoff message that did not run as tools.

        The handoff call gets ``handoff_result_text(target)``, a second handoff call gets a
        "not run" result, and the regular calls are skipped: they ran before the handoff.
        """
        ran = {call.get("id", "") for call in regular}
        return handoff_tool_messages(last_message, winner, target, ran=ran)

    @staticmethod
    def _regular_calls_message(last_message: Message, regular: list[dict]) -> Message:
        """``last_message`` with only ``regular`` calls, to run them before the handoff.

        Same message id, so the idempotency ledger keys stay those of the real message.
        """
        return last_message.model_copy(update={"tools_calls": regular})

    @staticmethod
    def _ledger_enabled(checkpointer: Any, tools: Any, function_name: str) -> bool:
        """Whether a call of ``function_name`` is read from and written to the tool ledger.

        Off when there is no checkpointer, when the checkpointer has
        ``record_tool_result=False`` (this wins), or when the local tool was decorated with
        ``@tool(record_tool_result=False)``. MCP, skill and client tools carry no flag and
        are recorded.
        """
        if not checkpointer or not getattr(checkpointer, "record_tool_result", True):
            return False
        funcs = getattr(tools, "_funcs", None)
        fn = funcs.get(function_name) if isinstance(funcs, dict) else None
        return getattr(fn, "_py_tool_record_tool_result", True) is not False

    @staticmethod
    def _ledger_key(origin_message_id: str | None, tool_call_id: str) -> str | None:
        """Identity of one tool invocation, for the idempotency ledger.

        A ``tool_call_id`` alone is NOT unique over a thread's lifetime -- models
        reuse ids like ``call_1`` on every turn. Keying the ledger on it alone
        would make a later turn collide with an earlier one and silently skip a
        tool that had never run.

        What uniquely identifies an invocation is the assistant message that
        issued it plus the call id. That message is persisted, so on replay it
        carries the same ``message_id`` and the key is stable -- which is exactly
        what makes replay detection work, while different turns never collide.
        """
        if not tool_call_id or not origin_message_id:
            return None
        return f"{origin_message_id}:{tool_call_id}"

    async def _invoke_tool_metered(
        self,
        function_name: str,
        function_args: dict,
        tool_call_id: str,
        state: AgentState,
        config: dict[str, Any],
        emit: StreamEmitter | None = None,
        **invoke_kwargs: Any,
    ) -> Any:
        """Run one tool call under ``tool_timeout``, recording the tool metrics.

        With an ``emit`` (streaming) the call goes through ``ToolNode.stream`` so the tool
        can report progress; without one through ``ToolNode.invoke``.

        Raises ``NodeTimeoutError`` when the deadline passes (the call is cancelled), or
        whatever the tool raised; :meth:`_failure_message` turns either into the result
        the model sees.
        """
        kwargs = {"tool_call_id": tool_call_id, "state": state, "config": config}
        if emit is None:
            call = self.func.invoke(function_name, function_args, **kwargs, **invoke_kwargs)
        else:
            call = _last_item(self.func.stream(function_name, function_args, **kwargs, emit=emit))
        timeout, tool_attrs = self._start_tool_metrics(function_name, config)
        with self._tool_metrics(tool_attrs):
            return await execute_with_guards(
                call,
                timeout=timeout,
                on_timeout=lambda: self._tool_timeout_error(function_name, timeout),
            )

    def _start_tool_metrics(
        self, function_name: str, config: dict[str, Any]
    ) -> tuple[float | None, dict[str, str]]:
        """Count one tool call. Returns its ``tool_timeout`` and metric attributes."""
        timeout = resolve_timeout(config, "tool_timeout", DEFAULT_TOOL_TIMEOUT_SECONDS)
        # The name comes from the model. Only a tool this node can run is a safe metric
        # label; anything else would create a new series per made-up name.
        known = self.func.knows_tool(function_name, config)
        tool_attrs = {"node": self.name, "tool": function_name if known else "unknown"}
        metrics.counter("tenxgraph.tool.calls").inc(attributes=tool_attrs)
        return timeout, tool_attrs

    @staticmethod
    @contextmanager
    def _tool_metrics(tool_attrs: dict[str, str]) -> Iterator[None]:
        """Time one tool call and count its timeouts and errors."""
        with metrics.timer("tenxgraph.tool.duration", attributes=tool_attrs):
            try:
                yield
            except NodeTimeoutError:
                metrics.counter("tenxgraph.tool.timeouts").inc(attributes=tool_attrs)
                raise
            except Exception:
                metrics.counter("tenxgraph.tool.errors").inc(attributes=tool_attrs)
                raise

    def _tool_timeout_error(self, function_name: str, timeout: float | None) -> NodeTimeoutError:
        return NodeTimeoutError(
            message=f"Tool '{function_name}' exceeded its timeout of {timeout}s and was cancelled",
            error_code="NODE_TIMEOUT_002",
            context={"node_name": self.name, "tool_name": function_name, "timeout": timeout},
        )

    async def _ledger_lookup(
        self,
        checkpointer: Any,
        config: dict[str, Any],
        origin_message_id: str | None,
        tool_call_id: str,
        function_name: str,
    ) -> tuple[str | None, Any]:
        """The call's ledger key, and its recorded result when it already ran.

        The key is None when the ledger is off for this call, which turns both the replay
        lookup and the record after the call off. A recorded result is reused instead of
        running the tool a second time.
        """
        ledger_key = (
            self._ledger_key(origin_message_id, tool_call_id)
            if self._ledger_enabled(checkpointer, self.func, function_name)
            else None
        )
        if not (checkpointer and ledger_key):
            return ledger_key, None
        recorded = await self._get_recorded_tool_result(checkpointer, config, ledger_key)
        if recorded is not None:
            logger.info(
                "Node '%s': tool '%s' (call %s) already executed; "
                "replaying recorded result instead of re-running it",
                self.name,
                log_safe(function_name),
                log_safe(tool_call_id),
            )
        return ledger_key, recorded

    def _checked_tool_args(
        self, tool_call: dict[str, Any], function_name: str
    ) -> tuple[dict[str, Any], str | None]:
        """Parse the call's arguments. Returns them and an error when they are unusable.

        An LLM can emit malformed arguments (cut off at the token limit, empty, not an
        object). That is the model's mistake: the caller reports it back as a tool error
        and the tool never runs with guessed arguments.
        """
        function_args, args_error = parse_call_arguments(tool_call)
        if args_error:
            logger.warning(
                "Node '%s': tool '%s' called with invalid arguments: %s",
                self.name,
                log_safe(function_name),
                log_safe(args_error, 500),
            )
            return function_args, args_error
        logger.info(
            "Node '%s' executing tool '%s' with %d arguments",
            self.name,
            log_safe(function_name),
            len(function_args),
        )
        logger.debug("Tool arguments: %s", function_args)
        return function_args, None

    def _tool_calls_or_raise(self, last_message: Message) -> list[dict[str, Any]]:
        """The tool calls of ``last_message``; a ToolNode with none to run is an error."""
        logger.debug("Node '%s' calling tools from message", self.name)
        if (
            hasattr(last_message, "tools_calls")
            and last_message.tools_calls
            and len(last_message.tools_calls) > 0
        ):
            return last_message.tools_calls
        logger.exception("Node '%s': No tool calls to execute", self.name)
        raise NodeError(
            message="No tool calls to execute",
            error_code="NODE_002",
            context={"node_name": self.name},
        )

    async def _get_recorded_tool_result(
        self,
        checkpointer: "BaseCheckpointer",
        config: dict[str, Any],
        tool_call_id: str,
    ) -> Message | dict[str, Any] | None:
        """Load a previously recorded tool result and rehydrate it."""
        try:
            recorded = await checkpointer.aget_tool_result(config, tool_call_id)
        except Exception as e:  # a ledger read must never take the run down
            logger.warning(
                "Node '%s': could not read tool ledger for call %s (%s); the tool will run again",
                self.name,
                tool_call_id,
                e,
            )
            return None

        # Only a well-formed ledger entry counts as a hit. Anything else (a
        # backend returning an unexpected shape, a test double) must be treated as
        # "no record" and the tool run normally -- silently skipping a tool because
        # we misread the ledger would be far worse than running it.
        if not isinstance(recorded, dict):
            return None

        kind = recorded.get("__kind__")
        if kind == "message":
            try:
                return Message.model_validate(recorded["value"])
            except Exception as e:
                logger.warning(
                    "Node '%s': tool ledger entry for call %s is unreadable (%s); "
                    "the tool will run again",
                    self.name,
                    tool_call_id,
                    e,
                )
                return None
        if kind == "raw":
            return recorded.get("value")
        return None

    async def _record_tool_result(
        self,
        checkpointer: "BaseCheckpointer",
        config: dict[str, Any],
        tool_call_id: str,
        tool_result: Any,
    ) -> None:
        """Durably record a completed tool call against its tool_call_id."""
        if isinstance(tool_result, Message):
            payload = {"__kind__": "message", "value": tool_result.model_dump(mode="json")}
        else:
            payload = {"__kind__": "raw", "value": tool_result}

        try:
            await checkpointer.aput_tool_result(config, tool_call_id, payload)
        except Exception as e:
            # The tool already ran. We could not record that, so a replay of this
            # node would run it again. Surface it loudly rather than pretending
            # the run is idempotent when it is not.
            logger.error(
                "Node '%s': tool call %s COMPLETED but could not be recorded (%s). "
                "If this run is replayed, the tool may execute a second time.",
                self.name,
                tool_call_id,
                e,
            )

    @staticmethod
    def _branch_untouched(branch: AgentState, baseline: AgentState | None, outcome: Any) -> bool:
        """True when a parallel branch has nothing to merge back."""
        if baseline is None:
            return False
        if isinstance(outcome, dict) and isinstance(outcome.get("state"), AgentState):
            return False
        try:
            return bool(branch == baseline)
        except Exception:
            return False

    @staticmethod
    def _get_field_reducer(state: AgentState, field_name: str) -> Callable | None:
        """Return the reducer annotated on a state field, if it has one.

        Reducers are declared as Annotated metadata, e.g.
        ``context: Annotated[list[Message], add_messages]``. When two parallel
        branches both change a field, the reducer is how the framework is told to
        combine them rather than pick a winner.
        """
        field = type(state).model_fields.get(field_name)
        for meta in getattr(field, "metadata", None) or []:
            if callable(meta):
                return meta
        return None

    def _merge_tool_state(
        self,
        target_state: AgentState,
        tool_state: AgentState,
        baseline: AgentState | None = None,
    ) -> None:
        """Merge one tool branch's state into the target.

        Previously every parallel tool mutated ONE shared state object and this
        method blindly `setattr` every field from it, so two tools touching
        different fields still clobbered each other: whichever merged last wrote
        back its own (stale) copy of the field the other had just changed.
        Non-deterministic lost updates.

        Now each branch runs on its own copy, and merging is field-by-field
        against the pre-tool `baseline`:

        - a field the branch did NOT change is skipped, so it can never overwrite
          a sibling's write;
        - a field that changed and has a reducer is combined with the reducer;
        - a field that changed with no reducer is written, and if a sibling also
          changed it we warn -- that is a genuine conflict only the developer can
          resolve (by declaring a reducer).
        """
        for field_name in type(tool_state).model_fields:
            if field_name in ("context", "context_summary", "execution_meta"):
                continue

            new_value = getattr(tool_state, field_name, None)

            if baseline is not None:
                original = getattr(baseline, field_name, None)
                if new_value == original:
                    # This branch left the field alone; leave a sibling's write intact.
                    continue

                reducer = self._get_field_reducer(tool_state, field_name)
                if reducer is not None:
                    setattr(
                        target_state,
                        field_name,
                        reducer(getattr(target_state, field_name), new_value),
                    )
                    continue

                current = getattr(target_state, field_name, None)
                if current != original:
                    logger.warning(
                        "Node '%s': parallel tools both wrote state field '%s' and it has "
                        "no reducer; keeping the last write. Annotate the field with a "
                        "reducer to make this deterministic.",
                        self.name,
                        field_name,
                    )

            setattr(target_state, field_name, new_value)

        if baseline is not None:
            self._merge_internal_data(target_state, tool_state, baseline)

    @staticmethod
    def _merge_internal_data(
        target_state: AgentState,
        tool_state: AgentState,
        baseline: AgentState,
    ) -> None:
        """Fold a parallel branch's ``execution_meta.internal_data`` writes back.

        ``execution_meta`` itself is never merged wholesale (it tracks the run),
        but tools may record data in ``internal_data`` (for example activated
        skills). Keys the branch left alone are skipped. A list the branch only
        appended to has the new items appended to the target, so two branches
        appending to the same list both survive; any other change is written.
        """
        branch_data = tool_state.execution_meta.internal_data
        base_data = baseline.execution_meta.internal_data
        target_data = target_state.execution_meta.internal_data

        for key, new_value in branch_data.items():
            original = base_data.get(key)
            if new_value == original:
                continue
            current = target_data.get(key)
            if (
                isinstance(new_value, list)
                and isinstance(original, list)
                and isinstance(current, list)
                and new_value[: len(original)] == original
            ):
                current.extend(new_value[len(original) :])
            elif isinstance(new_value, list) and original is None and isinstance(current, list):
                current.extend(new_value)
            else:
                target_data[key] = new_value

    def _extract_tool_messages(self, result_item: dict[str, Any]) -> list[Message]:
        """Extract tool messages from either legacy or normalized payload keys."""
        payload = result_item.get("messages", result_item.get("message"))

        if isinstance(payload, Message):
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, Message)]
        return []

    def _stamp_node_name(self, item: Any, roles: tuple[str, ...] = ("tool",)) -> Any:
        """Record the producing node on tool result messages (``metadata["node_name"]``).

        Assistant replies already carry it; this gives tool results the same label so
        clients can group a turn by node. Existing values are never overwritten.
        """
        if isinstance(item, Message):
            if item.role in roles:
                item.metadata.setdefault("node_name", self.name)
        elif isinstance(item, dict):
            for message in self._extract_tool_messages(item):
                self._stamp_node_name(message, roles)
        elif isinstance(item, list):
            for entry in item:
                self._stamp_node_name(entry, roles)
        return item

    def _failure_message(self, tool_call: dict[str, Any], outcome: BaseException) -> Message:
        """Turn a tool that raised into the error result the model sees."""
        fn_name = tool_call.get("function", {}).get("name", "")
        call_id = tool_call.get("id", "")
        if isinstance(outcome, NodeTimeoutError):
            # Names only the tool and its deadline, so the model may see it.
            logger.error("Node '%s': tool '%s' failed: %s", self.name, fn_name, outcome)
            failure = f"Tool '{fn_name}' failed: {outcome}"
        else:
            failure = generic_tool_failure(fn_name, call_id, outcome, log=logger)
        return Message.tool_message(
            content=[
                ToolResultBlock(
                    call_id=call_id,
                    name=fn_name or None,
                    output=failure,
                    is_error=True,
                    status="failed",
                )
            ],
        )

    def _fold_parallel_branches(
        self,
        state: AgentState,
        baseline: AgentState | None,
        branch_states: list[AgentState | None],
        result: list[Any],
    ) -> None:
        """Fold each parallel branch back into the shared state, in call order.

        A tool may either have mutated its injected branch in place, or returned a state
        explicitly; prefer the returned one, else use the branch object. The state is then
        dropped from the result payloads so the downstream normalizer cannot merge it a
        SECOND time -- a double apply through a reducer would duplicate the branch's changes.
        """
        for branch, outcome in zip(branch_states, result, strict=True):
            effective = branch
            if isinstance(outcome, dict) and isinstance(outcome.get("state"), AgentState):
                effective = outcome["state"]
            if effective is None:
                continue
            self._merge_tool_state(state, effective, baseline=baseline)

        for outcome in result:
            if isinstance(outcome, dict):
                outcome.pop("state", None)

    def _get_last_tool_message(self, state: AgentState) -> Message:
        """Return the latest context message required for tool execution."""
        if state.context and len(state.context) > 0:
            return state.context[-1]

        error_msg = "No context available for tool execution"
        logger.error("Node '%s': %s", self.name, error_msg)
        raise NodeError(
            message=error_msg,
            error_code="NODE_003",
            context={"node_name": self.name},
        )
