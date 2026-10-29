"""Tool-call machinery shared by the invoke and stream node handlers.

Both paths run the same tool calls the same way: a durable ledger of finished calls (so a
replayed node does not fire a tool twice), isolated state per parallel branch, and a
deterministic merge of the branches back into the shared state. One copy, used by
``InvokeNodeHandler`` and ``StreamNodeHandler``.
"""

import logging
from collections.abc import Callable
from typing import Any

from tenxgraph.core.exceptions import NodeTimeoutError
from tenxgraph.core.graph.tool_node._helpers import generic_tool_failure
from tenxgraph.core.graph.utils.tool_dispatch import (
    allowed_tool_names,
    handoff_target,
    hidden_tool_message,
)
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.storage.checkpointer import BaseCheckpointer


logger = logging.getLogger("tenxgraph.graph")


class ToolRuntimeMixin:
    """Ledger and parallel-merge helpers. The host class provides ``self.name``."""

    name: str
    func: Any

    def _find_handoff(self, last_message: Message) -> tuple[str, list[Message]] | None:
        """Detect a handoff in ``last_message`` and answer every tool call of it.

        Returns ``(target_agent, tool_messages)`` or ``None`` when there is no handoff.
        Providers reject a history in which an assistant tool call has no tool result, so
        the target agent must not inherit one: the handoff call gets "Transferred to
        <agent>", and every other call of the same message gets a result saying it was not
        run (the first handoff wins). Results come in call order.
        """
        calls = last_message.tools_calls or []
        allowed = allowed_tool_names(last_message)
        target: str | None = None
        winner = -1
        for index, call in enumerate(calls):
            name = call.get("function", {}).get("name", "")
            if allowed is not None and name not in allowed:
                continue
            target = handoff_target(self.func, name)
            if target:
                winner = index
                break
        if not target:
            return None
        logger.info(
            "Handoff detected in node '%s': tool '%s' -> agent '%s'",
            self.name,
            calls[winner].get("function", {}).get("name", ""),
            target,
        )
        messages: list[Message] = []
        for index, call in enumerate(calls):
            name = call.get("function", {}).get("name", "")
            if allowed is not None and name not in allowed:
                messages.append(hidden_tool_message(call))
                continue
            output = (
                f"Transferred to {target}."
                if index == winner
                else f"Not run: control was transferred to {target}."
            )
            messages.append(
                Message.tool_message(
                    content=[
                        ToolResultBlock(
                            call_id=call.get("id", ""), output=output, status="completed"
                        )
                    ],
                )
            )
        return target, messages

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
                ToolResultBlock(call_id=call_id, output=failure, is_error=True, status="failed")
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
