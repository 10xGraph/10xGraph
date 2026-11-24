"""LocalExecMixin — executes locally registered tool functions."""

from __future__ import annotations

import inspect
import logging
import types
import typing as t

from tenxgraph.core.authz import UserCredential, get_credential, public_config
from tenxgraph.core.exceptions import ToolDenied
from tenxgraph.core.state import (
    AgentState,
    ContentBlock,
    ErrorBlock,
    Message,
    ToolResult,
    ToolResultBlock,
)
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.utils import CallbackContext, CallbackManager, InvocationType, call_sync_or_async
from tenxgraph.utils.constants import TRUSTED_TOOL_RESULT_META
from tenxgraph.utils.interrupt import activate as activate_interrupts

from ._helpers import (
    _extract_block_meta,
    _safe_serialize,
    generic_tool_failure,
    publish_tool_end,
    tool_start_event,
)
from .coercion import ToolArgumentError, coerce_tool_argument
from .constants import CREDENTIAL_PARAM, INJECTABLE_PARAMS, SKILL_TOOL_ATTR, has_injected_default
from .denial import denied_tool_message
from .schema import _safe_type_hints


if t.TYPE_CHECKING:
    from tenxgraph.core.state.stream_emitter import StreamEmitter

logger = logging.getLogger("tenxgraph.graph.tool_node")

_CREDENTIAL_HINTS = (UserCredential, UserCredential | None, t.Optional[UserCredential])


def check_credential_annotation(fn: t.Callable) -> None:
    """Warn once when a ``credential`` parameter is annotated as something else.

    The parameter always receives the UserCredential object (never a str or dict, which would
    end up in logs and events), so a different annotation is misleading.
    """
    try:
        param = inspect.signature(fn).parameters.get(CREDENTIAL_PARAM)
    except (TypeError, ValueError):
        return
    if param is None or param.annotation is inspect.Parameter.empty:
        return
    hint = _safe_type_hints(fn).get(CREDENTIAL_PARAM, param.annotation)
    if hint in _CREDENTIAL_HINTS:
        return
    if t.get_origin(hint) in (t.Union, types.UnionType) and set(t.get_args(hint)) == {
        UserCredential,
        type(None),
    }:
        return
    logger.warning(
        "Tool '%s': the annotation of parameter 'credential' (%s) is ignored; it always "
        "receives a UserCredential, or None when the run has none.",
        getattr(fn, "__name__", fn),
        hint,
    )


class LocalExecMixin:
    _funcs: dict[str, t.Callable]

    def _prepare_input_data_tool(
        self,
        fn: t.Callable,
        name: str,
        args: dict,
        default_data: dict,
    ) -> dict:
        sig = inspect.signature(fn)
        hints = _safe_type_hints(fn)
        input_data = {}
        for param_name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            if param_name in default_data:
                input_data[param_name] = default_data[param_name]
                continue

            if param_name in INJECTABLE_PARAMS:
                continue

            if has_injected_default(param):
                logger.debug(
                    "Skipping injectable parameter '%s' with Inject syntax",
                    param_name,
                )
                continue

            if param_name in args:
                # The provider returns plain JSON, so a model/dataclass/enum parameter
                # arrives as a dict or str and must be validated into its real type.
                input_data[param_name] = coerce_tool_argument(
                    args[param_name],
                    hints.get(param_name, param.annotation),
                    tool_name=name,
                    param_name=param_name,
                )
            elif param.default is inspect.Parameter.empty:
                raise ToolArgumentError(
                    f"Missing required parameter '{param_name}' for function '{name}'"
                )

        return input_data

    def _build_internal_result_blocks(
        self,
        result: t.Any,
        tool_call_id: str,
        name: str | None = None,
    ) -> list[ContentBlock]:
        is_error = False

        if isinstance(result, str):
            cap = getattr(self, "_max_result_chars", None)
            if cap is not None and len(result) > cap:
                result = result[:cap] + f"\n[output cut: {len(result) - cap} more characters]"
            output: str | list[dict[str, t.Any]] = result
        elif isinstance(result, dict):
            is_error = _extract_block_meta(result)
            output = [_safe_serialize(dict(result))]
        elif hasattr(result, "model_dump"):
            dumped = result.model_dump()  # type: ignore
            if isinstance(dumped, dict):
                is_error = _extract_block_meta(dumped)
                output = [_safe_serialize(dumped)]
            else:
                output = [_safe_serialize(dumped)]
        elif hasattr(result, "__dict__"):
            is_error = _extract_block_meta(result.__dict__)
            output = [_safe_serialize(dict(result.__dict__))]
        elif isinstance(result, list):
            output = [_safe_serialize(item) for item in result]
        else:
            output = str(result)

        return [
            ToolResultBlock(
                call_id=tool_call_id,
                name=name,
                output=output,
                status="failed" if is_error else "completed",
                is_error=is_error,
            )
        ]

    def _message_from_internal_result(
        self,
        result: t.Any,
        state: AgentState,
        tool_call_id: str,
        meta: dict[str, t.Any],
        name: str | None = None,
    ) -> dict[str, t.Any] | Message:
        if isinstance(result, ToolResult):
            if result.state:
                for key, value in result.state.items():
                    if hasattr(state, key):
                        setattr(state, key, value)

            return {
                "state": state,
                "messages": Message.tool_message(
                    content=[
                        ToolResultBlock(
                            call_id=tool_call_id,
                            name=name,
                            output=result.message,
                            status="failed" if result.is_error else "completed",
                            is_error=result.is_error,
                        )
                    ],
                    meta=meta,
                ),
            }

        if isinstance(result, Message):
            result.metadata = {**meta, **(result.metadata or {})}
            return result

        return Message.tool_message(
            content=self._build_internal_result_blocks(result, tool_call_id, name),
            meta=meta,
        )

    async def _internal_execute(
        self,
        name: str,
        args: dict,
        tool_call_id: str,
        config: dict[str, t.Any],
        state: AgentState,
        callback_mgr: CallbackManager,
        emit: StreamEmitter | None = None,
    ) -> dict[str, t.Any] | Message:
        fn = self._funcs[name]
        # Skill tools (activate_skill / read_skill_resource) report as SKILL
        # invocations so callbacks can tell them apart from ordinary tools.
        invocation_type = (
            InvocationType.SKILL if getattr(fn, SKILL_TOOL_ATTR, False) else InvocationType.TOOL
        )
        context = CallbackContext(
            invocation_type=invocation_type,
            node_name=config.get("_node_name", ""),
            function_name=name,
            metadata={
                "tool_call_id": tool_call_id,
                "args": args,
                "config": public_config(config),
            },
        )
        # The model's raw arguments until they are validated, so an error callback still has
        # what was asked for when validation is what failed.
        input_data: dict[str, t.Any] = dict(args)

        meta: dict[str, t.Any] = {
            "function_name": name,
            "function_argument": args,
            "tool_call_id": tool_call_id,
        }
        if invocation_type == InvocationType.SKILL:
            # Skill text is the developer's instructions, so it reaches the model unfenced.
            meta[TRUSTED_TOOL_RESULT_META] = True

        event = tool_start_event(config, name, args, tool_call_id, is_mcp=False)
        publish_event(event)

        try:
            # Inside the try: a call missing a required argument is the model's mistake to
            # correct, so it comes back as a failed tool result like any other tool error,
            # rather than failing the whole graph run.
            input_data = self._prepare_input_data_tool(
                fn,
                name,
                args,
                {
                    "tool_call_id": tool_call_id,
                    "state": state,
                    "config": config,
                    "emit": emit,
                },
            )
            # before_invoke runs in the call's interrupt scope too, so a callback that asks a
            # human (ToolGuard approval) pauses keyed to this call, like interrupt() in the tool.
            with activate_interrupts(config, tool_call_id):
                input_data = await callback_mgr.execute_before_invoke(context, input_data)

            # Added after the callbacks, so their input_data never holds the credential.
            call_data = dict(input_data)
            call_data.pop(CREDENTIAL_PARAM, None)
            if CREDENTIAL_PARAM in inspect.signature(fn).parameters:
                call_data[CREDENTIAL_PARAM] = get_credential(config)

            # interrupt() inside the tool is keyed to this call, so parallel calls resume apart.
            with activate_interrupts(config, tool_call_id):
                result = await call_sync_or_async(fn, **call_data)

            result = await callback_mgr.execute_after_invoke(
                context,
                input_data,
                result,
            )

            msg = self._message_from_internal_result(result, state, tool_call_id, meta, name)
            done = msg.get("messages") if isinstance(msg, dict) else msg
            if isinstance(done, Message):
                publish_tool_end(event, "Internal tool execution complete", message=done)
            return msg

        except ToolDenied as denied:
            # A policy block is final: on_error recovery must not turn it into a success.
            return denied_tool_message(denied, name, tool_call_id, config, event, meta)

        except Exception as e:
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if isinstance(recovery_result, Message):
                publish_tool_end(
                    event,
                    "Internal tool execution complete, with recovery",
                    message=recovery_result,
                )
                return recovery_result

            # Argument errors are the model's to correct and carry no server detail. Anything
            # else may hold paths, DSNs or SQL, so the model and the event get a reference.
            if isinstance(e, ToolArgumentError):
                error_text = f"Internal execution error: {e}"
            else:
                error_text = generic_tool_failure(name, tool_call_id, e, log=logger)

            failed = Message.tool_message(
                content=[
                    ToolResultBlock(
                        call_id=tool_call_id,
                        name=name,
                        output=error_text,
                        status="failed",
                        is_error=True,
                    ),
                    ErrorBlock(message=error_text),
                ],
                meta=meta,
            )
            publish_tool_end(
                event,
                "Internal tool execution complete, with error",
                message=failed,
                error=error_text,
                exc=e,
                config=config,
            )
            return failed
