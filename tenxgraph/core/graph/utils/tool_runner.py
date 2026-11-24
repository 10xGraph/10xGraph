"""One runner for the tool calls of a ToolNode, under invoke and stream alike.

``ToolRunner.run`` answers every tool call of the last message: handoff, parallel workers
with branch isolation, the replay-safe tool ledger, ``tool_timeout``, metrics and, when
streaming, live chunks. It yields ``StreamChunk``s (only when streaming) and last a
``ToolRunResult``. Under invoke no chunk, emitter or queue is built: each tool is awaited
directly through ``ToolNode.invoke``.
"""

import asyncio
import copy
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from injectq import Inject

from tenxgraph.core.graph.tool_node.denial import TOOL_DENIED_CODE
from tenxgraph.core.graph.utils.tool_dispatch import (
    allowed_tool_names,
    hidden_tool_message,
    invalid_arguments_message,
    resolve_max_parallel_tool_calls,
)
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ErrorBlock, ToolResultBlock
from tenxgraph.core.state.stream_chunks import StreamChunk, StreamEvent
from tenxgraph.core.state.stream_emitter import StreamEmitter
from tenxgraph.storage.checkpointer import BaseCheckpointer
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.interrupt import GraphInterrupt

from .tool_runtime import ToolRuntimeMixin


logger = logging.getLogger("tenxgraph.graph")


@dataclass(slots=True)
class ToolRunResult:
    """What a ToolNode step produced.

    ``messages`` are the tool results in call order (handoff answers last), ``next_node``
    the handoff target, and ``state_updated`` whether a tool returned a state (already
    merged into the state the runner was given).
    """

    messages: list[Message] = field(default_factory=list)
    next_node: str | None = None
    state_updated: bool = False


class ToolRunner(ToolRuntimeMixin):
    """Runs the tool calls of one ToolNode step (``name``: graph node, ``func``: ToolNode)."""

    def __init__(self, name: str, func: Any):
        self.name = name
        self.func = func

    async def run(
        self, state: AgentState, config: dict[str, Any], *, streaming: bool
    ) -> AsyncIterator[StreamChunk | ToolRunResult]:
        """Answer the tool calls of the state's last message."""
        logger.debug("Node '%s' is a ToolNode, executing tool calls", self.name)
        last_message = self._get_last_tool_message(state)
        async for item in self.call_tools(last_message, state, config, streaming=streaming):
            yield item

    async def call_tools(
        self,
        last_message: Message,
        state: AgentState,
        config: dict[str, Any],
        *,
        streaming: bool,
    ) -> AsyncIterator[StreamChunk | ToolRunResult]:
        """Run the calls of ``last_message``; chunks while they run, the result last."""
        calls = self._tool_calls_or_raise(last_message)

        # A handoff is honoured only for a registered handoff tool. The message's other
        # calls run first, so the target sees their results; every call is answered.
        handoff = self._find_handoff(last_message)
        if handoff:
            target, winner, regular = handoff
            ran = ToolRunResult()
            if regular:
                regular_message = self._regular_calls_message(last_message, regular)
                async for item in self.call_tools(
                    regular_message, state, config, streaming=streaming
                ):
                    if isinstance(item, ToolRunResult):
                        ran = item
                    else:
                        yield item
            answers = self._handoff_answers(last_message, winner, target, regular)
            self._stamp_node_name(answers)
            if streaming:
                # The answers never ran as tools, so no worker queued their chunks.
                for answer in answers:
                    chunk = self._result_chunk(answer, *self._answer_ids(answer), config)
                    if chunk is not None:
                        yield chunk
            yield ToolRunResult([*ran.messages, *answers], target, ran.state_updated)
            return

        logger.info("Node '%s' executing %d tool calls in parallel", self.name, len(calls))
        async for item in self._run_calls(calls, last_message, state, config, streaming):
            yield self._collect(item, state) if isinstance(item, list) else item

    @staticmethod
    def _answer_ids(answer: Message) -> tuple[Any, dict[str, Any], str]:
        """Tool name, args and call id of a handoff answer, for its chunk."""
        block = next((b for b in answer.content or [] if isinstance(b, ToolResultBlock)), None)
        return (getattr(block, "name", None) or "", {}, getattr(block, "call_id", "") or "")

    def _collect(self, outcomes: list[Any], state: AgentState) -> ToolRunResult:
        """Merge the outcomes (call order) into one result; messages deduped by id."""
        result = ToolRunResult()
        seen: set[str | int] = set()
        for outcome in outcomes:
            if isinstance(outcome, dict):
                tool_state = outcome.get("state")
                if isinstance(tool_state, AgentState):
                    if tool_state is not state:
                        self._merge_tool_state(state, tool_state)
                    result.state_updated = True
                messages = self._extract_tool_messages(outcome)
            else:
                messages = [outcome] if isinstance(outcome, Message) else []
            for message in messages:
                if message.message_id not in seen:
                    seen.add(message.message_id)
                    result.messages.append(message)
        self._stamp_node_name(result.messages)
        return result

    async def _run_calls(  # noqa: PLR0915
        self,
        calls: list[dict[str, Any]],
        last_message: Message,
        state: AgentState,
        config: dict[str, Any],
        streaming: bool,
    ) -> AsyncIterator[Any]:
        """Run every call in its own worker task.

        Yields the workers' chunks as they come (streaming only) and, last, the outcomes in
        call order, whichever tool finished first.
        """
        allowed = allowed_tool_names(last_message)
        # The assistant message that issued these calls makes each invocation uniquely
        # identifiable for the idempotency ledger.
        origin_message_id = getattr(last_message, "message_id", None)

        # Branch isolation: parallel tools each get a copy of the state, merged back
        # afterwards against a baseline. A single call keeps the shared state.
        is_parallel = len(calls) > 1
        baseline = copy.deepcopy(state) if is_parallel else None
        branch_states: list[AgentState | None] = [None] * len(calls)
        results: list[Any] = [None] * len(calls)
        interrupts: list[GraphInterrupt] = []
        # At most this many tools run at once. The branch copy is taken inside the slot,
        # so peak copies are bounded by the limit.
        semaphore = asyncio.Semaphore(resolve_max_parallel_tool_calls(config))
        queue: asyncio.Queue | None = asyncio.Queue() if streaming else None
        worker_done = object()

        async def worker(index: int, tool_call: dict[str, Any]) -> None:
            name = tool_call.get("function", {}).get("name", "")
            try:
                if allowed is not None and name not in allowed:
                    logger.warning(
                        "Node '%s': tool '%s' was not offered to the model; not executing",
                        self.name,
                        name,
                    )
                    if queue is not None:
                        reason = "Tool was not offered to the model"
                        queue.put_nowait(
                            self._tool_error_chunk(
                                tool_call, config, reason, f"Tool '{name}' not found."
                            )
                        )
                    results[index] = hidden_tool_message(tool_call)
                    return
                async with semaphore:
                    branch = copy.deepcopy(state) if is_parallel else state
                    branch_states[index] = branch
                    results[index] = await self._handle_single_tool(
                        tool_call, branch, config, origin_message_id=origin_message_id, queue=queue
                    )
                    if is_parallel and self._branch_untouched(branch, baseline, results[index]):
                        branch_states[index] = None
            except GraphInterrupt as paused:
                # A tool that called interrupt() paused the run; not a tool failure.
                interrupts.append(paused)
            except BaseException as e:
                # Every call is answered, also for a non-Exception BaseException, unless
                # this worker itself is being cancelled (run cancel, consumer gone).
                current = asyncio.current_task()
                if isinstance(e, KeyboardInterrupt | SystemExit) or (
                    isinstance(e, asyncio.CancelledError) and current and current.cancelling()
                ):
                    raise
                results[index] = self._failure_message(tool_call, e)
                if queue is not None:
                    queue.put_nowait(
                        self._tool_error_chunk(tool_call, config, "Tool execution failed")
                    )
            finally:
                if queue is not None:
                    queue.put_nowait(worker_done)

        tasks = [asyncio.create_task(worker(i, tc)) for i, tc in enumerate(calls)]
        try:
            if queue is None:
                await asyncio.wait(tasks)
            else:
                remaining = len(tasks)
                while remaining > 0:
                    item = await queue.get()
                    if item is worker_done:
                        remaining -= 1
                    else:
                        yield item
        finally:
            # Cancel and the consumer going away both end here: no tool outlives the run.
            pending = [t for t in tasks if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.wait(pending)

        if interrupts:
            raise interrupts[0]
        outcomes = [r for r in results if r is not None]
        if is_parallel:
            with_state = [
                r for r in results if isinstance(r, dict) and isinstance(r.get("state"), AgentState)
            ]
            # Fold, then drop the state from the payloads so it is not merged twice; point it
            # at the shared state so the step still reports a state update.
            self._fold_parallel_branches(state, baseline, branch_states, results)
            for r in with_state:
                r["state"] = state
        yield outcomes

    async def _handle_single_tool(
        self,
        tool_call: dict[str, Any],
        state: AgentState,
        config: dict[str, Any],
        origin_message_id: str | None = None,
        queue: asyncio.Queue | None = None,
        checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],  # type: ignore[assignment]
    ) -> Any:
        """Run one tool call and return its result; live chunks go to ``queue`` when streaming.

        A call already in the checkpointer's tool ledger is replayed instead of run (the
        loop re-runs a node killed mid-way, and a tool must not fire twice), a finished
        call is recorded at once, before its result leaves the runner, and a
        ``tool_timeout`` cuts the call off.
        """
        checkpointer = fresh(checkpointer)
        function_name = tool_call.get("function", {}).get("name", "")
        tool_call_id = tool_call.get("id", "")

        ledger_key, recorded = await self._ledger_lookup(
            checkpointer, config, origin_message_id, tool_call_id, function_name
        )
        if recorded is not None:
            if queue is not None:
                self._put_result_chunk(queue, recorded, function_name, {}, tool_call_id, config)
            return recorded

        function_args, args_error = self._checked_tool_args(tool_call, function_name)
        if args_error:
            if queue is not None:
                reason = "Tool call arguments were not a valid JSON object"
                queue.put_nowait(self._tool_error_chunk(tool_call, config, reason, args_error))
            return invalid_arguments_message(tool_call)

        emitter = None
        if queue is not None:
            ids: dict[str, Any] = {
                "thread_id": config.get("thread_id"),
                "run_id": config.get("run_id"),
            }
            queue.put_nowait(
                StreamChunk(
                    event=StreamEvent.UPDATES,
                    data={
                        "status": "invoking_tool",
                        "tool_name": function_name,
                        "args": function_args,
                        "tool_call_id": tool_call_id,
                        "node": self.name,
                    },
                    **ids,
                )
            )
            # The tool's own progress chunks (also from sync-tool threads) go to the queue.
            emitter = StreamEmitter(
                tool_name=function_name,
                tool_call_id=tool_call_id,
                node_name=self.name,
                queue=queue,
                loop=asyncio.get_running_loop(),
                **ids,
            )

        result = await self._invoke_tool_metered(
            function_name, function_args, tool_call_id, state, config, emit=emitter
        )
        if checkpointer and ledger_key and result is not None:
            await self._record_tool_result(checkpointer, config, ledger_key, result)
        if queue is not None:
            self._put_result_chunk(
                queue, result, function_name, function_args, tool_call_id, config
            )
        return result

    def _put_result_chunk(
        self,
        queue: asyncio.Queue,
        item: Any,
        function_name: str,
        function_args: dict[str, Any],
        tool_call_id: str,
        config: dict[str, Any],
    ) -> None:
        """Queue the chunk of one tool result, if it has one."""
        chunk = self._result_chunk(item, function_name, function_args, tool_call_id, config)
        if chunk is not None:
            queue.put_nowait(chunk)

    def _result_chunk(
        self,
        item: Any,
        function_name: str,
        function_args: dict[str, Any],
        tool_call_id: str,
        config: dict[str, Any],
    ) -> StreamChunk | None:
        """The chunk a client sees for one tool result (a Message or a state-update dict)."""
        meta = {
            "tool_name": function_name,
            "args": function_args,
            "tool_call_id": tool_call_id,
            "node": self.name,
        }
        ids: dict[str, Any] = {"thread_id": config.get("thread_id"), "run_id": config.get("run_id")}
        message = item.get("messages") if isinstance(item, dict) else item
        if not isinstance(message, Message):
            return None
        error = next((b for b in message.content or [] if isinstance(b, ErrorBlock)), None)
        if error is None or isinstance(item, dict):
            return StreamChunk(
                event=StreamEvent.MESSAGE,
                message=message,
                data={"status": "tool_result", **meta},
                **ids,
            )
        denied = error.code == TOOL_DENIED_CODE
        return StreamChunk(
            event=StreamEvent.ERROR,
            data={
                "status": "tool_denied" if denied else "tool_failed",
                **meta,
                "reason": "Denied by policy" if denied else "Tool execution resulted in error",
                "error": error.message,
            },
            **ids,
        )

    def _tool_error_chunk(
        self,
        tool_call: dict[str, Any],
        config: dict[str, Any],
        reason: str,
        error: str | None = None,
    ) -> StreamChunk:
        """An ERROR chunk for a call that failed before or while it ran."""
        data = {
            "status": "tool_failed",
            "tool_name": tool_call.get("function", {}).get("name", ""),
            "tool_call_id": tool_call.get("id", ""),
            "node": self.name,
            "reason": reason,
        }
        if error is not None:
            data["error"] = error
        return StreamChunk(
            event=StreamEvent.ERROR,
            data=data,
            thread_id=config.get("thread_id"),
            run_id=config.get("run_id"),
        )
