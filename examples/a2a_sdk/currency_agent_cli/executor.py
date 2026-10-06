"""
Custom executor for the CurrencyAgent.

Extends AgentFlowExecutor to emit INPUT_REQUIRED when the LLM asks
for missing information (e.g. "Which currency do you want to convert to?").

Referenced in 10xgraph.json as:
    "executor": "executor:CurrencyAgentExecutor"
"""

from __future__ import annotations

import logging

from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import TaskState, TextPart

from tenxgraph.core.state import Message as AFMessage
from tenxgraph.runtime.protocols.a2a.executor import AgentFlowExecutor
from tenxgraph.utils.constants import ResponseGranularity


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Heuristic: is the LLM asking for more input?                                #
# --------------------------------------------------------------------------- #

_ASKING_PHRASES = [
    "could you",
    "please provide",
    "please specify",
    "what amount",
    "which currency",
    "what currency",
    "let me know",
    "can you tell",
    "i need",
    "please tell",
    "what is the",
    "what date",
]


def _is_asking_for_input(text: str) -> bool:
    low = text.lower().strip()
    return low.endswith("?") or any(p in low for p in _ASKING_PHRASES)


# --------------------------------------------------------------------------- #
#  Executor                                                                     #
# --------------------------------------------------------------------------- #


class CurrencyAgentExecutor(AgentFlowExecutor):
    """Runs the currency graph; emits INPUT_REQUIRED for vague queries."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.submit()
        await updater.start_work()

        try:
            user_text = context.get_user_input() if context.message else ""
            messages = [AFMessage.text_message(user_text, role="user")]
            run_config = {"thread_id": context.context_id or context.task_id or ""}

            result = await self.graph.ainvoke(
                {"messages": messages},
                config=run_config,
                response_granularity=ResponseGranularity.FULL,
            )
            response_text = self._extract_response_text(result)

            if _is_asking_for_input(response_text):
                msg = updater.new_agent_message(parts=[TextPart(text=response_text)])
                await updater.update_status(TaskState.input_required, message=msg)
            else:
                await updater.add_artifact([TextPart(text=response_text)])
                await updater.complete()

        except Exception as exc:
            logger.exception("CurrencyAgentExecutor failed")
            err = updater.new_agent_message(parts=[TextPart(text=f"Error: {exc!s}")])
            await updater.failed(message=err)
