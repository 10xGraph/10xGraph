"""PlanActReflectAgent — a self-contained Plan → Act → Reflect looping agent.

Pattern::

    PLAN --[tool calls?]--> ACT --> REFLECT --[done or max_iters?]--> END
        \\                                  \\
         +--> (no tools) --> REFLECT         +--> PLAN (iterate)

Three purpose-specific ``Agent`` instances are created internally — planner,
actor (via ToolNode), and reflector.  All three share the same *model* but
each has its own default system prompt that can be overridden.

The REFLECT node signals task completion by including ``[DONE]`` anywhere in
its response (case-insensitive); the token is removed from the saved reply.  The
reflector's default system prompt instructs it to emit this token automatically.
If ``max_iterations`` is reached without a ``[DONE]`` signal the graph ends gracefully
and returns the last context.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from injectq import InjectQ

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.compiled_graph import CompiledGraph
from tenxgraph.core.graph.state_graph import StateGraph
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.skills.models import SkillConfig
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.base_context import BaseContextManager
from tenxgraph.core.state.execution_state import register_per_turn_key
from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import ReasoningBlock, TextBlock
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.storage.checkpointer.base_checkpointer import BaseCheckpointer
from tenxgraph.storage.media.config import MultimodalConfig
from tenxgraph.storage.media.storage.base import BaseMediaStore
from tenxgraph.storage.media.url_policy import MediaFetchPolicy
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.storage.store.memory_config import MemoryConfig
from tenxgraph.utils.callbacks import CallbackManager
from tenxgraph.utils.constants import END
from tenxgraph.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


logger = logging.getLogger("tenxgraph.prebuilt.plan_act_reflect")

StateT = TypeVar("StateT", bound=AgentState)

# Key stored in execution_meta.internal_data
_ITERATIONS_KEY = register_per_turn_key("par_iterations")
_INCREMENT_ITERATIONS_NODE = "INCREMENT_ITERATIONS"
# Set when the reflector's reply carried the [DONE] token (dropped at the start of each turn).
_DONE_KEY = register_per_turn_key("par_done")
_DONE_PATTERN = re.compile(r"\s*\[done\]\s*", re.IGNORECASE)

# Transient user turns. A request that ends on an assistant turn makes some providers
# (Gemini) answer with nothing and others (Anthropic) drop that turn, so the reflector and a
# later-iteration planner each get one closing user message. They are added to the request
# view only and never saved to state.
REFLECT_REQUEST_NOTE = (
    "Review the work above against my latest request. If it is complete, or the reply asks "
    "me for information that is needed, or my message needs no further work (a greeting or "
    "small talk), reply with a one-line verdict ending in [DONE]; do not repeat the answer, "
    "the reply above is already my answer. Otherwise list what is missing and what the next "
    "planning step should do, without [DONE]."
)
PLAN_CONTINUE_NOTE = "Continue the task using the review above."

# ---------------------------------------------------------------------------
# Default system prompts
# ---------------------------------------------------------------------------

DEFAULT_PLAN_SYSTEM_PROMPT: list[dict[str, str]] = [
    {
        "role": "system",
        "content": (
            "You are a strategic planner. Your job is to break the user's task "
            "into clear, actionable steps and make concrete progress toward it.\n\n"
            "When a step requires external information or actions, call the "
            "appropriate tools with precise parameters.\n"
            "When you can make progress without tools, provide your analysis or "
            "partial answer directly.\n"
            "Be concise and focused — each response should advance the task."
        ),
    }
]

DEFAULT_REFLECT_SYSTEM_PROMPT: list[dict[str, str]] = [
    {
        "role": "system",
        "content": (
            "You are a critical evaluator. Review all work done so far — the "
            "plan, the actions taken, and any results obtained — and decide "
            "whether the original task is fully complete.\n\n"
            "If the task IS complete:\n"
            "  • Write a one-line verdict only. Do not repeat or rewrite the answer: the "
            "planner's reply is the user's answer.\n"
            "  • End your response with the exact token: [DONE]\n\n"
            "Treat the task as complete when the reply asks the user for information that "
            "is needed to go on, or when the user's message is a greeting or small talk that "
            "needs no further work. In those cases write a one-line verdict and end it "
            "with [DONE].\n\n"
            "If the task is NOT complete:\n"
            "  • Identify the specific gaps or unresolved sub-tasks.\n"
            "  • Give clear, actionable guidance for the next planning step.\n"
            "  • Do NOT include [DONE] in your response."
        ),
    }
]


# ---------------------------------------------------------------------------
# Lightweight increment nodes
# ---------------------------------------------------------------------------


def _has_done_token(text: str) -> bool:
    return _DONE_PATTERN.search(text) is not None


def _strip_done_token(message: Message) -> None:
    """Remove ``[DONE]`` (and the whitespace around it) from the message's text blocks."""
    for block in message.content or []:
        if not isinstance(block, TextBlock):
            continue
        text = block.text

        def _drop(match: re.Match[str], text: str = text) -> str:
            inside = match.start() > 0 and match.end() < len(text)
            return " " if inside else ""

        block.text = _DONE_PATTERN.sub(_drop, text).strip()


def _is_empty_reply(message: Message) -> bool:
    """True for an assistant reply with no text, no tool calls and no media."""
    if message.role != "assistant" or message.tools_calls:
        return False
    if message.text().strip():
        return False
    blocks = message.content if isinstance(message.content, list) else []
    return all(isinstance(b, TextBlock | ReasoningBlock) for b in blocks)


def _planner_left_text(context: list[Message]) -> bool:
    """True when the planner's last reply of the turn has text (not only tool calls)."""
    for msg in reversed(context):
        if msg.role == "user":
            return False
        if msg.role == "assistant" and msg.metadata.get("node_name") != "REFLECT":
            return bool(msg.text().strip())
    return False


def _make_reflect_node(reflect_agent: Agent) -> Callable:
    """Wrap *reflect_agent* so it only sees non-tool messages.

    Tool results (role="tool") are noisy and can overflow context on long runs.
    The wrapper hides them from the reflector's context view while keeping them
    intact in the shared state so downstream PLAN steps still have full history.

    The view ends with one transient user message (``REFLECT_REQUEST_NOTE``) so the
    request never ends on an assistant turn. When the reflector's reply carries
    ``[DONE]``, the token is removed from the saved message and a per-turn flag is set.
    Every reflector message is marked ``metadata["internal"] = True`` (a review step, not
    the answer), except a done verdict when the planner's last reply has no text: then the
    verdict is the turn's only text answer.
    """

    async def _reflect(state: AgentState, config: dict) -> object:
        original_context = state.context
        planner_has_text = _planner_left_text(list(original_context or []))
        # Hide tool results, and the tool calls they answer: a provider rejects an
        # assistant tool call that has no matching result.
        visible: list[Message] = []
        for msg in original_context or []:
            if msg.role == "tool":
                continue
            if msg.role == "assistant" and msg.tools_calls:
                if not msg.text().strip():
                    continue
                msg = msg.model_copy(update={"tools_calls": None})  # noqa: PLW2901
            visible.append(msg)
        visible.append(Message.text_message(REFLECT_REQUEST_NOTE, role="user"))
        state.context = visible
        try:
            result = await reflect_agent.execute(state, config)
        finally:
            state.context = original_context

        if isinstance(result, ModelResponseConverter):

            def _on_reply(message: Message) -> None:
                if message.role != "assistant":
                    return
                done = _has_done_token(message.text())
                if done:
                    state.execution_meta.internal_data[_DONE_KEY] = True
                    _strip_done_token(message)
                if not done or planner_has_text:
                    message.metadata["internal"] = True

            result.message_hooks.append(_on_reply)
        return result

    _reflect.__name__ = "reflect"
    return _reflect


def _make_plan_node(plan_agent: Agent) -> Callable:
    """Wrap *plan_agent* so its request never ends on an assistant turn.

    On a later iteration the context ends with the reflector's assistant message. In that
    case one transient user message (``PLAN_CONTINUE_NOTE``) is added to the request view.
    Nothing is added on the first iteration (the last message is the user's) or after tool
    results (the last message is a tool message). The message is never saved to state.
    """

    async def _plan(state: AgentState, config: dict) -> object:
        original_context = state.context
        last = original_context[-1] if original_context else None
        if last is not None and last.role == "assistant" and not last.tools_calls:
            state.context = [
                *original_context,
                Message.text_message(PLAN_CONTINUE_NOTE, role="user"),
            ]
        try:
            return await plan_agent.execute(state, config)
        finally:
            state.context = original_context

    _plan.__name__ = "plan"
    _plan.agent = plan_agent  # type: ignore[attr-defined]
    return _plan


def _make_increment_node(key: str) -> Callable[[AgentState], list]:
    """Return an async node that increments an integer counter in internal_data."""

    async def _increment(state: AgentState) -> list:
        state.execution_meta.internal_data[key] = state.execution_meta.internal_data.get(key, 0) + 1
        return []

    _increment.__name__ = f"increment_{key}"
    return _increment  # type: ignore


# ---------------------------------------------------------------------------
# Routing factories
# ---------------------------------------------------------------------------


def _make_plan_route(*, has_tools: bool) -> Callable[[AgentState], str]:
    """Build the conditional routing function for the PLAN node.

    Routes to ``"ACT"`` when the PLAN agent emitted tool calls, and to
    ``"REFLECT"`` when it produced a direct (text-only) response.
    """

    def _route(state: AgentState) -> str:
        if not state.context:
            return "REFLECT"

        last = state.context[-1]

        if (
            has_tools
            and last.role == "assistant"
            and last.tools_calls
            and len(last.tools_calls) > 0
        ):
            return "ACT"

        return "REFLECT"

    return _route


def _make_reflect_route(max_iterations: int) -> Callable[[AgentState], str]:
    """Build the conditional routing function for the REFLECT node.

    Routing logic (evaluated in order):

    1. If ``max_iterations`` is reached → ``END``.
    2. If the reflector signalled ``[DONE]`` (per-turn flag, or the token still in the
       last message, case-insensitive) → ``END``.
    3. If the last REFLECT reply is empty (no text, no tool calls) → ``END`` with a warning.
    4. Otherwise increment the iteration counter and route to ``"PLAN"``.

    .. note::
        The counter increment is a deliberate, controlled side effect kept
        inside this routing function so the graph topology stays simple
        (no additional increment-only nodes).
    """

    def _route(state: AgentState) -> str:
        iterations = state.execution_meta.internal_data.get(_ITERATIONS_KEY, 0)

        # Hard cap first — avoids wasted LLM calls.
        if iterations >= max_iterations:
            logger.warning("PlanActReflect reached max_iterations=%d; terminating.", max_iterations)
            return END

        if state.execution_meta.internal_data.get(_DONE_KEY):
            logger.debug("REFLECT signalled [DONE] after %d iteration(s).", iterations)
            return END

        if not state.context:
            return END

        last = state.context[-1]
        text = last.text()

        # Fallback for a reply that was not post-processed (e.g. a custom reflector node).
        if "[done]" in text.lower():
            logger.debug("REFLECT signalled [DONE] after %d iteration(s).", iterations)
            return END

        if _is_empty_reply(last):
            logger.warning(
                "REFLECT returned an empty reply (no text, no tool calls); ending the turn "
                "instead of looping."
            )
            return END

        logger.debug(
            "REFLECT iteration %d/%d — routing back to PLAN.", iterations + 1, max_iterations
        )
        return _INCREMENT_ITERATIONS_NODE

    return _route


# ---------------------------------------------------------------------------
# PlanActReflectAgent
# ---------------------------------------------------------------------------


class PlanActReflectAgent[StateT: AgentState]:
    """Self-contained Plan → Act → Reflect looping agent.

    Creates three internal ``Agent`` instances (planner, actor, reflector) with
    purpose-specific default system prompts.  Pass ``model`` and optionally
    ``tools``; everything else has sensible defaults.

    Usage::

        from tenxgraph.prebuilt.agent import PlanActReflectAgent

        def web_search(query: str) -> str:
            ...

        agent = PlanActReflectAgent(
            model="gpt-4o-mini",
            tools=[web_search],
            max_iterations=4,
        )
        app = agent.compile(checkpointer=...)
        result = await app.ainvoke(
            {"message": "Research the impact of AI on climate science."},
            config={"thread_id": "t1"},
        )

    Graph topology::

        PLAN --[tool calls?]--> ACT --> REFLECT --[done or max_iters?]--> END
            \\                                  \\
             +--> (no tools) --> REFLECT         +--> PLAN (iterate)

    Message metadata: reflector messages carry ``metadata["internal"] = True``, a step
    message clients may show collapsed; the turn's answer is the last assistant message
    without it (the planner's reply). The one exception is a done verdict after a planner
    reply that has no text (tool calls only): that verdict is the only text answer and is
    not marked.

    Args:
        model: LLM model identifier (e.g. ``"gpt-4o-mini"``, ``"gemini-2.0-flash"``).
        tools: Callable tools made available to the PLAN agent.
        plan_system_prompt: Override the default planner system prompt.
        reflect_system_prompt: Override the default reflector system prompt.
        max_iterations: Maximum number of PLAN→ACT→REFLECT cycles per turn
            before the graph terminates (default ``3``).  The count resets on
            each new turn of a thread.
        state: Optional initial ``AgentState`` (or subclass) instance.
        context_manager: Optional custom context manager.
        publisher: Optional publisher for streaming/events.
        id_generator: ID generation strategy.
        container: InjectQ DI container.
        **agent_kwargs: Extra keyword arguments forwarded to all inner ``Agent``
            instances (e.g. ``provider``, ``temperature``, ``reasoning_config``,
            ``retry_config``).
    """

    def __init__(  # noqa: PLR0913
        self,
        model: str,
        tools: Iterable[Callable] | None = None,
        plan_system_prompt: list[dict[str, Any]] | None = None,
        reflect_system_prompt: list[dict[str, Any]] | None = None,
        max_iterations: int = 3,
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        publisher: BasePublisher | list[BasePublisher] | None = None,
        id_generator: BaseIDGenerator = DefaultIDGenerator(),
        container: InjectQ | None = None,
        *,
        # Per-phase model overrides — fall back to `model` when not set
        plan_model: str | None = None,
        reflect_model: str | None = None,
        # Per-phase reasoning config overrides — fall back to `reasoning_config` when not set
        plan_reasoning_config: dict[str, Any] | bool | None = None,
        reflect_reasoning_config: dict[str, Any] | bool | None = None,
        # Agent pass-through options
        client: Any = None,
        pass_user_info_to_mcp: bool = False,
        extra_messages: list[Message] | None = None,
        trim_context: bool = False,
        tools_tags: set[str] | None = None,
        reasoning_config: dict[str, Any] | bool | None = True,
        skills: SkillConfig | None = None,
        memory: MemoryConfig | None = None,
        retry_config: Any = True,
        fallback_models: list[str | tuple[str, str]] | None = None,
        multimodal_config: MultimodalConfig | None = None,
        **agent_kwargs: Any,
    ):
        self._model = model
        self._plan_model = plan_model or model
        self._reflect_model = reflect_model or model
        self._max_iterations = max_iterations
        self._plan_system_prompt = plan_system_prompt or DEFAULT_PLAN_SYSTEM_PROMPT
        self._reflect_system_prompt = reflect_system_prompt or DEFAULT_REFLECT_SYSTEM_PROMPT

        # Per-phase reasoning config — explicit override wins, else shared config, else default True
        self._plan_reasoning_config = (
            plan_reasoning_config if plan_reasoning_config is not None else reasoning_config
        )
        self._reflect_reasoning_config = (
            reflect_reasoning_config if reflect_reasoning_config is not None else reasoning_config
        )

        # Agent pass-through
        self._client = client
        self._pass_user_info_to_mcp = pass_user_info_to_mcp
        self._extra_messages = extra_messages
        self._trim_context = trim_context
        self._tools_tags = tools_tags
        self._reasoning_config = reasoning_config
        self._skills = skills
        self._memory = memory
        self._retry_config = retry_config
        self._fallback_models = fallback_models
        self._multimodal_config = multimodal_config
        self._agent_kwargs = agent_kwargs

        # Graph infrastructure
        self._state = state
        self._context_manager = context_manager
        self._publisher = publisher
        self._id_generator = id_generator
        self._container = container

        # Build tool node once; reused across compile() calls.
        self._tool_node: ToolNode | None = self._build_tool_node(
            tools=list(tools or []),
            client=client,
            pass_user_info_to_mcp=pass_user_info_to_mcp,
        )

        # Lazy graph handle — recreated on each _configure_graph() call.
        self._graph: StateGraph[StateT] = self._new_graph()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_graph(self) -> StateGraph[StateT]:
        return StateGraph[StateT](
            state=self._state,
            context_manager=self._context_manager,
            publisher=self._publisher,
            id_generator=self._id_generator,
            container=self._container,
        )

    @staticmethod
    def _build_tool_node(
        *,
        tools: list[Callable],
        client: Any,
        pass_user_info_to_mcp: bool,
    ) -> ToolNode | None:
        if not tools and client is None:
            return None
        return ToolNode(tools, client=client, pass_user_info_to_mcp=pass_user_info_to_mcp)

    def _build_agent(
        self,
        system_prompt: list[dict[str, Any]],
        *,
        with_tools: bool,
        model: str | None = None,
        reasoning_config: dict[str, Any] | bool | None = None,
    ) -> Agent:
        return Agent(
            model=model or self._model,
            system_prompt=system_prompt,
            tool_node=self._tool_node if with_tools else None,
            extra_messages=self._extra_messages,
            trim_context=self._trim_context,
            tools_tags=self._tools_tags if with_tools else None,
            reasoning_config=reasoning_config
            if reasoning_config is not None
            else self._reasoning_config,
            skills=self._skills,
            memory=self._memory,
            retry_config=self._retry_config,
            fallback_models=self._fallback_models,
            multimodal_config=self._multimodal_config,
            **self._agent_kwargs,
        )

    def _configure_graph(self) -> None:
        self._graph = self._new_graph()

        # --- PLAN node (planner agent with tools) ---
        plan_agent = self._build_agent(
            self._plan_system_prompt,
            with_tools=True,
            model=self._plan_model,
            reasoning_config=self._plan_reasoning_config,
        )
        self._graph.add_node("PLAN", _make_plan_node(plan_agent))

        # --- ACT node (ToolNode, optional) ---
        if self._tool_node is not None:
            self._graph.add_node("ACT", self._tool_node)
            self._graph.add_edge("ACT", "REFLECT")

        # --- REFLECT node (reflector agent, no tools, tool messages filtered out) ---
        reflect_agent = self._build_agent(
            self._reflect_system_prompt,
            with_tools=False,
            model=self._reflect_model,
            reasoning_config=self._reflect_reasoning_config,
        )
        self._graph.add_node("REFLECT", _make_reflect_node(reflect_agent))

        # --- Conditional edges from PLAN ---
        plan_path_map: dict[str, str] = {"REFLECT": "REFLECT"}
        if self._tool_node is not None:
            plan_path_map["ACT"] = "ACT"

        self._graph.add_conditional_edges(
            "PLAN",
            _make_plan_route(has_tools=self._tool_node is not None),
            plan_path_map,
        )

        # --- INCREMENT_ITERATIONS node: increments counter then falls through to PLAN ---
        self._graph.add_node(_INCREMENT_ITERATIONS_NODE, _make_increment_node(_ITERATIONS_KEY))
        self._graph.add_edge(_INCREMENT_ITERATIONS_NODE, "PLAN")

        # --- Conditional edges from REFLECT ---
        self._graph.add_conditional_edges(
            "REFLECT",
            _make_reflect_route(self._max_iterations),
            {_INCREMENT_ITERATIONS_NODE: _INCREMENT_ITERATIONS_NODE, END: END},
        )

        # --- Entry point ---
        self._graph.set_entry_point("PLAN")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(
        self,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
        media_store: BaseMediaStore | None = None,
        shutdown_timeout: float = 30.0,
        media_policy: MediaFetchPolicy | None = None,
    ) -> CompiledGraph:
        """Wire the graph and return a :class:`~tenxgraph.core.graph.CompiledGraph`.

        Args:
            checkpointer: Persistence backend for state snapshots.
            store: Long-term key-value store.
            interrupt_before: Node names to pause execution before.
            interrupt_after: Node names to pause execution after.
            callback_manager: Callback hooks for observability.
            media_store: Media/file storage backend.
            shutdown_timeout: Graceful-shutdown timeout in seconds.
            media_policy: Which external media URLs the server may download.

        Returns:
            A compiled, ready-to-invoke graph.
        """
        self._configure_graph()
        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
            media_store=media_store,
            shutdown_timeout=shutdown_timeout,
            media_policy=media_policy,
        )
