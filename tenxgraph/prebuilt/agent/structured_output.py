"""StructuredOutputAgent — LLM generation with schema validation and automatic repair.

Pattern: GENERATE → (validate) → END
                              ↘ (invalid, attempts < max) → REPAIR → GENERATE

The agent guarantees that tool-use is handled transparently inside the generation
loop (GENERATE ↔ TOOL) before validation is attempted, so tools and structured
output work together without extra graph wiring.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import pydantic
from injectq import InjectQ

from agentflow.core.graph.agent import Agent
from agentflow.core.graph.compiled_graph import CompiledGraph
from agentflow.core.graph.state_graph import StateGraph
from agentflow.core.graph.tool_node import ToolNode
from agentflow.core.skills.models import SkillConfig
from agentflow.core.state.agent_state import AgentState
from agentflow.core.state.base_context import BaseContextManager
from agentflow.core.state.message import Message
from agentflow.runtime.publisher.base_publisher import BasePublisher
from agentflow.storage.checkpointer.base_checkpointer import BaseCheckpointer
from agentflow.storage.media.config import MultimodalConfig
from agentflow.storage.media.storage.base import BaseMediaStore
from agentflow.storage.store.base_store import BaseStore
from agentflow.storage.store.memory_config import MemoryConfig
from agentflow.utils.callbacks import CallbackManager
from agentflow.utils.constants import END
from agentflow.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


logger = logging.getLogger("agentflow.prebuilt.structured_output")

StateT = TypeVar("StateT", bound=AgentState)

# Keys used in execution_meta.internal_data
_ATTEMPTS_KEY = "soa_attempts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_type_adapter(schema: type) -> pydantic.TypeAdapter:
    """Return a ``pydantic.TypeAdapter`` for *schema* (Pydantic model or TypedDict)."""
    return pydantic.TypeAdapter(schema)


def _schema_json(schema: type) -> str:
    """Return the JSON Schema string for *schema*."""
    try:
        return json.dumps(_build_type_adapter(schema).json_schema(), indent=2)
    except Exception:
        return schema.__name__


def _validate_message(
    message: Message,
    adapter: pydantic.TypeAdapter,
) -> tuple[bool, str]:
    """Try to validate *message* content against *adapter*.

    Returns:
        ``(is_valid, error_string)`` — *error_string* is empty when valid.
    """
    # 1. Prefer ``parsed_content`` populated by the provider converter.
    if message.parsed_content is not None:
        try:
            raw = (
                message.parsed_content.model_dump()
                if isinstance(message.parsed_content, pydantic.BaseModel)
                else message.parsed_content
            )
            adapter.validate_python(raw)
            return True, ""
        except Exception:  # nosec: B110  # noqa: B110, S110, RUF100, RUF102
            pass  # fall through to text-based parsing

    # 2. Attempt to parse the text content as JSON.
    text = message.text().strip()
    if not text:
        return False, "LLM returned an empty response."

    # Strip optional markdown code fences (```json … ``` or ``` … ```).
    if text.startswith("```"):
        lines = text.splitlines()
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end])

    try:
        data = json.loads(text)
        adapter.validate_python(data)
        return True, ""
    except json.JSONDecodeError as exc:
        return False, f"Response is not valid JSON: {exc}"
    except pydantic.ValidationError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Routing + repair node factories
# ---------------------------------------------------------------------------


def _make_route_fn(
    output_schema: type,
    max_attempts: int,
    *,
    has_tools: bool,
) -> Callable[[AgentState], str]:
    """Build the conditional-edge routing function for the GENERATE node."""

    adapter = _build_type_adapter(output_schema)

    def _route(state: AgentState) -> str:
        if not state.context:
            return END

        last = state.context[-1]

        # If the LLM requested tool calls, execute them first.
        if (
            has_tools
            and last.role == "assistant"
            and last.tools_calls
            and len(last.tools_calls) > 0
        ):
            return "TOOL"

        # Validate the final assistant response.
        attempts = state.execution_meta.internal_data.get(_ATTEMPTS_KEY, 0)
        is_valid, _ = _validate_message(last, adapter)

        if is_valid:
            logger.debug("Structured output validated successfully after %d attempt(s).", attempts)
            return END

        if attempts >= max_attempts:
            logger.warning(
                "Max structured output repair attempts (%d) reached; returning best response.",
                max_attempts,
            )
            return END

        return "REPAIR"

    return _route


def _make_repair_fn(output_schema: type) -> Callable:
    """Build the REPAIR node function.

    The node injects a correction user-message that carries the validation
    error and the target JSON Schema, then increments the attempt counter.
    """

    adapter = _build_type_adapter(output_schema)
    schema_hint = _schema_json(output_schema)

    async def _repair(state: AgentState, config: dict) -> list[Message]:
        attempts = state.execution_meta.internal_data.get(_ATTEMPTS_KEY, 0)
        state.execution_meta.internal_data[_ATTEMPTS_KEY] = attempts + 1

        last = state.context[-1] if state.context else None
        _, error = (
            _validate_message(last, adapter) if last else (False, "No response was generated.")
        )

        correction = (
            "Your previous response did not conform to the required output schema.\n"
            f"Validation error:\n{error}\n\n"
            f"Target JSON Schema:\n{schema_hint}\n\n"
            "Please produce a response that is **valid JSON** and strictly matches the "
            "schema above. Output only the JSON object — no extra text or code fences."
        )
        logger.debug(
            "Injecting repair message (attempt %d/%d): %s",
            attempts + 1,
            state.execution_meta.internal_data.get(_ATTEMPTS_KEY),
            error[:120],
        )
        return [Message.text_message(correction, role="user")]

    return _repair


# ---------------------------------------------------------------------------
# StructuredOutputAgent
# ---------------------------------------------------------------------------


class StructuredOutputAgent[StateT: AgentState]:
    """Self-contained agent that guarantees structured LLM output.

    Pass a Pydantic ``BaseModel`` or ``TypedDict`` subclass as *output_schema*.
    If the LLM's response fails schema validation the agent automatically
    injects a correction prompt and retries up to *max_attempts* times.

    Usage::

        from pydantic import BaseModel
        from agentflow.prebuilt.agent import StructuredOutputAgent


        class MovieReview(BaseModel):
            title: str
            rating: float
            summary: str


        agent = StructuredOutputAgent(
            model="gpt-4o-mini",
            output_schema=MovieReview,
            system_prompt=[{"role": "system", "content": "You are a film critic."}],
            max_attempts=3,
        )
        app = agent.compile()
        result = await app.ainvoke({"message": "Review Inception."}, config={"thread_id": "t1"})

    Graph topology::

        GENERATE --[valid]--> END
                 \\-[invalid, attempts < max]--> REPAIR --> GENERATE
                 \\-[tool calls (if tools provided)]--> TOOL --> GENERATE

    Args:
        model: LLM model identifier (e.g. ``"gpt-4o-mini"``, ``"gemini-2.0-flash"``).
        output_schema: Pydantic ``BaseModel`` subclass or ``TypedDict`` subclass
            that defines the expected output shape.
        tools: Optional callables to expose as tools during generation.
        system_prompt: System prompt for the generation agent.
        max_attempts: Maximum number of repair+retry cycles before accepting
            the best available response (default ``2``).
        repair_system_prompt: System prompt for the repair agent.  When
            ``None`` (default) the repair node is a lightweight context-
            injection function rather than a full LLM call, which keeps
            token usage low.  Provide a list of message dicts to enable a
            dedicated repair LLM pass.
        state: Optional initial ``AgentState`` (or subclass) instance.
        context_manager: Optional custom context manager.
        publisher: Optional publisher for streaming/events.
        id_generator: ID generation strategy.
        container: InjectQ DI container.
        **agent_kwargs: Extra keyword arguments forwarded to the inner
            ``Agent`` instances (e.g. ``provider``, ``temperature``,
            ``reasoning_config``, ``retry_config``).
    """

    def __init__(  # noqa: PLR0913
        self,
        model: str,
        output_schema: type,
        tools: Iterable[Callable] | None = None,
        system_prompt: list[dict[str, Any]] | None = None,
        max_attempts: int = 2,
        repair_system_prompt: list[dict[str, Any]] | None = None,
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        publisher: BasePublisher | list[BasePublisher] | None = None,
        id_generator: BaseIDGenerator = DefaultIDGenerator(),
        container: InjectQ | None = None,
        *,
        # Pass-through Agent kwargs
        output_type: str = "text",
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
        self._output_schema = output_schema
        self._max_attempts = max_attempts
        self._repair_system_prompt = repair_system_prompt
        self._system_prompt = system_prompt

        # Agent pass-through options
        self._output_type = output_type
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

        # Build the tool node once (reused across compile calls)
        self._tool_node: ToolNode | None = self._build_tool_node(
            tools=list(tools or []),
            client=client,
            pass_user_info_to_mcp=pass_user_info_to_mcp,
        )

        # Lazy graph handle — created in _configure_graph()
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

    def _build_generate_agent(self) -> Agent:
        return Agent(
            model=self._model,
            output_type=self._output_type,
            system_prompt=self._system_prompt,
            tool_node=self._tool_node,
            extra_messages=self._extra_messages,
            trim_context=self._trim_context,
            tools_tags=self._tools_tags,
            reasoning_config=self._reasoning_config,
            skills=self._skills,
            memory=self._memory,
            retry_config=self._retry_config,
            fallback_models=self._fallback_models,
            multimodal_config=self._multimodal_config,
            output_schema=self._output_schema,
            **self._agent_kwargs,
        )

    def _build_repair_agent(self) -> Agent:
        """Build a dedicated LLM repair agent (used when *repair_system_prompt* is given)."""
        return Agent(
            model=self._model,
            output_type=self._output_type,
            system_prompt=self._repair_system_prompt,
            tool_node=None,  # no tools during repair
            extra_messages=self._extra_messages,
            trim_context=self._trim_context,
            reasoning_config=self._reasoning_config,
            memory=self._memory,
            retry_config=self._retry_config,
            fallback_models=self._fallback_models,
            output_schema=self._output_schema,
            **self._agent_kwargs,
        )

    def _configure_graph(self) -> None:
        self._graph = self._new_graph()

        # --- GENERATE node ---
        generate_agent = self._build_generate_agent()
        self._graph.add_node("GENERATE", generate_agent)

        # --- TOOL node (optional) ---
        if self._tool_node is not None:
            self._graph.add_node("TOOL", self._tool_node)
            self._graph.add_edge("TOOL", "GENERATE")

        # --- REPAIR node ---
        # If the caller supplied a dedicated repair system prompt, use a full
        # Agent so the LLM actively corrects itself.  Otherwise use the cheap
        # context-injection function.
        if self._repair_system_prompt is not None:
            repair_node: Callable = self._build_repair_agent()
        else:
            repair_node = _make_repair_fn(self._output_schema)

        self._graph.add_node("REPAIR", repair_node)
        self._graph.add_edge("REPAIR", "GENERATE")

        # --- Conditional edges from GENERATE ---
        path_map: dict[str, str] = {"REPAIR": "REPAIR", END: END}
        if self._tool_node is not None:
            path_map["TOOL"] = "TOOL"

        self._graph.add_conditional_edges(
            "GENERATE",
            _make_route_fn(
                self._output_schema,
                self._max_attempts,
                has_tools=self._tool_node is not None,
            ),
            path_map,
        )

        # --- Entry point ---
        self._graph.set_entry_point("GENERATE")

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
    ) -> CompiledGraph:
        """Wire the graph and return a :class:`~agentflow.core.graph.CompiledGraph`.

        Args:
            checkpointer: Persistence backend for state snapshots.
            store: Long-term key-value store.
            interrupt_before: Node names to pause execution before.
            interrupt_after: Node names to pause execution after.
            callback_manager: Callback hooks for observability.
            media_store: Media/file storage backend.
            shutdown_timeout: Graceful-shutdown timeout in seconds.

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
        )
