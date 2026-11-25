"""Public Agent facade for graph-based LLM interactions.

The public import path remains ``tenxgraph.graph.agent.Agent`` while the
implementation lives in smaller internal modules under ``tenxgraph.graph.agent_internal``.
"""

import logging
from typing import TYPE_CHECKING, Any

from tenxgraph.core.graph.base_agent import BaseAgent
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.skills.models import SkillConfig
from tenxgraph.core.state.message import Message
from tenxgraph.storage.media.config import MultimodalConfig

from .agent_internal.anthropic import AgentAnthropicMixin
from .agent_internal.constants import REASONING_DEFAULT, RetryConfig
from .agent_internal.execution import AgentExecutionMixin
from .agent_internal.google import AgentGoogleMixin
from .agent_internal.memory import AgentMemoryMixin
from .agent_internal.model_registry import ModelEntry, build_registry, resolve_entry
from .agent_internal.openai import AgentOpenAIMixin
from .agent_internal.providers import AgentProviderMixin, validate_output_type
from .agent_internal.skills import AgentSkillsMixin


if TYPE_CHECKING:
    from tenxgraph.storage.store.memory_config import MemoryConfig


logger = logging.getLogger("tenxgraph.agent")


class Agent(
    AgentExecutionMixin,
    AgentAnthropicMixin,
    AgentGoogleMixin,
    AgentOpenAIMixin,
    AgentProviderMixin,
    AgentSkillsMixin,
    AgentMemoryMixin,
    BaseAgent,
):
    """A smart node function wrapper for LLM interactions.

    This class handles common boilerplate for agent implementations including:
    - Automatic message conversion
    - LLM calls via native provider SDKs (OpenAI, Google)
    - Tool handling with conditional logic
    - Optional learning/RAG capabilities
    - Response conversion

    The Agent is designed to be used as a node within a StateGraph, providing
    a high-level interface while maintaining full graph flexibility.

    Example:
        ```python
        # Create an agent node with a ToolNode
        tool_node = ToolNode([weather_tool])
        agent = Agent(
            model="gpt-4o",
            provider="openai",
            system_prompt=[{"role": "system", "content": "You are a helpful assistant"}],
            tool_node=tool_node,
        )

        # Use it in a graph
        graph = StateGraph()
        graph.add_node("MAIN", agent)
        graph.add_node("TOOL", tool_node)
        # ... setup conditional edges
        ```

    Attributes:
        model: Model identifier (e.g., "gpt-4o", "gemini-2.0-flash")
        provider: Provider name ("openai", "google")
        system_prompt: System prompt string or list of message dicts
        tool_node: ToolNode instance or name of an existing TOOL graph node (str)
        client: Optional custom client instance (escape hatch for power users)
        temperature: LLM sampling temperature
        max_tokens: Maximum tokens to generate
        llm_kwargs: Additional provider-specific parameters
    """

    def __init__(  # noqa: PLR0913
        self,
        model: str,
        output_type: str = "text",
        system_prompt: list[dict[str, Any]] | None = None,
        tool_node: "str | ToolNode | None" = None,
        extra_messages: list[Message] | None = None,
        trim_context: bool = False,
        tools_tags: set[str] | None = None,
        reasoning_config: dict[str, Any] | bool | None = REASONING_DEFAULT,  # type: ignore
        skills: "SkillConfig | None" = None,
        memory: "MemoryConfig | None" = None,
        retry_config: RetryConfig | bool | None = True,
        fallback_models: list[str | tuple[str, str]] | None = None,
        multimodal_config: MultimodalConfig | None = None,
        output_schema: Any | None = None,
        model_registry: dict[str, dict[str, Any]] | None = None,
        model_config_key: str = "model",
        **kwargs,
    ):
        """Initialize an Agent node.

        Args:
            model: Model identifier (any model name - no parsing required).
                Examples: "gpt-4o", "gemini-2.0-flash-exp", "qwen-2.5-72b", "deepseek-chat"
            provider: Provider name ("openai", "google"). If None, will auto-detect from model.
            output_type: Type of output to generate (default: "text").
                - "text": Text generation (default, most common)
                - "image": Image generation
                - "video": Video generation
                - "audio": Audio/TTS generation
            system_prompt: System prompt as list of message dicts.
                Supports state variable interpolation using placeholders like ``{field_name}``.
                At execution time, placeholders are replaced with actual values from the state.

                Example with interpolation::

                    class MyState(AgentState):
                        user_name: str = "Guest"
                        occasion: str = "casual"

                    agent = Agent(
                        model="gpt-4o",
                        system_prompt=[{
                            "role": "system",
                            "content": "You are helping {user_name} with {occasion} planning."
                        }]
                    )
                    # At runtime, placeholders are replaced with state values
            tool_node: A ``ToolNode`` instance containing the tools this agent may call,
                **or** a ``str`` naming an existing graph node whose ``func`` is a
                ``ToolNode`` (resolved at execution time via the DI container).
                Pass ``None`` when the agent needs no tools.

                Examples::

                    # Inline ToolNode — agent owns the tools
                    tool_node = ToolNode([get_weather, search])
                    agent = Agent(model="gpt-4o", tool_node=tool_node)

                    # Named reference — ToolNode lives as a separate graph node
                    agent = Agent(model="gpt-4o", tool_node="TOOL")

            extra_messages: Messages sent after the history on every request and never saved
                to the state. ``{field}`` placeholders in their text are filled from the
                state like the system prompt; a field that is not in the state stays as
                written, and ``{{`` / ``}}`` give literal braces. Put changing per-request
                context here, not in the system prompt, so the cached prefix stays stable.
                Prefer role ``"user"`` (the Google GenAI and Anthropic adapters move role
                ``"system"`` to the front).
            trim_context: Whether to trim context using context manager.
            tools_tags: Optional tags to filter tools.
            base_url (via **kwargs): Optional custom endpoint: OpenAI-compatible APIs
                (ollama, vllm, openrouter, deepseek, etc.), or a proxy, AI gateway
                (LiteLLM) or mock server for Anthropic and Google. Default: ``None``.
            api_style: API style for OpenAI provider. ``"chat"`` uses
                Chat Completions, ``"responses"`` uses the Responses API.
                Default: ``"responses"``, except when ``base_url`` is set, where
                it is ``"chat"`` because most OpenAI-compatible servers only
                implement Chat Completions. An explicit value always wins. With
                ``"responses"`` and a ``base_url``, the first call probes the
                server; if it answers 404, 405 or 501 (no Responses API) the agent logs
                one warning and uses Chat Completions for the rest of its lifetime.
                Other errors (429, timeouts, 5xx) are raised and do not switch it.
            memory: Optional ``MemoryConfig`` enabling agent-level long-term
                memory tools and system prompts.
            reasoning_config: Unified reasoning control for all providers. Default
                (and ``True``) is reasoning on at the provider's lowest effort: OpenAI
                ``"minimal"``, Anthropic ``"low"``, Google ``"low"`` (``thinking_budget=512``).
                Pass ``None`` to turn off. ``effort`` applies to every provider;
                ``summary`` is OpenAI-only (Responses defaults it to ``"auto"``);
                ``thinking_budget`` is Google-only and overrides ``effort``.

                For Google, ``effort`` is translated to ``thinking_budget`` automatically:
                ``"low"`` → 512, ``"medium"`` → 8192, ``"high"`` → 24576.

                Examples::

                    reasoning_config=None                        # OFF for both
                    reasoning_config={"effort": "high"}          # high, both providers
                    reasoning_config={"effort": "low", "summary": "auto"}  # OpenAI: low+summary
                    reasoning_config={"thinking_budget": 5000}   # Google exact budget
            retry_config: Controls automatic retry with exponential back-off for
                transient LLM errors (429, 500, 502, 503, 529).  Default is
                ``True`` which uses ``RetryConfig()`` (3 retries, 1 s initial
                delay, 2x backoff, 30 s cap).  Pass ``False`` or ``None`` to
                disable.  Pass a ``RetryConfig`` instance for fine-grained
                control::

                    retry_config = RetryConfig(max_retries=5, initial_delay=2.0)
                    retry_config = False  # disable retries entirely

            fallback_models: Ordered list of fallback models to try when the
                primary model exhausts all retries.  Each entry is either a
                plain model string (inherits the agent's provider) or a
                ``(model, provider)`` tuple for cross-provider fallback::

                    fallback_models = ["gpt-4o-mini"]
                    fallback_models = [("gemini-2.0-flash", "google")]
                    fallback_models = ["gpt-4o-mini", ("gemini-2.0-flash", "google")]

                A string that names a ``model_registry`` entry is that entry, with its
                own client and settings. A fallback on the same provider reuses the
                Agent's ``http_client``; with an ``http_client`` set, a ``(model,
                provider)`` fallback on another provider raises ``ValueError``: give it a
                registry entry with its own ``http_client``.

            model_registry: Optional named models the run config can pick from, as
                ``{name: entry}``. An entry takes the model keys the Agent takes:
                ``model`` (required), ``provider``, ``base_url``, ``api_key``,
                ``api_style``, ``use_vertex_ai``, ``anthropic_backend``,
                ``reasoning_config``, ``retry_config`` and provider kwargs. With a
                registry, ``model`` is the name of the default entry, and ``provider``,
                ``base_url``, ``api_key``, ``api_style``, ``use_vertex_ai`` and
                ``anthropic_backend`` on the Agent raise ``ValueError``. Provider kwargs
                and ``reasoning_config`` on the Agent are shared defaults; an entry's own
                keys win::

                    Agent(
                        model="openai",
                        model_registry={
                            "openai": {"model": "gpt-5-mini"},
                            "vllm": {
                                "provider": "openai",
                                "model": "deepseek-r1",
                                "base_url": "http://gpu1:8000/v1",
                            },
                        },
                    )
                    await app.ainvoke(state, config={"thread_id": "t1", "model": "vllm"})

            model_config_key: The run-config key that names the entry for a run
                (default ``"model"``). An unregistered name raises
                ``ModelNotFoundError``. An Agent without a registry ignores the key.
            **llm_kwargs: Additional provider-specific parameters
                (temperature, max_tokens, top_p, or model args, organization_id, project_id).

        Raises:
            ImportError: If required provider SDK is not installed.
            ValueError: If provider cannot be determined or doesn't support output_type.

        Example:
            ```python
            # Text generation with inline ToolNode
            tool_node = ToolNode([weather_tool, calculator])
            text_agent = Agent(
                model="openai/gpt-4o",
                system_prompt=[{"role": "system", "content": "You are a helpful assistant"}],
                tool_node=tool_node,
                temperature=0.8,
            )

            # Text generation with named ToolNode in graph
            agent = Agent(
                model="google/gemini-2.5-flash",
                tool_node="TOOL",  # references graph node named "TOOL"
            )

            # No tools
            agent = Agent(model="gpt-4o")

            # Image generation
            image_agent = Agent(
                model="openai/dall-e-3",
                output_type="image",
            )

            # Third-party models (Qwen, DeepSeek, Ollama)
            qwen_agent = Agent(
                model="qwen-2.5-72b-instruct",
                provider="openai",
                base_url="https://api.qwen.com/v1",
            )

            # With retry and fallback
            resilient_agent = Agent(
                model="gemini-2.5-flash",
                provider="google",
                retry_config=RetryConfig(max_retries=5, initial_delay=2.0),
                fallback_models=[
                    "gemini-2.0-flash",
                    ("gpt-4o-mini", "openai"),
                ],
            )
            ```
        """
        # One model is resolved by the same rules as a registry entry, so a single model and
        # a registry share one path. A registry's entries are resolved once, here.
        self.model_config_key = model_config_key
        self.default_model_name: str | None = None
        self._model_entries: dict[str, ModelEntry] | None = None
        if model_registry is None:
            spec: dict[str, Any] = {"model": model}
            for key in ("provider", "base_url", "api_style", "use_vertex_ai"):
                if key in kwargs:
                    spec[key] = kwargs.pop(key)
            default = resolve_entry(None, spec, kwargs, reasoning_config, retry_config)
        else:
            self._model_entries = build_registry(
                model_registry, model, kwargs, reasoning_config, retry_config
            )
            self.default_model_name = model
            default = self._model_entries[model]
        # Clients of non-default entries, built on first use; shared by per-attempt copies.
        self._entry_clients: dict[str | None, Any] = {}

        # Persist so fallback clients (created lazily at call time) honour the
        # same Vertex AI selection as the primary client.
        self.use_vertex_ai = default.use_vertex_ai
        # Call parent constructor
        super().__init__(
            model=default.model,
            system_prompt=system_prompt or [],
            tool_node=tool_node,
            base_url=default.base_url,
            **default.llm_kwargs,
        )

        # Store output type
        self.output_type = output_type.lower()
        self.output_schema = output_schema
        self._validate_output_schema_output_type()

        # self.llm_kwargs is set by super().__init__ and is read by _create_client().
        self.base_url = default.base_url
        self.provider = default.provider
        self.client = self._create_client(self.provider, self.base_url, self.use_vertex_ai)

        # Validate that provider supports the output type
        self._validate_output_type()
        for entry in (self._model_entries or {}).values():
            validate_output_type(entry.provider, self.output_type)

        self.extra_messages = extra_messages
        self.trim_context = trim_context
        self.tools_tags = tools_tags
        self.tool_node_name = None  # may be set to a str by _setup_tools()

        # Internal setup
        self._tool_node = self._setup_tools()

        self.api_style = default.api_style
        # base_url values whose server rejected the Responses API. Mutable and shared with the
        # per-attempt copies of this Agent, so one failed probe sticks for the Agent's lifetime.
        self._responses_unsupported_urls: set[str] = set()

        # Reasoning and retry settings, resolved by resolve_entry (see model_registry.py).
        self._reasoning_mode = default.reasoning_mode
        self.reasoning_config: dict[str, Any] | None = default.reasoning_config
        self.retry_config: RetryConfig | None = default.retry_config

        # Normalise fallback_models to list[tuple[str, str | None]]
        self.fallback_models: list[tuple[str, str | None]] = []
        if fallback_models:
            for entry in fallback_models:
                if isinstance(entry, str):
                    self.fallback_models.append((entry, None))  # inherit provider
                else:
                    self.fallback_models.append(tuple(entry))  # type: ignore[arg-type]
        # The http_client is built for this provider's SDK (httpx or httpx2) and endpoint (CA,
        # client cert), so a fallback on another provider cannot use it. Refuse it here instead
        # of failing, or leaving without mTLS, at call time.
        if "http_client" in self.llm_kwargs:
            for fb_model, fb_provider in self.fallback_models:
                if fb_provider is not None and fb_provider != self.provider:
                    raise ValueError(
                        f"fallback ({fb_model!r}, {fb_provider!r}) is on another provider "
                        f"than the Agent's http_client ({self.provider!r}). Give it its own "
                        "http_client in a model_registry entry and name that entry in "
                        "fallback_models."
                    )

        logger.info(
            f"Agent initialized: model={self.model}, provider={self.provider}, "
            f"output_type={self.output_type}, has_tools={self._tool_node is not None}"
        )

        # Memory setup (via mixin) runs before skills so a memory-only Agent can
        # lazily create the internal ToolNode that both systems append to.
        self._setup_memory(memory)

        # Skills setup (via mixin)
        self._setup_skills(skills)

        # Multimodal configuration
        self.multimodal_config = multimodal_config

    def _validate_output_schema_output_type(self) -> None:
        """Validate compatibility between output_schema and output_type."""
        if self.output_schema is None:
            return

        if self.output_type in {"image", "video", "audio"}:
            raise ValueError(
                "output_schema can only be used with text generation. "
                "Use output_type='text' (or 'json' for legacy compatibility), "
                "or remove output_schema for media generation."
            )
