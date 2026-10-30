"""SupervisorTeamAgent — supervisor routes tasks to specialist worker agents.

Pattern::

    SUPERVISOR --[pick WORKER_A]--> WORKER_A ---> SUPERVISOR
    SUPERVISOR --[pick WORKER_B]--> WORKER_B ---> SUPERVISOR
    SUPERVISOR --[FINISH]---------> END
    SUPERVISOR --[FINISH, no worker ran this turn]--> RESPOND --> END

The SUPERVISOR is a dedicated ``Agent`` built internally from *supervisor_model*.
Its system prompt lists all workers and their descriptions so the LLM can make
informed routing decisions.  The supervisor responds with **only** the name of
the next worker (or ``FINISH``); a routing function extracts this token and
directs the graph accordingly. Its own call sees each worker's reply labelled with
the worker's name and the list of workers already run; naming the worker that just
ran counts as ``FINISH``. When it finishes before any worker ran (a greeting, small
talk), the RESPOND node answers the user so the turn is never blank.

Each WORKER is a **pre-built** ``Agent`` (or any
:class:`~tenxgraph.core.graph.base_agent.BaseAgent` subclass) supplied by the
caller via :class:`WorkerConfig`, so every worker can be independently
configured with its own model, tools, memory, skills, retry config, etc.

Round counter lives in ``execution_meta.internal_data["sta_rounds"]`` and is reset
at the start of each turn, so ``max_rounds`` limits one turn, not the whole thread.

Example::

    from tenxgraph.core.graph import Agent, ToolNode
    from tenxgraph.prebuilt.agent import SupervisorTeamAgent
    from tenxgraph.prebuilt.agent.supervisor_team import WorkerConfig


    def web_search(query: str) -> str: ...
    def run_code(code: str) -> str: ...


    agent = SupervisorTeamAgent(
        supervisor_model="gpt-4o",
        workers={
            "RESEARCHER": WorkerConfig(
                agent=Agent(model="gpt-4o-mini", tool_node=ToolNode([web_search])),
                description="Searches the web and returns factual information.",
            ),
            "CODER": WorkerConfig(
                agent=Agent(model="gpt-4o", tool_node=ToolNode([run_code])),
                description="Writes and executes Python code.",
            ),
        },
        max_rounds=8,
    )
    app = agent.compile(checkpointer=...)
    result = await app.ainvoke(
        {"message": "Write a Python script that fetches the top AI papers."},
        config={"thread_id": "t1"},
    )
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from injectq import InjectQ

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.graph.base_agent import BaseAgent
from tenxgraph.core.graph.compiled_graph import CompiledGraph
from tenxgraph.core.graph.state_graph import StateGraph
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.base_context import BaseContextManager
from tenxgraph.core.state.execution_state import register_per_turn_key
from tenxgraph.core.state.message import Message
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.storage.checkpointer.base_checkpointer import BaseCheckpointer
from tenxgraph.storage.media.storage.base import BaseMediaStore
from tenxgraph.storage.media.url_policy import MediaFetchPolicy
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.utils.callbacks import CallbackManager
from tenxgraph.utils.command import Command
from tenxgraph.utils.constants import END
from tenxgraph.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


logger = logging.getLogger("tenxgraph.prebuilt.supervisor_team")

StateT = TypeVar("StateT", bound=AgentState)

# Keys stored in execution_meta.internal_data
_ROUNDS_KEY = register_per_turn_key("sta_rounds")
_ROUTE_HISTORY_KEY = register_per_turn_key("sta_route_history")

_FINISH_TOKEN = "FINISH"  # nosec: B105  # noqa: S105
_SUPERVISOR_NODE = "SUPERVISOR"
_PRE_SUPERVISOR_NODE = "PRE_SUPERVISOR"
_RESPOND_NODE = "RESPOND"
_RESERVED_NODES = (_SUPERVISOR_NODE, _PRE_SUPERVISOR_NODE, _RESPOND_NODE)

# Transient user turns that keep a request from ending on an assistant turn. They are
# added to one request view only and never saved to state.
SUPERVISOR_REQUEST_NOTE = (
    f"Reply with the name of the next worker, or {_FINISH_TOKEN} if my request is answered."
)
WORKER_CONTINUE_NOTE = "Continue the task: do your part of the work described above."

_RESPOND_PROMPT_TEMPLATE = """\
You are the assistant of a team of specialist workers:
{worker_list}

No worker is needed for the user's latest message (for example a greeting, small talk \
or an unclear request). Reply to it yourself, briefly. When it helps, say what the team \
can do or ask what the user needs."""


# ---------------------------------------------------------------------------
# WorkerConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class WorkerConfig:
    """Configuration for a single worker in a :class:`SupervisorTeamAgent`.

    Each worker is a **pre-built** agent instance so that callers can
    independently configure ``skills``, ``memory``, ``multimodal_config``,
    ``retry_config``, or any other per-worker option.

    Args:
        agent: A fully-configured :class:`~tenxgraph.core.graph.agent.Agent`
            (or any :class:`~tenxgraph.core.graph.base_agent.BaseAgent`
            subclass).
        description: Short description of what this worker can do.  Injected
            into the supervisor's system prompt so the LLM knows when to
            delegate here.
    """

    agent: BaseAgent
    description: str = ""


# ---------------------------------------------------------------------------
# Supervisor system-prompt builder
# ---------------------------------------------------------------------------

_SUPERVISOR_PROMPT_TEMPLATE = """\
You are a supervisor agent that coordinates a team of specialist workers to \
complete tasks assigned by the user.

Available workers:
{worker_list}
- {finish_token}: All tasks are fully completed and no further delegation is needed.

Based on the conversation so far, respond with **only** the name of the next \
worker to invoke, or ``{finish_token}`` if the task is complete.

Rules:
- Respond with a single word — exactly one worker name or ``{finish_token}``.
- Do NOT explain your choice.
- Do NOT include any other text.
"""


def _worker_list(workers: dict[str, WorkerConfig]) -> str:
    return "\n".join(
        f"- {name}: {cfg.description or f'Specialist worker {name}.'}"
        for name, cfg in workers.items()
    )


def _build_supervisor_prompt(workers: dict[str, WorkerConfig]) -> list[dict[str, str]]:
    """Build the default supervisor system prompt from the worker registry."""
    content = _SUPERVISOR_PROMPT_TEMPLATE.format(
        worker_list=_worker_list(workers),
        finish_token=_FINISH_TOKEN,
    )
    return [{"role": "system", "content": content}]


def _build_respond_prompt(workers: dict[str, WorkerConfig]) -> list[dict[str, str]]:
    """System prompt of the RESPOND node, which answers when no worker is needed."""
    content = _RESPOND_PROMPT_TEMPLATE.format(worker_list=_worker_list(workers))
    return [{"role": "system", "content": content}]


def _supervisor_view(context: list[Message], history: list[str]) -> list[Message]:
    """The conversation as the supervisor's own call sees it (transient, never saved).

    Earlier turns are kept as plain user and assistant text. The work done for the latest
    user message is folded into one user message that names each worker's reply in order
    ("WEATHER replied: ...") and lists the workers already run, then asks for the next
    worker or FINISH. Unlabelled worker replies and replayed routing tokens made the
    supervisor re-dispatch workers that had already answered: consecutive assistant turns
    are merged by some providers, so it could not tell who said what or that it was done.
    """
    last_user = next(
        (i for i in range(len(context) - 1, -1, -1) if context[i].role == "user"), None
    )
    if last_user is None:
        return list(context)

    view: list[Message] = []
    for msg in context[: last_user + 1]:
        if msg.role == "user":
            view.append(msg)
        elif msg.role == "assistant" and msg.text().strip():
            # Text only: an earlier turn's tool calls and results are not the supervisor's.
            view.append(Message.text_message(msg.text(), role="assistant"))

    replies = [
        f"{msg.metadata.get('node_name') or 'A worker'} replied: {msg.text().strip()}"
        for msg in context[last_user + 1 :]
        if msg.role == "assistant" and msg.text().strip()
    ]
    if not replies and not history:
        return view  # first call of the turn: the request ends with the user's message

    work = "\n\n".join(replies) if replies else "No worker has replied yet."
    already = ", ".join(history) if history else "none"
    note = (
        f"Work done so far for my latest message:\n\n{work}\n\n"
        f"Workers already run for it, in order: {already}.\n{SUPERVISOR_REQUEST_NOTE}"
    )
    view.append(Message.text_message(note, role="user"))
    return view


# ---------------------------------------------------------------------------
# Lightweight increment node
# ---------------------------------------------------------------------------


def _make_increment_rounds_node() -> Callable[[AgentState], list]:
    """Return an async node that increments the round counter in internal_data."""

    async def _increment(state: AgentState) -> list:
        state.execution_meta.internal_data[_ROUNDS_KEY] = (
            state.execution_meta.internal_data.get(_ROUNDS_KEY, 0) + 1
        )
        return []

    _increment.__name__ = "increment_rounds"
    return _increment  # type: ignore


# ---------------------------------------------------------------------------
# Routing factory
# ---------------------------------------------------------------------------


def _make_worker_node(worker: BaseAgent) -> Callable:
    """Wrap *worker* so its request never ends on an assistant turn.

    From the second delegation of a turn on, the context ends with the previous worker's
    reply. In that case one transient user message (``WORKER_CONTINUE_NOTE``) is added to
    the request view. Nothing is added when the context ends with the user's message or a
    tool result. The message is never saved to state.
    """

    async def _worker(state: AgentState, config: dict) -> object:
        original_context = state.context
        last = original_context[-1] if original_context else None
        if last is not None and last.role == "assistant" and not last.tools_calls:
            state.context = [
                *original_context,
                Message.text_message(WORKER_CONTINUE_NOTE, role="user"),
            ]
        try:
            return await worker.execute(state, config)
        finally:
            state.context = original_context

    _worker.__name__ = "worker"
    _worker.agent = worker  # type: ignore[attr-defined]
    return _worker


def _make_worker_route(tool_node_name: str) -> Callable[[AgentState], str]:
    """Return a routing function for a worker node that has tools.

    Routes to *tool_node_name* when the worker emitted tool calls, and to
    PRE_SUPERVISOR when the worker produced a final text response.
    """

    def _route(state: AgentState) -> str:
        if not state.context:
            return _PRE_SUPERVISOR_NODE
        last = state.context[-1]
        if last.role == "assistant" and last.tools_calls and len(last.tools_calls) > 0:
            return tool_node_name
        return _PRE_SUPERVISOR_NODE

    return _route


def _resolve_decision(raw: str, worker_names: list[str], rounds: int) -> str:
    """Map a supervisor's raw response text to a worker name or ``END``.

    1. ``FINISH`` token → ``END``.
    2. First worker name matched with word boundaries → that worker (avoids
       e.g. worker ``CODE`` matching ``DECODE``).
    3. Nothing recognisable → ``END``.
    """
    if re.search(rf"\b{_FINISH_TOKEN}\b", raw):
        logger.debug("Supervisor signalled FINISH after %d round(s).", rounds)
        return END

    for name in worker_names:
        if re.search(rf"\b{re.escape(name.upper())}\b", raw):
            logger.debug("Supervisor routing to %s (round %d).", name, rounds)
            return name

    logger.warning("Supervisor response %r does not match any worker; terminating.", raw)
    return END


def _make_supervisor_node(
    supervisor_agent: Agent,
    worker_names: list[str],
    max_rounds: int,
    respond_node: str | None = None,
) -> Callable[[AgentState, dict[str, Any]], Any]:
    """Return the SUPERVISOR node.

    Rather than emitting its routing token as a normal assistant message (which
    would surface ``SCOUT`` / ``FINISH`` noise in the conversation and pollute
    the context seen by workers), this node runs the supervisor agent
    *internally*, reads its one-word decision, and returns a
    :class:`~tenxgraph.utils.command.Command` whose ``goto`` drives navigation.
    Because the ``Command`` carries no ``update``, nothing is streamed to the
    caller and nothing is persisted to ``state.context``.

    Decision order:

    1. Hard-cap check: if ``max_rounds`` reached → ``END``.
    2. Run the supervisor agent on :func:`_supervisor_view` and extract its decision.
    3. Map the decision to a worker name or ``END``. Naming the worker that just ran
       again counts as ``END``: a worker already runs its own tool loop, so an immediate
       repeat only re-answers.
    4. With ``respond_node`` set, an ``END`` before any worker ran this turn goes to
       ``respond_node`` instead, so the user always gets a reply (a greeting needs none
       of the workers).
    """

    async def _supervisor(state: AgentState, config: dict[str, Any]) -> Command:
        history: list[str] = state.execution_meta.internal_data.setdefault(_ROUTE_HISTORY_KEY, [])

        def finish() -> Command:
            if respond_node is not None and not history:
                return Command(goto=respond_node)
            return Command(goto=END)

        rounds = state.execution_meta.internal_data.get(_ROUNDS_KEY, 0)
        if rounds >= max_rounds:
            logger.warning("SupervisorTeam reached max_rounds=%d; terminating.", max_rounds)
            return finish()

        # Force a non-streaming call so the converter resolves to a single Message
        # and the routing token is never emitted as a visible stream chunk.
        exec_config = dict(config)
        exec_config["is_stream"] = False

        # The supervisor's decisions are never saved to the context (they would show the
        # user routing words). Its own call gets a transient view that says, in order,
        # which worker replied what and which workers already ran.
        original_context = state.context
        state.context = _supervisor_view(original_context, history)
        try:
            converter = await supervisor_agent.execute(state, exec_config)
            message = await converter.invoke()
        finally:
            state.context = original_context

        raw = (message.text() or "").strip().strip("`.,:;'\"").upper()
        decision = _resolve_decision(raw, worker_names, rounds)
        if decision != END and history and decision == history[-1]:
            logger.info("Supervisor named %s again right after it ran; finishing.", decision)
            decision = END
        if decision == END:
            return finish()
        history.append(decision)
        return Command(goto=decision)

    _supervisor.__name__ = "supervisor"
    return _supervisor


# ---------------------------------------------------------------------------
# SupervisorTeamAgent
# ---------------------------------------------------------------------------


class SupervisorTeamAgent[StateT: AgentState]:
    """Supervisor → Worker routing pattern.

    The SUPERVISOR ``Agent`` is built internally from *supervisor_model*.  Its
    system prompt is auto-generated to list all workers + their descriptions
    (override with *supervisor_system_prompt*).

    Each WORKER is a **pre-built** :class:`~tenxgraph.core.graph.agent.Agent`
    (via :class:`WorkerConfig`), so every worker can be independently
    configured — different models, tools, ``memory``, ``skills``, etc.

    Usage::

        from tenxgraph.core.graph import Agent, ToolNode
        from tenxgraph.prebuilt.agent import SupervisorTeamAgent
        from tenxgraph.prebuilt.agent.supervisor_team import WorkerConfig

        agent = SupervisorTeamAgent(
            supervisor_model="gpt-4o",
            workers={
                "RESEARCHER": WorkerConfig(
                    agent=Agent(model="gpt-4o-mini", tool_node=ToolNode([web_search])),
                    description="Searches the web for factual information.",
                ),
                "CODER": WorkerConfig(
                    agent=Agent(model="gpt-4o", tool_node=ToolNode([run_code])),
                    description="Writes and runs Python code.",
                ),
            },
            max_rounds=8,
        )
        app = agent.compile(checkpointer=...)

    Args:
        supervisor_model: LLM model for the supervisor agent.
        workers: Mapping of node names (UPPER-CASE recommended) to
            :class:`WorkerConfig`.
        supervisor_system_prompt: Override the auto-generated supervisor system
            prompt.  If ``None`` the prompt is built from the worker registry.
        max_rounds: Maximum number of supervisor → worker delegations per turn
            before the graph terminates (default ``10``).  The count resets on
            each new turn of a thread.
        state: Optional custom :class:`~tenxgraph.core.state.AgentState`
            subclass instance.
        context_manager: Optional custom context manager.
        publisher: Optional event publisher.
        id_generator: ID generation strategy.
        container: InjectQ dependency-injection container.
        **supervisor_kwargs: Extra keyword arguments forwarded to the supervisor
            ``Agent`` only (e.g. ``provider``, ``temperature``, ``retry_config``).
    """

    def __init__(
        self,
        supervisor_model: str,
        workers: dict[str, WorkerConfig],
        supervisor_system_prompt: list[dict[str, Any]] | None = None,
        max_rounds: int = 10,
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        publisher: BasePublisher | list[BasePublisher] | None = None,
        id_generator: BaseIDGenerator = DefaultIDGenerator(),
        container: InjectQ | None = None,
        **supervisor_kwargs: Any,
    ):
        if not workers:
            raise ValueError("SupervisorTeamAgent requires at least one worker.")
        for reserved in _RESERVED_NODES:
            if reserved in workers:
                raise ValueError(
                    f"Worker name {reserved!r} is reserved for the supervisor's own nodes."
                )

        self._supervisor_model = supervisor_model
        self._workers = workers
        self._max_rounds = max_rounds
        self._supervisor_kwargs = supervisor_kwargs

        # Resolve system prompt (auto-generated or user-supplied).
        self._supervisor_system_prompt: list[dict[str, Any]] = (
            supervisor_system_prompt
            if supervisor_system_prompt is not None
            else _build_supervisor_prompt(workers)
        )

        # Graph infra
        self._state = state
        self._context_manager = context_manager
        self._publisher = publisher
        self._id_generator = id_generator
        self._container = container

        self._graph: StateGraph[StateT] = self._new_graph()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _new_graph(self) -> StateGraph[StateT]:
        return StateGraph[StateT](
            state=self._state,
            context_manager=self._context_manager,
            publisher=self._publisher,
            id_generator=self._id_generator,
            container=self._container,
        )

    def _build_supervisor_agent(self) -> Agent:
        return Agent(
            model=self._supervisor_model,
            system_prompt=self._supervisor_system_prompt,
            **self._supervisor_kwargs,
        )

    def _build_respond_agent(self) -> Agent:
        """The RESPOND node's agent: answers the user when no worker is needed."""
        return Agent(
            model=self._supervisor_model,
            system_prompt=_build_respond_prompt(self._workers),
            **self._supervisor_kwargs,
        )

    def _configure_graph(self) -> None:
        self._graph = self._new_graph()

        worker_names = list(self._workers.keys())

        # --- SUPERVISOR node ---
        # Wrapped so the routing token (worker name / FINISH) is consumed
        # internally and drives navigation via Command.goto instead of leaking
        # into the conversation as an assistant message.
        supervisor_agent = self._build_supervisor_agent()
        self._graph.add_node(
            _SUPERVISOR_NODE,
            _make_supervisor_node(
                supervisor_agent, worker_names, self._max_rounds, respond_node=_RESPOND_NODE
            ),
        )

        # --- RESPOND node: the reply when the supervisor finishes before any worker ran ---
        self._graph.add_node(_RESPOND_NODE, self._build_respond_agent())
        self._graph.add_edge(_RESPOND_NODE, END)

        # --- PRE_SUPERVISOR node: increments round counter, then routes to SUPERVISOR ---
        self._graph.add_node(_PRE_SUPERVISOR_NODE, _make_increment_rounds_node())
        self._graph.add_edge(_PRE_SUPERVISOR_NODE, _SUPERVISOR_NODE)

        # --- WORKER nodes ---
        for name, cfg in self._workers.items():
            # Wrapped so a later worker's request never ends on the previous worker's
            # reply; the agent itself is ``node.func.agent``.
            self._graph.add_node(name, _make_worker_node(cfg.agent))

            worker_tool_node = cfg.agent.get_tool_node()
            if worker_tool_node is not None:
                # Wire a mini react-loop: WORKER → WORKER_TOOL → WORKER
                # The worker routes to PRE_SUPERVISOR only when it has no more tool calls.
                tool_node_name = f"{name}_TOOL"
                self._graph.add_node(tool_node_name, worker_tool_node)
                self._graph.add_edge(tool_node_name, name)
                self._graph.add_conditional_edges(
                    name,
                    _make_worker_route(tool_node_name),
                    {tool_node_name: tool_node_name, _PRE_SUPERVISOR_NODE: _PRE_SUPERVISOR_NODE},
                )
            else:
                # No tools — worker returns via PRE_SUPERVISOR to SUPERVISOR.
                self._graph.add_edge(name, _PRE_SUPERVISOR_NODE)

        # No static edges leave SUPERVISOR: it routes dynamically via Command.goto
        # to a worker node or END. Worker nodes are still reachable (they appear as
        # the source of their own outgoing edges), so graph validation passes.

        # --- Entry point ---
        self._graph.set_entry_point(_SUPERVISOR_NODE)

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
        """Wire the graph and return a compiled, ready-to-invoke graph.

        Args:
            checkpointer: Persistence backend.
            store: Long-term key-value store.
            interrupt_before: Node names to pause before.
            interrupt_after: Node names to pause after.
            callback_manager: Observability hooks.
            media_store: Media/file storage backend.
            shutdown_timeout: Graceful-shutdown timeout in seconds.
            media_policy: Which external media URLs the server may download.
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
