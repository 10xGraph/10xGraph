"""SwarmAgent — peer-to-peer multi-agent handoff pattern.

Pattern::

    TRIAGE --[transfer_to_researcher]--> TRIAGE_HANDOFF --> RESEARCHER
    TRIAGE --[transfer_to_writer]-----> TRIAGE_HANDOFF --> WRITER
    RESEARCHER --[transfer_to_writer]--> RESEARCHER_HANDOFF --> WRITER
    RESEARCHER --[no handoff]---------> END
    WRITER --[no handoff]-------------> END

Each member is a **pre-built** :class:`~tenxgraph.core.graph.agent.Agent` (or
any :class:`~tenxgraph.core.graph.base_agent.BaseAgent` subclass) configured
independently by the caller.  ``SwarmAgent`` auto-generates
``transfer_to_<name>`` handoff tools and injects them into each member's
:class:`~tenxgraph.core.graph.tool_node.ToolNode` so the LLM can call them.

After each member runs, a per-member routing function inspects
``state.context[-1].tools_calls`` via :func:`is_handoff_tool`.  If a handoff
tool call is found the graph routes to the member's ``<NAME>_HANDOFF`` node,
which answers every tool call of that message with a tool result ("Transferred
to <target>." for the handoff, "Not run: ..." for a second handoff) and then routes
to the target.  When the same message also holds regular tool calls (models such
as OpenAI call ``get_weather`` and ``transfer_to_planner`` together), it goes to
``<NAME>_TOOL`` instead, which runs those calls, answers the handoff and jumps to
the target, so the target sees their results.  The target therefore never inherits
an unanswered tool call, which providers reject.  Without a handoff, regular tool
calls go to ``<NAME>_TOOL`` and a plain reply goes to ``END``.

Example::

    from tenxgraph.core.graph import Agent, ToolNode
    from tenxgraph.prebuilt.agent import SwarmAgent
    from tenxgraph.prebuilt.agent.swarm import SwarmMemberConfig


    def web_search(query: str) -> str: ...
    def draft_document(topic: str) -> str: ...


    triage_agent = Agent(
        model="gpt-4o-mini",
        system_prompt=[{"role": "system", "content": "Route requests."}],
    )
    researcher_agent = Agent(
        model="gpt-4o",
        tool_node=ToolNode([web_search]),
        memory=MemoryConfig(...),  # each member configured independently
    )
    writer_agent = Agent(
        model="gpt-4o-mini",
        tool_node=ToolNode([draft_document]),
        skills=SkillConfig(...),
    )

    swarm = SwarmAgent(
        members={
            "TRIAGE": SwarmMemberConfig(
                agent=triage_agent,
                can_handoff_to=["RESEARCHER", "WRITER"],
                description="Triages requests and routes to the right specialist.",
            ),
            "RESEARCHER": SwarmMemberConfig(
                agent=researcher_agent,
                can_handoff_to=["WRITER"],
                description="Performs deep research.",
            ),
            "WRITER": SwarmMemberConfig(
                agent=writer_agent,
                description="Writes final documents.",
            ),
        },
        entry="TRIAGE",
    )
    app = swarm.compile(checkpointer=...)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from injectq import InjectQ

from tenxgraph.core.graph.base_agent import BaseAgent
from tenxgraph.core.graph.compiled_graph import CompiledGraph
from tenxgraph.core.graph.state_graph import StateGraph
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.graph.utils.tool_dispatch import (
    allowed_tool_names,
    handoff_tool_messages,
    tool_call_name,
)
from tenxgraph.core.state.agent_state import AgentState
from tenxgraph.core.state.base_context import BaseContextManager
from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.prebuilt.tools.handoff import create_handoff_tool, is_handoff_tool
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.storage.checkpointer.base_checkpointer import BaseCheckpointer
from tenxgraph.storage.media.storage.base import BaseMediaStore
from tenxgraph.storage.media.url_policy import MediaFetchPolicy
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.utils.callbacks import CallbackManager
from tenxgraph.utils.constants import END
from tenxgraph.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


logger = logging.getLogger("tenxgraph.prebuilt.swarm")

StateT = TypeVar("StateT", bound=AgentState)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class SwarmMemberConfig:
    """Configuration for a single member of a :class:`SwarmAgent`.

    Each member is a **pre-built** agent instance so that callers can
    independently configure ``skills``, ``memory``, ``multimodal_config``,
    ``retry_config``, or any other per-member option — without those settings
    being forced on every member.

    Args:
        agent: A fully-configured :class:`~tenxgraph.core.graph.agent.Agent`
            (or any :class:`~tenxgraph.core.graph.base_agent.BaseAgent`
            subclass).  **Do not** include handoff tools in its
            :class:`~tenxgraph.core.graph.tool_node.ToolNode`; they are
            injected automatically by :class:`SwarmAgent`.
        can_handoff_to: Names of other members this member may hand off to.
            ``None`` means it can hand off to *all* other members.
        description: Short description injected into the handoff tool
            docstrings of **other** members that target this member, so the
            LLM understands when to route here.
    """

    agent: BaseAgent
    can_handoff_to: list[str] | None = None  # None = all others
    description: str = ""


# ---------------------------------------------------------------------------
# Internal routing factory
# ---------------------------------------------------------------------------


HANDOFF_NODE_SUFFIX = "_HANDOFF"


def _find_member_handoff(
    message: Message, allowed_by_upper: dict[str, str]
) -> tuple[int, str] | None:
    """Return ``(call index, target member)`` of the first allowed handoff call, or None.

    Member names are matched case-insensitively and returned as registered. A call to a
    tool the model was not offered (``Agent.tools_tags``) does not count, as in the ToolNode.
    """
    if message.role != "assistant" or not message.tools_calls:
        return None
    offered = allowed_tool_names(message)
    for index, tc in enumerate(message.tools_calls):
        tool_name = tool_call_name(tc)
        if offered is not None and tool_name not in offered:
            continue
        is_handoff, target = is_handoff_tool(tool_name)
        real = allowed_by_upper.get(target.upper()) if is_handoff and target else None
        if real is not None:
            return index, real
    return None


def _has_regular_calls(message: Message) -> bool:
    """True when ``message`` has an offered tool call that is not a handoff."""
    offered = allowed_tool_names(message)
    for tc in message.tools_calls or []:
        tool_name = tool_call_name(tc)
        if offered is not None and tool_name not in offered:
            continue
        if not is_handoff_tool(tool_name)[0]:
            return True
    return False


def _last_assistant_index(state: AgentState) -> int | None:
    for index in range(len(state.context) - 1, -1, -1):
        if state.context[index].role == "assistant":
            return index
    return None


def _make_handoff_node(
    node_name: str,
    allowed_targets: list[str],
) -> Callable[[AgentState], Any]:
    """Return the ``<NAME>_HANDOFF`` node: answer the handoff message's tool calls.

    The member's last assistant message holds the handoff call. Every tool call of it gets
    a tool result (the shared ToolNode wording), so the target agent's request never ends
    on an unanswered call. The results are returned as the node's messages, so they are
    saved to state and to the messages table like any other node output. Calls that are
    already answered (a node re-run after a resume) are skipped.
    """
    allowed_by_upper = {t.upper(): t for t in allowed_targets}

    async def _handoff(state: AgentState) -> list[Message]:
        index = _last_assistant_index(state)
        if index is None:
            return []
        message = state.context[index]
        found = _find_member_handoff(message, allowed_by_upper)
        if found is None:
            return []
        winner, target = found
        answered = {
            block.call_id
            for later in state.context[index + 1 :]
            for block in later.content or []
            if isinstance(block, ToolResultBlock)
        }
        answers = []
        for answer in handoff_tool_messages(message, winner, target):
            call_ids = {b.call_id for b in answer.content if isinstance(b, ToolResultBlock)}
            if call_ids & answered:
                continue
            answer.metadata.setdefault("node_name", node_name)
            answers.append(answer)
        logger.info("Swarm handoff: %s -> %s", node_name, target)
        return answers

    _handoff.__name__ = "handoff"
    return _handoff


def _make_handoff_route(allowed_targets: list[str]) -> Callable[[AgentState], str]:
    """Route a ``<NAME>_HANDOFF`` node to the target its member's handoff call named."""
    allowed_by_upper = {t.upper(): t for t in allowed_targets}

    def _route(state: AgentState) -> str:
        index = _last_assistant_index(state)
        if index is None:
            return END
        found = _find_member_handoff(state.context[index], allowed_by_upper)
        return found[1] if found else END

    return _route


def _make_member_route(
    name: str,
    allowed_targets: list[str],
    tool_node_name: str | None,
    handoff_node_name: str | None = None,
) -> Callable[[AgentState], str]:
    """Return a routing function for the given member node.

    Priority:
    1. Handoff tool call (``transfer_to_<target>``) → *handoff_node_name*, which answers
       the calls and then routes to the target (or straight to the target when
       *handoff_node_name* is None).
    2. Regular tool call → route to *tool_node_name* for execution.
    3. No tool calls → ``END``.
    """
    # Member names are matched case-insensitively and returned as registered.
    allowed_by_upper = {t.upper(): t for t in allowed_targets}

    def _route(state: AgentState) -> str:
        if not state.context:
            return END

        last = state.context[-1]

        if last.role != "assistant" or not last.tools_calls:
            return END

        # Handoff tool calls are never executed by the ToolNode: the handoff node
        # answers them and routes to the target.
        found = _find_member_handoff(last, allowed_by_upper)
        if found is not None:
            logger.debug("Member %s handing off to %s.", name, found[1])
            if tool_node_name is not None and _has_regular_calls(last):
                # Other tool calls in the same message (get_weather + transfer_to_planner)
                # run first: the member's ToolNode runs them, answers the handoff and jumps
                # to the target, so the target sees their results.
                return tool_node_name
            return handoff_node_name or found[1]

        # Regular tool calls need ToolNode execution.
        if tool_node_name is not None:
            return tool_node_name

        return END

    return _route


# ---------------------------------------------------------------------------
# SwarmAgent
# ---------------------------------------------------------------------------


class SwarmAgent[StateT: AgentState]:
    """Peer-to-peer multi-agent handoff pattern.

    Each member is a **pre-built** :class:`~tenxgraph.core.graph.agent.Agent`
    so every member can be configured independently (different models,
    ``skills``, ``memory``, ``multimodal_config``, ``retry_config``, etc.).

    :class:`SwarmAgent` automatically:

    * generates ``transfer_to_<name>`` handoff tools for each member's
      allowed targets and injects them into the member's
      :class:`~tenxgraph.core.graph.tool_node.ToolNode`;
    * wires per-member conditional edges that inspect the last assistant
      message for a handoff tool call and route accordingly;
    * adds a ``<NAME>_HANDOFF`` node per member with handoff targets, which
      answers the handoff message's tool calls before the target runs.

    Usage::

        from tenxgraph.core.graph import Agent, ToolNode
        from tenxgraph.prebuilt.agent import SwarmAgent
        from tenxgraph.prebuilt.agent.swarm import SwarmMemberConfig

        swarm = SwarmAgent(
            members={
                "TRIAGE": SwarmMemberConfig(
                    agent=Agent(model="gpt-4o-mini"),
                    can_handoff_to=["RESEARCHER"],
                    description="Routes requests.",
                ),
                "RESEARCHER": SwarmMemberConfig(
                    agent=Agent(
                        model="gpt-4o",
                        tool_node=ToolNode([web_search]),
                        memory=MemoryConfig(...),
                    ),
                    description="Performs research.",
                ),
            },
            entry="TRIAGE",
        )
        app = swarm.compile(checkpointer=...)

    Args:
        members: Mapping of node names (UPPER-CASE recommended) to
            :class:`SwarmMemberConfig`.
        entry: Name of the member that receives the first message.
        state: Optional custom :class:`~tenxgraph.core.state.AgentState`
            subclass instance.
        context_manager: Optional custom context-trimming manager.
        publisher: Optional event publisher.
        id_generator: ID generation strategy.
        container: InjectQ dependency-injection container.
    """

    def __init__(
        self,
        members: dict[str, SwarmMemberConfig],
        entry: str,
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        publisher: BasePublisher | list[BasePublisher] | None = None,
        id_generator: BaseIDGenerator = DefaultIDGenerator(),
        container: InjectQ | None = None,
    ):
        if not members:
            raise ValueError("SwarmAgent requires at least one member.")
        if entry not in members:
            raise ValueError(f"entry={entry!r} is not in members. Available: {list(members)}")

        upper_names = [n.upper() for n in members]
        if len(set(upper_names)) != len(upper_names):
            raise ValueError("SwarmAgent member names must be unique ignoring case.")
        generated = {
            f"{n}{suffix}".upper() for n in members for suffix in ("_TOOL", HANDOFF_NODE_SUFFIX)
        }
        clashes = [n for n in members if n.upper() in generated]
        if clashes:
            raise ValueError(
                f"SwarmAgent member names {clashes} clash with the generated "
                f"<NAME>_TOOL / <NAME>{HANDOFF_NODE_SUFFIX} nodes; rename them."
            )

        self._members = members
        self._entry = entry

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

    def _resolve_targets(self, name: str) -> list[str]:
        """Return the allowed handoff targets for *name*, as registered member names."""
        cfg = self._members[name]
        if cfg.can_handoff_to is None:
            return [n for n in self._members if n != name]
        by_upper = {n.upper(): n for n in self._members}
        return [by_upper.get(t.upper(), t) for t in cfg.can_handoff_to]

    def _build_handoff_tools(self, name: str) -> list[Callable]:
        """Generate ``create_handoff_tool`` callables for each allowed target."""
        targets = self._resolve_targets(name)
        tools = []
        for target in targets:
            target_cfg = self._members.get(target)
            description = (
                target_cfg.description
                if target_cfg and target_cfg.description
                else f"Transfer control to {target}."
            )
            tool = create_handoff_tool(agent_name=target.lower(), description=description)
            # The tool name stays transfer_to_<lower>; the target is the registered node
            # name, so a ToolNode that ever runs it routes to a node that exists.
            tool.__target_agent__ = target  # type: ignore[attr-defined]
            tools.append(tool)
        return tools

    def _inject_handoff_tools(self, name: str) -> None:
        """Inject handoff tools into the member agent's ToolNode.

        If the agent has no ``tool_node`` yet, a new
        :class:`~tenxgraph.core.graph.tool_node.ToolNode` is created for the
        handoff tools.  If it already has one, each handoff tool is added via
        :meth:`~tenxgraph.core.graph.tool_node.ToolNode.add_tool`.

        Note:
            The agent's ``tool_node`` is mutated in-place.  This is safe
            because the agent was passed in solely to be used inside this
            swarm.
        """
        handoff_tools = self._build_handoff_tools(name)
        if not handoff_tools:
            return

        agent = self._members[name].agent
        if agent.tool_node is None:
            new_tn = ToolNode(handoff_tools)
            agent.tool_node = new_tn
            agent._tool_node = new_tn  # type: ignore
        elif isinstance(agent.tool_node, ToolNode):
            for tool in handoff_tools:
                agent.tool_node.add_tool(tool)
        else:
            # tool_node is a string (named graph-node reference) — unsupported
            logger.warning(
                "Member %r has a string tool_node reference; handoff tools cannot "
                "be injected.  Handoffs from this member will not work.",
                name,
            )

    def _configure_graph(self) -> None:
        self._graph = self._new_graph()

        all_names = list(self._members.keys())

        # Record which members have regular (non-handoff) tools BEFORE handoff
        # injection, so we know whether a TOOL node is needed.
        has_regular_tools: dict[str, bool] = {
            name: cfg.agent.get_tool_node() is not None for name, cfg in self._members.items()
        }

        # Inject handoff tools and register member nodes.
        for name, cfg in self._members.items():
            self._inject_handoff_tools(name)
            self._graph.add_node(name, cfg.agent)

        # Wire edges for each member.
        for name in all_names:
            allowed_targets = self._resolve_targets(name)

            tool_node_name: str | None = None
            if has_regular_tools[name]:
                # Register the member's ToolNode (now contains both regular and
                # handoff tools) as a dedicated graph node and loop it back.
                tool_node_name = f"{name}_TOOL"
                self._graph.add_node(tool_node_name, self._members[name].agent.get_tool_node())
                self._graph.add_edge(tool_node_name, name)

            if not allowed_targets and tool_node_name is None:
                logger.warning(
                    "SwarmAgent member %r has no handoff targets and no tools — "
                    "it will always route directly to END. "
                    "Set can_handoff_to or add tools to this member.",
                    name,
                )
                self._graph.add_edge(name, END)
                continue

            handoff_node_name: str | None = None
            path_map: dict[str, str] = {END: END}
            if allowed_targets:
                # MEMBER -> MEMBER_HANDOFF -> target: the handoff node answers the
                # handoff message's tool calls before the target runs.
                handoff_node_name = f"{name}{HANDOFF_NODE_SUFFIX}"
                self._graph.add_node(
                    handoff_node_name, _make_handoff_node(handoff_node_name, allowed_targets)
                )
                handoff_map: dict[str, str] = {END: END}
                for target in allowed_targets:
                    handoff_map[target] = target
                self._graph.add_conditional_edges(
                    handoff_node_name, _make_handoff_route(allowed_targets), handoff_map
                )
                path_map[handoff_node_name] = handoff_node_name
            if tool_node_name is not None:
                path_map[tool_node_name] = tool_node_name

            route_fn = _make_member_route(name, allowed_targets, tool_node_name, handoff_node_name)
            self._graph.add_conditional_edges(name, route_fn, path_map)

        self._graph.set_entry_point(self._entry)

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
