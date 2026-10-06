"""SwarmAgent — peer-to-peer multi-agent handoff pattern.

Pattern::

    TRIAGE --[transfer_to_researcher]--> RESEARCHER
    TRIAGE --[transfer_to_writer]-----> WRITER
    RESEARCHER --[transfer_to_writer]--> WRITER
    RESEARCHER --[no handoff]---------> END
    WRITER --[no handoff]-------------> END

Each member is a **pre-built** :class:`~agentflow.core.graph.agent.Agent` (or
any :class:`~agentflow.core.graph.base_agent.BaseAgent` subclass) configured
independently by the caller.  ``SwarmAgent`` auto-generates
``transfer_to_<name>`` handoff tools and injects them into each member's
:class:`~agentflow.core.graph.tool_node.ToolNode` so the LLM can call them.

After each member runs, a per-member routing function inspects
``state.context[-1].tools_calls`` via :func:`is_handoff_tool`.  If a handoff
tool call is found the graph routes to that target; otherwise control goes to
``END``.

Example::

    from agentflow.core.graph import Agent, ToolNode
    from agentflow.prebuilt.agent import SwarmAgent
    from agentflow.prebuilt.agent.swarm import SwarmMemberConfig


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
from typing import TypeVar

from injectq import InjectQ

from agentflow.core.graph.base_agent import BaseAgent
from agentflow.core.graph.compiled_graph import CompiledGraph
from agentflow.core.graph.state_graph import StateGraph
from agentflow.core.graph.tool_node import ToolNode
from agentflow.core.state.agent_state import AgentState
from agentflow.core.state.base_context import BaseContextManager
from agentflow.prebuilt.tools.handoff import create_handoff_tool, is_handoff_tool
from agentflow.runtime.publisher.base_publisher import BasePublisher
from agentflow.storage.checkpointer.base_checkpointer import BaseCheckpointer
from agentflow.storage.media.storage.base import BaseMediaStore
from agentflow.storage.store.base_store import BaseStore
from agentflow.utils.callbacks import CallbackManager
from agentflow.utils.constants import END
from agentflow.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


logger = logging.getLogger("agentflow.prebuilt.swarm")

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
        agent: A fully-configured :class:`~agentflow.core.graph.agent.Agent`
            (or any :class:`~agentflow.core.graph.base_agent.BaseAgent`
            subclass).  **Do not** include handoff tools in its
            :class:`~agentflow.core.graph.tool_node.ToolNode`; they are
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


def _make_member_route(
    name: str,
    allowed_targets: list[str],
    tool_node_name: str | None,
) -> Callable[[AgentState], str]:
    """Return a routing function for the given member node.

    Priority:
    1. Handoff tool call (``transfer_to_<target>``) → route to target member.
    2. Regular tool call → route to *tool_node_name* for execution.
    3. No tool calls → ``END``.
    """
    allowed_set = set(allowed_targets)

    def _route(state: AgentState) -> str:
        if not state.context:
            return END

        last = state.context[-1]

        if last.role != "assistant" or not last.tools_calls:
            return END

        # Handoff tool calls are intercepted here; they are never executed by
        # the ToolNode — the graph routing IS the handoff.
        for tc in last.tools_calls:
            is_handoff, target = is_handoff_tool(tc.get("name", ""))
            if is_handoff and target and target.upper() in allowed_set:
                logger.debug("Member %s handing off to %s.", name, target.upper())
                return target.upper()

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

    Each member is a **pre-built** :class:`~agentflow.core.graph.agent.Agent`
    so every member can be configured independently (different models,
    ``skills``, ``memory``, ``multimodal_config``, ``retry_config``, etc.).

    :class:`SwarmAgent` automatically:

    * generates ``transfer_to_<name>`` handoff tools for each member's
      allowed targets and injects them into the member's
      :class:`~agentflow.core.graph.tool_node.ToolNode`;
    * wires per-member conditional edges that inspect the last assistant
      message for a handoff tool call and route accordingly.

    Usage::

        from agentflow.core.graph import Agent, ToolNode
        from agentflow.prebuilt.agent import SwarmAgent
        from agentflow.prebuilt.agent.swarm import SwarmMemberConfig

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
        state: Optional custom :class:`~agentflow.core.state.AgentState`
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
        """Return the uppercase list of allowed handoff targets for *name*."""
        cfg = self._members[name]
        if cfg.can_handoff_to is None:
            return [n for n in self._members if n != name]
        return [t.upper() for t in cfg.can_handoff_to]

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
            tools.append(create_handoff_tool(agent_name=target.lower(), description=description))
        return tools

    def _inject_handoff_tools(self, name: str) -> None:
        """Inject handoff tools into the member agent's ToolNode.

        If the agent has no ``tool_node`` yet, a new
        :class:`~agentflow.core.graph.tool_node.ToolNode` is created for the
        handoff tools.  If it already has one, each handoff tool is added via
        :meth:`~agentflow.core.graph.tool_node.ToolNode.add_tool`.

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

            route_fn = _make_member_route(name, allowed_targets, tool_node_name)
            path_map: dict[str, str] = {END: END}
            for target in allowed_targets:
                path_map[target] = target
            if tool_node_name is not None:
                path_map[tool_node_name] = tool_node_name

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
