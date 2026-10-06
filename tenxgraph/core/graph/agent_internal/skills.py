"""Skills support for Agent."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agentflow.core.graph.tool_node import ToolNode


if TYPE_CHECKING:
    from agentflow.core.skills.models import SkillConfig
    from agentflow.core.skills.registry import SkillsRegistry


logger = logging.getLogger("agentflow.agent")


class AgentSkillsMixin:
    """Skills registration helpers for Agent."""

    # Instance attributes set by _setup_skills
    _skills_config: SkillConfig | None
    _skills_registry: SkillsRegistry | None
    _skill_catalog_prompt: dict[str, Any] | None
    _skill_files_readable: bool
    _tool_node: ToolNode | None

    def _setup_skills(self, skills: SkillConfig | None) -> None:
        """Initialize skills infrastructure if a SkillConfig is provided.

        Discovers skills, then registers the skill tools on the agent's ToolNode:
        ``activate_skill`` (on-demand mode) and ``read_skill_resource`` (when any
        skill bundles files). No tool or catalog is added when no skill is found.

        Args:
            skills: Optional SkillConfig instance with skills_dir and options.
        """
        self._skills_config = None
        self._skills_registry = None
        self._skill_catalog_prompt = None
        self._skill_files_readable = False

        if skills is None:
            return

        from agentflow.core.skills.activation import (
            build_catalog_prompt,
            has_bundled_files,
            make_activate_skill_tool,
            make_read_skill_resource_tool,
        )
        from agentflow.core.skills.models import SkillConfig
        from agentflow.core.skills.registry import SkillsRegistry

        if not isinstance(skills, SkillConfig):
            raise TypeError(f"Expected SkillConfig, got {type(skills)}")

        config = skills
        registry = SkillsRegistry()
        registry.discover(config.skill_dirs)
        self._skills_config = config
        self._skills_registry = registry

        if len(registry) == 0:
            logger.warning(
                "Skills enabled but no skills were discovered in %s", config.skill_dirs or "[]"
            )
            return

        can_read_files = has_bundled_files(registry)
        tools: list[Callable] = []
        if config.mode == "on-demand":
            tools.append(
                make_activate_skill_tool(
                    registry,
                    config,
                    embed_catalog=not config.inject_catalog,
                    can_read_files=can_read_files,
                )
            )
        if can_read_files:
            tools.append(make_read_skill_resource_tool(registry, config))

        # Session mode works without a ToolNode; its skill files are then simply
        # not readable. On-demand mode cannot work without one.
        self._skill_files_readable = (
            self._attach_skill_tools(tools, required=config.mode == "on-demand") and can_read_files
        )

        if config.mode == "on-demand" and config.inject_catalog:
            catalog = build_catalog_prompt(registry, can_read_files=can_read_files)
            if catalog:
                self._skill_catalog_prompt = {"role": "system", "content": catalog}

        logger.info("Skills enabled (%s mode): %d skill(s) discovered", config.mode, len(registry))

    def _attach_skill_tools(self, tools: list[Callable], *, required: bool) -> bool:
        """Add *tools* to the agent's ToolNode. Returns False if there is none.

        If the agent was configured with a named ToolNode reference
        (``tool_node="TOOL"``), the tools are queued until the actual ToolNode
        is resolved at execution time.
        """
        if not tools:
            return True
        if self._tool_node is not None:
            for tool in tools:
                self._tool_node.add_tool(tool)
            return True
        if getattr(self, "tool_node_name", None) is not None:
            extra = getattr(self, "_extra_tools", None)
            if extra is None:
                self._extra_tools = list(tools)
            else:
                extra.extend(tools)
            return True
        if required:
            raise RuntimeError(
                "Skills require an existing ToolNode when skills are enabled. "
                "Provide a ToolNode to the Agent before configuring skills."
            )
        logger.warning(
            "Skills in session mode bundle files but the agent has no ToolNode; "
            "read_skill_resource is not available, so those files cannot be read."
        )
        return False

    def _build_skill_prompts(
        self,
        state: Any,
        system_prompt: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build effective system prompts with skill content if configured.

        For ``on-demand`` mode: appends the skill catalog (if enabled), and
        re-injects any activated skill whose instructions are no longer in
        the conversation (for example after context trimming).
        For ``session`` mode: reads ``state.<preload_from>``, loads the
        matching skill and appends it as a system message.

        Args:
            state: Current AgentState.
            system_prompt: Base system prompt list.

        Returns:
            Effective system prompt list with skill content appended if configured.
        """
        effective_system_prompt = list(system_prompt)

        config = self._skills_config
        registry = self._skills_registry
        if not config or not registry or len(registry) == 0:
            return effective_system_prompt

        from agentflow.core.skills.activation import (
            format_skill_content,
            get_active_skills,
            skill_in_context,
        )

        if config.mode == "session":
            field = config.preload_from
            skill_name: str = ""
            if field and state is not None:
                skill_name = getattr(state, field, None) or ""

            if not skill_name:
                logger.warning(
                    "Skills in session mode require a skill name in state.%s; no skill "
                    "content injected",
                    field,
                )
                return effective_system_prompt

            content = format_skill_content(
                registry, skill_name, config, can_read_files=self._skill_files_readable
            )
            if content:
                effective_system_prompt.append({"role": "system", "content": content})
            else:
                logger.warning("Session skill '%s' not found or empty", skill_name)
            return effective_system_prompt

        # ── on-demand mode ───────────────────────────────────────────────────
        if self._skill_catalog_prompt is not None:
            effective_system_prompt.append(self._skill_catalog_prompt)

        for name in get_active_skills(state):
            if name not in registry or skill_in_context(state, name):
                continue
            content = format_skill_content(
                registry, name, config, can_read_files=self._skill_files_readable
            )
            if content:
                effective_system_prompt.append({"role": "system", "content": content})

        return effective_system_prompt
