"""Skill tools and prompt helpers for Agentflow.

Implements the activation side of the Agent Skills client guide
(https://agentskills.io/client-implementation/adding-skills-support):

* ``activate_skill`` loads a skill's instructions, wrapped in
  ``<skill_content>`` tags together with the list of bundled files.
* ``read_skill_resource`` reads one bundled file (reference docs, scripts,
  templates, data) on demand.

Activated skills are recorded in ``state.execution_meta.internal_data`` so the
agent can re-inject their instructions if context trimming or summarisation
later drops the tool result that carried them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from html import escape
from typing import TYPE_CHECKING, Any, Literal

from tenxgraph.core.graph.tool_node.constants import SKILL_TOOL_ATTR
from tenxgraph.core.state import AgentState, ToolResult

from .loader import SkillResourceError


if TYPE_CHECKING:
    from .models import SkillConfig
    from .registry import SkillsRegistry


logger = logging.getLogger("tenxgraph.skills.activation")

ACTIVATE_SKILL_TOOL = "activate_skill"
READ_SKILL_RESOURCE_TOOL = "read_skill_resource"

# Key in ``execution_meta.internal_data`` listing skills activated in the thread.
ACTIVE_SKILLS_KEY = "active_skills"

_MAX_FILES_IN_ERROR = 50


def skill_content_marker(name: str) -> str:
    """Opening tag that identifies a skill's instructions in the conversation."""
    return f'<skill_content name="{escape(name)}">'


def format_skill_content(
    registry: SkillsRegistry,
    name: str,
    config: SkillConfig,
    *,
    can_read_files: bool,
) -> str | None:
    """Render a skill's instructions for the model.

    Returns ``None`` when the skill is unknown or its body cannot be read.
    """
    meta = registry.get(name)
    if meta is None:
        return None
    body = registry.load_content(name, hot_reload=config.hot_reload)
    if not body:
        return None

    parts = [skill_content_marker(name), body, ""]
    if meta.compatibility:
        parts.append(f"Compatibility: {meta.compatibility}")
    if config.include_skill_path:
        parts.append(f"Skill directory: {meta.skill_dir}")

    files, truncated = registry.list_files(name)
    if files:
        note = "Relative paths in this skill are relative to the skill directory."
        if can_read_files:
            note += (
                f' Read a bundled file with {READ_SKILL_RESOURCE_TOOL}(skill_name="{name}", '
                'path="<relative path>").'
            )
        parts.append(note)
        parts.append("")
        parts.append("<skill_resources>")
        parts.extend(f"  <file>{escape(path, quote=False)}</file>" for path in files)
        if truncated:
            parts.append("  <!-- more files exist; the listing is truncated -->")
        parts.append("</skill_resources>")
    parts.append("</skill_content>")
    return "\n".join(parts)


# -- activation tracking ------------------------------------------------------


def _internal_data(state: Any) -> dict[str, Any] | None:
    execution_meta = getattr(state, "execution_meta", None)
    data = getattr(execution_meta, "internal_data", None)
    return data if isinstance(data, dict) else None


def get_active_skills(state: Any) -> list[str]:
    """Names of skills activated in this thread, in activation order."""
    data = _internal_data(state)
    if data is None:
        return []
    value = data.get(ACTIVE_SKILLS_KEY)
    if not isinstance(value, list):
        return []
    # Parallel activations of the same skill can record it twice.
    return list(dict.fromkeys(str(item) for item in value))


def mark_skill_active(state: Any, name: str) -> None:
    """Record *name* as activated in *state*."""
    data = _internal_data(state)
    if data is None:
        return
    active = data.setdefault(ACTIVE_SKILLS_KEY, [])
    if name not in active:
        active.append(name)


def skill_in_context(state: Any, name: str) -> bool:
    """True when the skill's instructions are still in ``state.context``."""
    marker = skill_content_marker(name)
    for message in getattr(state, "context", None) or []:
        try:
            if marker in message.text():
                return True
        except AttributeError:
            continue
    return False


# -- prompts ------------------------------------------------------------------


def build_catalog_prompt(registry: SkillsRegistry, *, can_read_files: bool) -> str:
    """System-prompt section listing available skills and how to load them."""
    catalog = registry.build_catalog()
    if not catalog:
        return ""
    lines = [
        "## Skills",
        "",
        (
            "The following skills provide specialized instructions for specific tasks. "
            "When a task matches a skill's description, call the "
            f"`{ACTIVATE_SKILL_TOOL}` tool with the skill's name to load its full "
            "instructions before proceeding."
        ),
    ]
    if can_read_files:
        lines.append(
            "Skills may bundle extra files (references, scripts, assets). When a skill's "
            f"instructions point to one, read it with the `{READ_SKILL_RESOURCE_TOOL}` tool."
        )
    lines.extend(["", catalog])
    return "\n".join(lines)


# -- tools --------------------------------------------------------------------


def _skill_name_type(registry: SkillsRegistry) -> Any:
    """``Literal`` of the registered names, so the tool schema is an enum."""
    return Literal[tuple(registry.names())]  # type: ignore[valid-type]


def make_activate_skill_tool(
    registry: SkillsRegistry,
    config: SkillConfig,
    *,
    embed_catalog: bool,
    can_read_files: bool,
) -> Callable:
    """Create the ``activate_skill`` tool.

    Args:
        registry: Registry holding the skills.
        config: Skills configuration.
        embed_catalog: List the skills in the tool description (used when the
            catalog is not in the system prompt).
        can_read_files: Whether ``read_skill_resource`` is registered.
    """

    def activate_skill(skill_name: str, state: AgentState | None = None) -> str | ToolResult:
        if skill_name not in registry:
            available = ", ".join(registry.names())
            logger.warning("Unknown skill requested: %r", skill_name)
            return ToolResult(
                message=f"Unknown skill '{skill_name}'. Available skills: {available}",
                is_error=True,
            )

        if state is not None and skill_in_context(state, skill_name):
            mark_skill_active(state, skill_name)
            return (
                f"Skill '{skill_name}' is already active; its instructions are earlier in "
                "this conversation. Follow them."
            )

        content = format_skill_content(registry, skill_name, config, can_read_files=can_read_files)
        if content is None:
            return ToolResult(
                message=f"Skill '{skill_name}' was found but its SKILL.md could not be read.",
                is_error=True,
            )

        if state is not None:
            mark_skill_active(state, skill_name)
        logger.info("Skill activated: '%s'", skill_name)
        return content

    description = (
        "Load the full instructions of a skill. Call this when the user's task matches a "
        "skill's description, before working on the task, then follow the instructions it "
        "returns."
    )
    if embed_catalog:
        skill_list = "\n".join(
            f"- {meta.name}: {' '.join(meta.description.split())}"
            for meta in sorted(registry.get_all(), key=lambda s: (-s.priority, s.name))
        )
        description += f"\n\nAvailable skills:\n{skill_list}"
    activate_skill.__doc__ = (
        f"{description}\n\nArgs:\n    skill_name: Name of the skill to activate."
    )
    activate_skill.__annotations__ = {
        "skill_name": _skill_name_type(registry),
        "state": AgentState | None,
        "return": str | ToolResult,
    }
    setattr(activate_skill, SKILL_TOOL_ATTR, True)
    return activate_skill


def make_read_skill_resource_tool(registry: SkillsRegistry, config: SkillConfig) -> Callable:
    """Create the ``read_skill_resource`` tool."""

    def read_skill_resource(skill_name: str, path: str) -> str | ToolResult:
        try:
            text = registry.read_file(skill_name, path, config.max_resource_bytes)
        except KeyError:
            available = ", ".join(registry.names())
            return ToolResult(
                message=f"Unknown skill '{skill_name}'. Available skills: {available}",
                is_error=True,
            )
        except SkillResourceError as exc:
            files, truncated = registry.list_files(skill_name, limit=_MAX_FILES_IN_ERROR)
            listing = ", ".join(files) if files else "none"
            if truncated:
                listing += ", ..."
            return ToolResult(message=f"{exc}. Bundled files: {listing}", is_error=True)

        logger.info("Skill resource read: '%s' from skill '%s'", path, skill_name)
        return (
            f'<skill_resource skill="{escape(skill_name)}" path="{escape(path)}">\n'
            f"{text}\n"
            "</skill_resource>"
        )

    read_skill_resource.__doc__ = (
        "Read a file bundled with a skill: reference docs, scripts (such as .py or .sh "
        "files), templates or data. The content is returned as text; nothing is executed. "
        "Use a path relative to the skill directory, as listed in the skill's "
        '<skill_resources> or mentioned in its instructions, e.g. "references/guide.md" '
        'or "scripts/extract.py". A directory path lists the files inside it.\n\n'
        "Args:\n"
        "    skill_name: Name of the skill that bundles the file.\n"
        "    path: File path relative to the skill directory."
    )
    read_skill_resource.__annotations__ = {
        "skill_name": _skill_name_type(registry),
        "path": str,
        "return": str | ToolResult,
    }
    setattr(read_skill_resource, SKILL_TOOL_ATTR, True)
    return read_skill_resource


def has_bundled_files(registry: SkillsRegistry) -> bool:
    """True when any registered skill bundles at least one file."""
    return any(registry.list_files(name, limit=1)[0] for name in registry.names())
