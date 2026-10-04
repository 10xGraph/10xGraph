"""Agentflow Skills — Agent Skills (https://agentskills.io) support for agents.

A skill is a directory with a ``SKILL.md`` file (YAML frontmatter + markdown
instructions) and optional bundled files such as ``scripts/``, ``references/``
and ``assets/``. Skills load progressively: the model sees each skill's name and
description, loads the full instructions when a task matches, and reads bundled
files only when the instructions point to them.

Two activation modes are supported:

**on-demand** (default) — the skill catalog is added to the system prompt and the
LLM calls ``activate_skill()`` when a user request matches a skill, then
``read_skill_resource()`` for any bundled file it needs::

    from agentflow.core.skills import SkillConfig

    agent = Agent(
        model="gpt-4o",
        system_prompt=[{"role": "system", "content": "You are helpful."}],
        tool_node=ToolNode([]),
        skills=SkillConfig(skills_dir="./.agents/skills/"),
    )

**session** — designed for multi-tenant agents where each session has a fixed
domain/persona.  The framework reads a state field to identify which skill to
preload, with no catalog and no extra tool-call round-trip::

    from agentflow.core.skills import SkillConfig
    from agentflow.core.state import AgentState


    class TenantState(AgentState):
        SKILL_NAME: str = ""


    agent = Agent(
        model="gpt-4o",
        skills=SkillConfig(
            skills_dir="./skills/",
            mode="session",
            preload_from="SKILL_NAME",  # reads state.SKILL_NAME each call
        ),
    )

Use :func:`validate_skill` (or ``agentflow skills --validate``) to check a skill
against the specification.
"""

from .loader import SkillResourceError
from .models import SkillConfig, SkillDiagnostic, SkillMeta
from .registry import SkillsRegistry
from .validation import validate_skill


__all__ = [
    "SkillConfig",
    "SkillDiagnostic",
    "SkillMeta",
    "SkillResourceError",
    "SkillsRegistry",
    "validate_skill",
]
