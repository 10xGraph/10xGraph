"""Data models for the Agentflow Skills system.

Skills follow the Agent Skills specification (https://agentskills.io/specification):
a skill is a directory holding a ``SKILL.md`` file (YAML frontmatter + markdown body)
plus any bundled files such as ``scripts/``, ``references/`` and ``assets/``.

Defines :class:`SkillMeta` (parsed from SKILL.md frontmatter),
:class:`SkillDiagnostic` (a spec violation or recommendation found while loading)
and :class:`SkillConfig` (user-facing configuration for enabling skills on an Agent).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Characters that make a name unusable as a lookup key or tool enum value.
# Spec-level naming rules (lowercase, hyphens, 64 chars, ...) are reported as
# diagnostics by ``validation.py`` instead, so skills written for other clients
# still load.
_UNSAFE_NAME_RE = re.compile(r"[\s/\\\x00-\x1f\x7f]")

_DEFAULT_MAX_RESOURCE_BYTES = 256 * 1024


class SkillDiagnostic(BaseModel):
    """A problem found while loading or validating a skill.

    ``error`` marks a violation of the Agent Skills specification; ``warning``
    marks a recommendation from the specification that is not followed.
    """

    level: Literal["error", "warning"]
    message: str
    path: str = ""

    def __str__(self) -> str:
        location = f"{self.path}: " if self.path else ""
        return f"{self.level}: {location}{self.message}"


class SkillMeta(BaseModel):
    """Metadata about a single skill, parsed from SKILL.md frontmatter.

    ``name``, ``description``, ``license``, ``compatibility``, ``allowed_tools``
    and ``metadata`` map to the specification's frontmatter fields.
    ``triggers``, ``tags`` and ``priority`` are Agentflow extensions read from
    the ``metadata`` block.
    """

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    triggers: list[str] = Field(default_factory=list)
    tags: set[str] = Field(default_factory=set)
    priority: int = 0

    skill_dir: str = ""
    skill_file: str = ""

    model_config = ConfigDict(frozen=False)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Skill name must not be empty")
        if _UNSAFE_NAME_RE.search(v):
            raise ValueError(
                f"Invalid skill name '{v}'. Names must not contain whitespace, "
                "path separators or control characters."
            )
        return v

    @field_validator("description")
    @classmethod
    def _validate_description(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Skill description must not be empty")
        return v

    @field_validator("triggers")
    @classmethod
    def _validate_triggers(cls, v: list[str]) -> list[str]:
        return [cleaned for trigger in v if (cleaned := trigger.strip())]

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, v: set[str]) -> set[str]:
        return {cleaned.lower() for tag in v if (cleaned := tag.strip())}


class SkillConfig(BaseModel):
    """Configuration for the skills system on an Agent."""

    skills_dir: str | list[str] | None = None
    """Directory (or ordered list of directories) to discover skills from.

    Each entry may be a folder of skill directories (``skills/<name>/SKILL.md``)
    or a single skill directory. When two skills share a name, the one from the
    earlier directory wins and a warning is logged, so list project-specific
    directories before shared ones. ``.agents/skills/`` is the cross-client
    convention for project skills.
    """

    inject_catalog: bool = True
    """Add the ``<available_skills>`` catalog to the system prompt.

    When ``False`` the catalog is embedded in the ``activate_skill`` tool
    description instead.
    """

    hot_reload: bool = True
    """Re-read a SKILL.md from disk when its modification time changes."""

    max_resource_bytes: int = Field(default=_DEFAULT_MAX_RESOURCE_BYTES, gt=0)
    """Largest number of bytes ``read_skill_resource`` returns for one file."""

    include_skill_path: bool = False
    """Show the absolute skill directory to the model on activation.

    Enable this when the agent has its own shell or file tools and should run
    bundled scripts directly. Off by default so server paths stay private.
    """

    mode: Literal["on-demand", "session"] = "on-demand"
    """Activation mode.

    * ``"on-demand"`` *(default)* — the skill catalog is added to the system
      prompt and the LLM calls ``activate_skill()`` to load a skill's
      instructions when the task matches its description.
    * ``"session"`` — the framework preloads a single skill (identified by
      ``preload_from``) before every LLM call. No catalog and no
      ``activate_skill`` tool are added. Designed for multi-tenant agents
      where each session has a fixed persona/domain.

    In both modes ``read_skill_resource`` is registered when a skill bundles
    files, so the model can read them on demand.
    """

    preload_from: str | None = None
    """Name of the ``AgentState`` field that contains the skill name to load.

    Only used when ``mode="session"``.  The framework reads
    ``state.<preload_from>`` at the start of every call to resolve which
    SKILL.md to inject as a system message.

    Example::

        class FashionState(AgentState):
            SKILL_NAME: str = ""

        SkillConfig(
            skills_dir="./skills/",
            mode="session",
            preload_from="SKILL_NAME",
        )
    """

    @field_validator("skills_dir", mode="before")
    @classmethod
    def _validate_skills_dir(cls, v: Any) -> str | list[str] | None:
        if v is None:
            return v
        if isinstance(v, str | Path):
            return cls._clean_dir(v)
        if isinstance(v, list | tuple):
            if not v:
                raise ValueError("skills_dir must not be an empty list (use None to disable)")
            return [cls._clean_dir(item) for item in v]
        raise ValueError("skills_dir must be a path, a list of paths, or None")

    @staticmethod
    def _clean_dir(value: Any) -> str:
        if not isinstance(value, str | Path):
            raise ValueError(f"skills_dir entries must be paths, got {type(value).__name__}")
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("skills_dir must not be an empty string (use None to disable)")
        return cleaned

    @field_validator("preload_from")
    @classmethod
    def _validate_preload_from(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("preload_from must not be an empty string (use None to disable)")
        # Must be a valid Python identifier (state field name)
        if not v.isidentifier():
            raise ValueError(
                f"preload_from '{v}' is not a valid Python identifier. "
                "It must match the name of a field on your AgentState subclass."
            )
        return v

    @model_validator(mode="after")
    def _validate_session_mode_fields(self) -> SkillConfig:
        """Ensure session mode is fully configured."""
        if self.mode == "session" and self.preload_from is None:
            raise ValueError(
                "SkillConfig: 'preload_from' must be set when mode='session'. "
                "Provide the name of the AgentState field that holds the skill name "
                "(e.g. preload_from='SKILL_NAME')."
            )
        return self

    @property
    def skill_dirs(self) -> list[str]:
        """``skills_dir`` normalised to a list (empty when unset)."""
        if self.skills_dir is None:
            return []
        if isinstance(self.skills_dir, str):
            return [self.skills_dir]
        return list(self.skills_dir)
