"""Central skills registry for Agentflow.

The registry is the single index of all discovered skills.  It can be
registered in InjectQ so that any graph node can access it.
"""

from __future__ import annotations

import contextlib
import logging
from html import escape
from pathlib import Path

from .loader import (
    DEFAULT_MAX_LISTED_FILES,
    discover_skills,
    list_skill_files,
    load_skill_content,
    read_skill_file,
)
from .models import SkillDiagnostic, SkillMeta


logger = logging.getLogger("tenxgraph.skills.registry")


def _log_diagnostic(diagnostic: SkillDiagnostic) -> None:
    logger.warning("Skill %s", diagnostic)


class SkillsRegistry:
    """Central registry that holds discovered :class:`SkillMeta` entries.

    Typical lifecycle::

        registry = SkillsRegistry()
        registry.discover(["./.agents/skills", "./shared-skills"])
        catalog = registry.build_catalog()
        body = registry.load_content("pdf-processing")
        script = registry.read_file("pdf-processing", "scripts/extract.py")
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillMeta] = {}
        # name -> (SKILL.md mtime, body) for hot-reload aware caching
        self._bodies: dict[str, tuple[float, str]] = {}
        self._diagnostics: list[SkillDiagnostic] = []

    # -- registration -------------------------------------------------------

    def register(self, meta: SkillMeta, *, replace: bool = False) -> None:
        """Register a single :class:`SkillMeta`.

        Re-registering the same skill file is a no-op. Registering a different
        skill under an existing name raises ``ValueError`` unless *replace* is
        ``True``.
        """
        existing = self._skills.get(meta.name)
        if existing is not None and not replace:
            if existing.skill_file == meta.skill_file:
                logger.debug("Skill '%s' already registered from %s", meta.name, meta.skill_file)
                return
            raise ValueError(
                f"Duplicate skill name '{meta.name}' from {meta.skill_file} "
                f"(already registered from {existing.skill_file})"
            )

        self._skills[meta.name] = meta
        self._bodies.pop(meta.name, None)
        logger.info("Registered skill: '%s'", meta.name)

    def discover(self, skills_dirs: str | Path | list[str] | list[Path]) -> list[SkillMeta]:
        """Discover skills from one or more directories and register them.

        Directories are scanned in order. When a name is already registered
        from another file, the earlier skill wins and the later one is reported
        as shadowed, so list project-specific directories first.

        Returns:
            The skills registered by this call.
        """
        found, diagnostics = discover_skills(skills_dirs)
        for diagnostic in diagnostics:
            _log_diagnostic(diagnostic)
        self._diagnostics.extend(diagnostics)

        registered: list[SkillMeta] = []
        for meta in found:
            existing = self._skills.get(meta.name)
            if existing is not None and existing.skill_file != meta.skill_file:
                shadowed = SkillDiagnostic(
                    level="warning",
                    message=(
                        f"Skill '{meta.name}' is shadowed by {existing.skill_file} and was "
                        "not loaded"
                    ),
                    path=meta.skill_file,
                )
                _log_diagnostic(shadowed)
                self._diagnostics.append(shadowed)
                continue
            self.register(meta)
            registered.append(meta)
        return registered

    @property
    def diagnostics(self) -> list[SkillDiagnostic]:
        """Problems found while discovering skills, in discovery order."""
        return list(self._diagnostics)

    # -- lookup -------------------------------------------------------------

    def get(self, name: str) -> SkillMeta | None:
        return self._skills.get(name)

    def get_all(self, tags: set[str] | None = None) -> list[SkillMeta]:
        skills = list(self._skills.values())
        if tags:
            skills = [s for s in skills if s.tags & tags]
        return skills

    def names(self) -> list[str]:
        return sorted(self._skills.keys())

    def unregister(self, name: str) -> bool:
        """Remove a skill by name. Returns True if it was present."""
        if name in self._skills:
            del self._skills[name]
            self._bodies.pop(name, None)
            logger.info("Unregistered skill: '%s'", name)
            return True
        return False

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    # -- content loading ----------------------------------------------------

    def load_content(self, name: str, hot_reload: bool = True) -> str:
        """Return the body of a skill's SKILL.md (frontmatter stripped).

        The body is cached after the first read. With *hot_reload* the cache
        is refreshed whenever the file's modification time changes.
        """
        meta = self._skills.get(name)
        if meta is None:
            return ""

        cached = self._bodies.get(name)
        if cached is not None and not hot_reload:
            return cached[1]

        mtime = 0.0
        with contextlib.suppress(OSError):
            mtime = Path(meta.skill_file).stat().st_mtime
        if cached is not None and cached[0] == mtime:
            return cached[1]

        body = load_skill_content(meta)
        if body:
            self._bodies[name] = (mtime, body)
        return body

    def list_files(
        self, name: str, limit: int = DEFAULT_MAX_LISTED_FILES
    ) -> tuple[list[str], bool]:
        """List files bundled with a skill (see :func:`loader.list_skill_files`)."""
        meta = self._skills.get(name)
        if meta is None:
            return [], False
        return list_skill_files(meta, limit=limit)

    def read_file(self, name: str, path: str, max_bytes: int) -> str:
        """Read a bundled file as text (see :func:`loader.read_skill_file`).

        Raises:
            KeyError: If the skill is not registered.
            SkillResourceError: If the file cannot be resolved or read.
        """
        meta = self._skills.get(name)
        if meta is None:
            raise KeyError(name)
        return read_skill_file(meta, path, max_bytes)

    # -- prompt helpers -----------------------------------------------------

    def build_catalog(self, tags: set[str] | None = None) -> str:
        """Build the ``<available_skills>`` catalog shown to the model.

        Each entry carries the skill's name and description, plus any
        Agentflow ``triggers`` as extra hints. Skills are ordered by priority
        (highest first), then name. Returns ``""`` when there are no skills.
        """
        skills = sorted(self.get_all(tags=tags), key=lambda s: (-s.priority, s.name))
        if not skills:
            return ""

        lines = ["<available_skills>"]
        for meta in skills:
            lines.append("  <skill>")
            lines.append(f"    <name>{escape(meta.name, quote=False)}</name>")
            description = " ".join(meta.description.split())
            lines.append(f"    <description>{escape(description, quote=False)}</description>")
            if meta.triggers:
                hints = "; ".join(" ".join(t.split()) for t in meta.triggers)
                lines.append(f"    <triggers>{escape(hints, quote=False)}</triggers>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)
