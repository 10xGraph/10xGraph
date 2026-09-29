"""Filesystem-based skill loader for Agentflow.

Discovers skill directories, parses ``SKILL.md`` files and reads bundled files,
following the Agent Skills specification (https://agentskills.io/specification).

Directory structure::

    skills/
    +-- pdf-processing/
    |   +-- SKILL.md          # YAML frontmatter + markdown body
    |   +-- scripts/          # executable code, e.g. extract.py
    |   +-- references/       # documentation loaded on demand
    |   +-- assets/           # templates, data files
    +-- code-review/
        +-- SKILL.md

Loading is lenient, as the specification's client guide recommends: a skill
that breaks a naming or length rule is loaded and the problem is recorded as a
diagnostic. A skill is skipped only when it has no parseable frontmatter, no
description, or a name that cannot be used as an identifier.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .models import SkillDiagnostic, SkillMeta
from .validation import SKILL_FILENAME, validate_frontmatter


logger = logging.getLogger("agentflow.skills.loader")

# Directory names never scanned for skills or listed as bundled files.
_IGNORED_DIRS = frozenset({"node_modules", "__pycache__"})
_IGNORED_SUFFIXES = (".pyc", ".pyo")

DEFAULT_MAX_LISTED_FILES = 200

# Bytes inspected when deciding whether a file is binary.
_BINARY_SNIFF_BYTES = 8192

# A top-level ``key: value`` line whose plain value may need quoting.
_PLAIN_SCALAR_LINE_RE = re.compile(r"^([A-Za-z0-9_-]+):[ \t]+(.+?)[ \t]*$")
_YAML_VALUE_INDICATORS = ("'", '"', "|", ">", "[", "{", "&", "*", "!", "#")


def _one_line(error: Exception) -> str:
    return " ".join(str(error).split())


class SkillResourceError(ValueError):
    """Raised when a bundled skill file cannot be resolved or read."""


@dataclass
class ParsedSkillFile:
    """Result of parsing one ``SKILL.md``.

    ``frontmatter`` is ``None`` when the file has no parseable frontmatter.
    """

    frontmatter: dict[str, Any] | None
    body: str
    diagnostics: list[SkillDiagnostic] = field(default_factory=list)


# -- parsing ----------------------------------------------------------------


def split_frontmatter(content: str) -> tuple[str | None, str]:
    """Split markdown into ``(raw_yaml, body)``.

    ``raw_yaml`` is ``None`` when the file does not start with a ``---`` line or
    the frontmatter is never closed; the body is then the whole file.
    """
    text = content.removeprefix("﻿")
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, text.strip()

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            raw_yaml = "".join(lines[1:index])
            body = "".join(lines[index + 1 :]).strip()
            return raw_yaml, body
    return None, text.strip()


def _quote_plain_values(raw_yaml: str) -> str:
    """Quote top-level plain values that contain ``': '``.

    Skills written for other clients often contain values such as
    ``description: Use this when: ...`` that strict YAML rejects.
    """
    fixed: list[str] = []
    for line in raw_yaml.splitlines():
        match = _PLAIN_SCALAR_LINE_RE.match(line)
        if match and ": " in match[2] and not match[2].startswith(_YAML_VALUE_INDICATORS):
            fixed.append(f"{match[1]}: {json.dumps(match[2])}")
        else:
            fixed.append(line)
    return "\n".join(fixed)


def parse_frontmatter_yaml(raw_yaml: str) -> tuple[dict[str, Any] | None, list[SkillDiagnostic]]:
    """Parse frontmatter YAML, retrying once with plain values quoted."""
    try:
        parsed = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as first_error:
        try:
            parsed = yaml.safe_load(_quote_plain_values(raw_yaml))
        except yaml.YAMLError:
            message = f"Invalid YAML in frontmatter: {_one_line(first_error)}"
            return None, [SkillDiagnostic(level="error", message=message)]
        diagnostics = [
            SkillDiagnostic(
                level="error",
                message=(
                    "Invalid YAML in frontmatter (a value contains ': '); loaded after "
                    "quoting it. Quote the value to fix this. "
                    f"Parser said: {_one_line(first_error)}"
                ),
            )
        ]
    else:
        diagnostics = []

    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        return None, [
            SkillDiagnostic(
                level="error",
                message=f"Frontmatter must be a mapping, got {type(parsed).__name__}",
            )
        ]
    return parsed, diagnostics


def parse_skill_file(skill_file: str | Path) -> ParsedSkillFile:
    """Read and parse a ``SKILL.md`` file."""
    path = Path(skill_file)
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return ParsedSkillFile(
            None, "", [SkillDiagnostic(level="error", message=f"Cannot read file: {exc}")]
        )

    raw_yaml, body = split_frontmatter(content)
    if raw_yaml is None:
        return ParsedSkillFile(
            None,
            body,
            [
                SkillDiagnostic(
                    level="error",
                    message="SKILL.md must start with YAML frontmatter delimited by '---' lines",
                )
            ],
        )

    frontmatter, diagnostics = parse_frontmatter_yaml(raw_yaml)
    return ParsedSkillFile(frontmatter, body, diagnostics)


# -- discovery --------------------------------------------------------------


def iter_skill_dirs(root: str | Path) -> list[Path]:
    """Return the skill directories under *root*.

    *root* is either a single skill directory (it contains ``SKILL.md``) or a
    folder whose immediate subdirectories are skills. Hidden directories,
    ``node_modules`` and ``__pycache__`` are skipped.
    """
    root_path = Path(root)
    if (root_path / SKILL_FILENAME).is_file():
        return [root_path]
    if not root_path.is_dir():
        return []

    return [
        entry
        for entry in sorted(root_path.iterdir(), key=lambda p: p.name)
        if entry.is_dir()
        and not entry.name.startswith(".")
        and entry.name not in _IGNORED_DIRS
        and (entry / SKILL_FILENAME).is_file()
    ]


def load_skill(skill_dir: str | Path) -> tuple[SkillMeta | None, list[SkillDiagnostic]]:
    """Parse one skill directory into a :class:`SkillMeta`.

    Returns:
        ``(meta, diagnostics)``. ``meta`` is ``None`` when the skill cannot be
        loaded; the diagnostics then explain why.
    """
    path = Path(skill_dir)
    skill_file = path / SKILL_FILENAME
    parsed = parse_skill_file(skill_file)
    diagnostics = list(parsed.diagnostics)

    def _finish(meta: SkillMeta | None) -> tuple[SkillMeta | None, list[SkillDiagnostic]]:
        for diagnostic in diagnostics:
            if not diagnostic.path:
                diagnostic.path = str(skill_file)
        return meta, diagnostics

    if parsed.frontmatter is None:
        return _finish(None)

    frontmatter = parsed.frontmatter
    diagnostics.extend(validate_frontmatter(frontmatter, path))

    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not isinstance(name, str) or not name.strip():
        diagnostics.append(
            SkillDiagnostic(level="error", message="Skipped: 'name' is missing or empty")
        )
        return _finish(None)
    if not isinstance(description, str) or not description.strip():
        diagnostics.append(
            SkillDiagnostic(level="error", message="Skipped: 'description' is missing or empty")
        )
        return _finish(None)

    raw_metadata = frontmatter.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

    try:
        meta = SkillMeta(
            name=name,
            description=description,
            license=_optional_str(frontmatter.get("license")),
            compatibility=_optional_str(frontmatter.get("compatibility")),
            allowed_tools=_parse_allowed_tools(frontmatter.get("allowed-tools")),
            metadata={str(key): _metadata_str(value) for key, value in metadata.items()},
            triggers=_parse_triggers(metadata.get("triggers")),
            tags=_parse_tags(metadata.get("tags")),
            priority=_parse_priority(metadata.get("priority"), diagnostics),
            skill_dir=str(path.resolve()),
            skill_file=str(skill_file.resolve()),
        )
    except ValueError as exc:
        diagnostics.append(SkillDiagnostic(level="error", message=f"Skipped: {exc}"))
        return _finish(None)

    return _finish(meta)


def discover_skills(
    skills_dirs: str | Path | list[str] | list[Path],
) -> tuple[list[SkillMeta], list[SkillDiagnostic]]:
    """Load every skill found under *skills_dirs*.

    Duplicate names are not resolved here; see :meth:`SkillsRegistry.discover`.

    Returns:
        ``(skills, diagnostics)`` in discovery order.
    """
    roots = [skills_dirs] if isinstance(skills_dirs, str | Path) else list(skills_dirs)
    skills: list[SkillMeta] = []
    diagnostics: list[SkillDiagnostic] = []

    for root in roots:
        if not Path(root).is_dir():
            diagnostics.append(
                SkillDiagnostic(
                    level="warning", message="Skills directory not found", path=str(root)
                )
            )
            continue
        for skill_dir in iter_skill_dirs(root):
            meta, skill_diagnostics = load_skill(skill_dir)
            diagnostics.extend(skill_diagnostics)
            if meta is not None:
                skills.append(meta)

    return skills, diagnostics


# -- content ----------------------------------------------------------------


def load_skill_content(meta: SkillMeta) -> str:
    """Read and return the body of a SKILL.md (frontmatter stripped)."""
    try:
        content = Path(meta.skill_file).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Cannot read %s: %s", meta.skill_file, exc)
        return ""
    _raw_yaml, body = split_frontmatter(content)
    return body


def _is_listed(rel_parts: tuple[str, ...]) -> bool:
    name = rel_parts[-1]
    return not (
        any(part.startswith(".") or part in _IGNORED_DIRS for part in rel_parts)
        or name.endswith(_IGNORED_SUFFIXES)
    )


def list_skill_files(
    meta: SkillMeta,
    limit: int = DEFAULT_MAX_LISTED_FILES,
    subdir: str = "",
) -> tuple[list[str], bool]:
    """List files bundled with a skill, relative to the skill directory.

    ``SKILL.md`` itself, hidden files, ``__pycache__`` and ``node_modules`` are
    excluded. Symlinked directories are not followed.

    Returns:
        ``(paths, truncated)`` where *paths* are sorted POSIX-style relative
        paths and *truncated* is ``True`` when more than *limit* files exist.
    """
    root = Path(meta.skill_dir)
    start = root / subdir if subdir else root
    found: list[str] = []

    for current, dirnames, filenames in os.walk(start):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d not in _IGNORED_DIRS
        )
        for filename in sorted(filenames):
            rel = (Path(current) / filename).relative_to(root)
            if rel.as_posix() == SKILL_FILENAME or not _is_listed(rel.parts):
                continue
            found.append(rel.as_posix())

    found.sort()
    return found[:limit], len(found) > limit


def resolve_skill_path(meta: SkillMeta, rel_path: str) -> tuple[Path, str]:
    """Resolve *rel_path* inside the skill directory.

    Returns:
        ``(absolute_path, normalised_relative_path)``.

    Raises:
        SkillResourceError: If the path is empty, absolute, or escapes the skill
            directory (including through symlinks).
    """
    cleaned = rel_path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if not cleaned or cleaned == ".":
        raise SkillResourceError("Path must not be empty")

    posix = PurePosixPath(cleaned)
    if posix.is_absolute() or re.match(r"^[A-Za-z]:", cleaned):
        raise SkillResourceError(f"Path '{rel_path}' must be relative to the skill directory")

    root = Path(meta.skill_dir).resolve()
    target = (root / posix).resolve()
    try:
        normalised = target.relative_to(root).as_posix()
    except ValueError:
        raise SkillResourceError(f"Path '{rel_path}' is outside the skill directory") from None
    return target, normalised


def read_skill_file(
    meta: SkillMeta,
    rel_path: str,
    max_bytes: int,
    list_limit: int = DEFAULT_MAX_LISTED_FILES,
) -> str:
    """Read a file bundled with a skill as text.

    Any file type can be read: markdown, Python and shell scripts, JSON, CSV,
    extension-less executables and so on. Text that is not valid UTF-8 is
    decoded with replacement characters. Binary files (containing NUL bytes)
    are described instead of dumped. Content beyond *max_bytes* is truncated
    with a note. A directory path returns a listing of its files.

    Raises:
        SkillResourceError: If the path is invalid, escapes the skill directory,
            does not exist, or cannot be read.
    """
    target, normalised = resolve_skill_path(meta, rel_path)

    if target.is_dir():
        subdir = "" if normalised == "." else normalised
        files, truncated = list_skill_files(meta, limit=list_limit, subdir=subdir)
        if not files:
            return f"Directory '{normalised}' contains no readable files."
        listing = "\n".join(files)
        more = f"\n(listing truncated at {list_limit} files)" if truncated else ""
        return f"Directory '{normalised}' contains:\n{listing}{more}"

    if not target.is_file():
        raise SkillResourceError(f"File '{normalised}' not found in skill '{meta.name}'")

    try:
        size = target.stat().st_size
        with target.open("rb") as handle:
            data = handle.read(max_bytes)
    except OSError as exc:
        raise SkillResourceError(f"Could not read '{normalised}': {exc}") from exc

    if b"\x00" in data[:_BINARY_SNIFF_BYTES]:
        return (
            f"'{normalised}' is a binary file ({size} bytes); its content cannot be shown as text."
        )

    text = data.decode("utf-8", errors="replace")
    if size > max_bytes:
        text += f"\n\n[truncated: showing the first {max_bytes} of {size} bytes]"
    return text


# -- field parsing ------------------------------------------------------------


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _metadata_str(value: Any) -> str:
    """Coerce a metadata value to a string, as the specification requires."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def _parse_allowed_tools(value: Any) -> list[str]:
    if isinstance(value, str):
        return value.split()
    if isinstance(value, list):
        return [text for item in value if (text := str(item).strip())]
    return []


def _parse_triggers(value: Any) -> list[str]:
    """Triggers are a ``;``- or newline-separated string (or a list)."""
    if isinstance(value, list):
        items = [str(item) for item in value]
    elif isinstance(value, str):
        items = re.split(r"[;\n]", value)
    else:
        return []
    return [text for item in items if (text := item.strip())]


def _parse_tags(value: Any) -> set[str]:
    """Tags are a comma- or whitespace-separated string (or a list)."""
    if isinstance(value, list):
        items = [str(item) for item in value]
    elif isinstance(value, str):
        items = re.split(r"[,\s]+", value)
    else:
        return set()
    return {text for item in items if (text := item.strip())}


def _parse_priority(value: Any, diagnostics: list[SkillDiagnostic]) -> int:
    if value is None:
        return 0
    try:
        return int(str(value).strip())
    except ValueError:
        diagnostics.append(
            SkillDiagnostic(
                level="warning",
                message=f"metadata.priority must be an integer, got {value!r}; using 0",
            )
        )
        return 0
