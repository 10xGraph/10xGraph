"""Agent Skills specification checks.

Implements the frontmatter and naming rules from
https://agentskills.io/specification. The same rules as the reference
``skills-ref validate`` tool, plus a few of the spec's authoring
recommendations as warnings.

The loader runs these checks leniently: a skill that breaks a naming or length
rule still loads and the problem is logged. :func:`validate_skill` runs them
strictly for CLI and CI use.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from .models import SkillDiagnostic


SKILL_FILENAME = "SKILL.md"

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_COMPATIBILITY_LENGTH = 500
RECOMMENDED_MAX_BODY_LINES = 500

SPEC_FIELDS = frozenset(
    {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
)

# Agentflow extensions that belong inside the ``metadata`` block.
EXTENSION_METADATA_KEYS = frozenset({"triggers", "tags", "priority"})

# Relative references to bundled files inside a SKILL.md body.
_BUNDLED_REF_RE = re.compile(r"(?<![\w./-])((?:scripts|references|assets)/[\w./-]*[\w-])")


def _error(message: str) -> SkillDiagnostic:
    return SkillDiagnostic(level="error", message=message)


def _warning(message: str) -> SkillDiagnostic:
    return SkillDiagnostic(level="warning", message=message)


def validate_name(name: Any, skill_dir: Path | None = None) -> list[SkillDiagnostic]:
    """Check the ``name`` field against the specification."""
    if not isinstance(name, str) or not name.strip():
        return [_error("Field 'name' must be a non-empty string")]

    name = unicodedata.normalize("NFKC", name.strip())
    issues: list[SkillDiagnostic] = []

    if len(name) > MAX_NAME_LENGTH:
        issues.append(
            _error(f"Skill name '{name}' exceeds {MAX_NAME_LENGTH} characters ({len(name)} chars)")
        )
    if name != name.lower():
        issues.append(_error(f"Skill name '{name}' must be lowercase"))
    if name.startswith("-") or name.endswith("-"):
        issues.append(_error(f"Skill name '{name}' must not start or end with a hyphen"))
    if "--" in name:
        issues.append(_error(f"Skill name '{name}' must not contain consecutive hyphens"))
    if not all(c.isalnum() or c == "-" for c in name):
        issues.append(
            _error(
                f"Skill name '{name}' contains invalid characters. "
                "Only lowercase letters, digits and hyphens are allowed."
            )
        )
    if skill_dir is not None:
        dir_name = unicodedata.normalize("NFKC", skill_dir.name)
        if dir_name != name:
            issues.append(
                _error(f"Directory name '{skill_dir.name}' must match skill name '{name}'")
            )
    return issues


def validate_frontmatter(
    frontmatter: dict[str, Any],
    skill_dir: Path | None = None,
) -> list[SkillDiagnostic]:
    """Check parsed frontmatter against the specification.

    Args:
        frontmatter: The parsed YAML mapping.
        skill_dir: The skill directory, used for the name/directory match check.

    Returns:
        Diagnostics without a ``path``; callers fill it in.
    """
    issues = _check_unknown_fields(frontmatter)

    if "name" not in frontmatter:
        issues.append(_error("Missing required field: name"))
    else:
        issues.extend(validate_name(frontmatter["name"], skill_dir))

    issues.extend(_check_description(frontmatter))
    issues.extend(_check_optional_fields(frontmatter))
    issues.extend(_check_metadata(frontmatter))
    return issues


def _check_unknown_fields(frontmatter: dict[str, Any]) -> list[SkillDiagnostic]:
    extra = sorted(str(key) for key in frontmatter if key not in SPEC_FIELDS)
    if not extra:
        return []
    moved = [key for key in extra if key in EXTENSION_METADATA_KEYS]
    hint = f" Move {', '.join(moved)} under 'metadata'." if moved else ""
    return [
        _error(
            f"Unexpected frontmatter fields: {', '.join(extra)}. "
            f"Only {', '.join(sorted(SPEC_FIELDS))} are allowed.{hint}"
        )
    ]


def _check_description(frontmatter: dict[str, Any]) -> list[SkillDiagnostic]:
    if "description" not in frontmatter:
        return [_error("Missing required field: description")]
    description = frontmatter["description"]
    if not isinstance(description, str) or not description.strip():
        return [_error("Field 'description' must be a non-empty string")]
    length = len(description.strip())
    if length > MAX_DESCRIPTION_LENGTH:
        return [_error(f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters ({length} chars)")]
    return []


def _check_optional_fields(frontmatter: dict[str, Any]) -> list[SkillDiagnostic]:
    issues: list[SkillDiagnostic] = []
    if "license" in frontmatter and not isinstance(frontmatter["license"], str):
        issues.append(_error("Field 'license' must be a string"))

    if "compatibility" in frontmatter:
        compatibility = frontmatter["compatibility"]
        if not isinstance(compatibility, str) or not compatibility.strip():
            issues.append(_error("Field 'compatibility' must be a non-empty string"))
        elif len(compatibility) > MAX_COMPATIBILITY_LENGTH:
            issues.append(
                _error(
                    f"Compatibility exceeds {MAX_COMPATIBILITY_LENGTH} characters "
                    f"({len(compatibility)} chars)"
                )
            )

    if "allowed-tools" in frontmatter and not isinstance(frontmatter["allowed-tools"], str):
        issues.append(_error("Field 'allowed-tools' must be a space-separated string"))
    return issues


def _check_metadata(frontmatter: dict[str, Any]) -> list[SkillDiagnostic]:
    if "metadata" not in frontmatter:
        return []
    metadata = frontmatter["metadata"]
    if not isinstance(metadata, dict):
        return [_error("Field 'metadata' must be a mapping of string keys to strings")]
    non_string = sorted(str(k) for k, v in metadata.items() if not isinstance(v, str))
    if not non_string:
        return []
    return [
        _error(
            f"Metadata values must be strings: {', '.join(non_string)}. "
            'Quote them, e.g. priority: "10" or triggers: "a; b".'
        )
    ]


def validate_body(body: str, skill_dir: Path | None = None) -> list[SkillDiagnostic]:
    """Check the SKILL.md body against the specification's recommendations."""
    issues: list[SkillDiagnostic] = []

    line_count = len(body.splitlines())
    if line_count > RECOMMENDED_MAX_BODY_LINES:
        issues.append(
            _warning(
                f"SKILL.md body has {line_count} lines; keep it under "
                f"{RECOMMENDED_MAX_BODY_LINES} and move details into references/"
            )
        )

    if skill_dir is not None:
        missing = sorted(
            {ref for ref in _BUNDLED_REF_RE.findall(body) if not (skill_dir / ref).exists()}
        )
        if missing:
            issues.append(
                _warning(
                    "SKILL.md references files that do not exist in the skill directory: "
                    + ", ".join(missing)
                )
            )

    return issues


def validate_skill(skill_dir: str | Path) -> list[SkillDiagnostic]:
    """Strictly validate one skill directory against the specification.

    Args:
        skill_dir: Path to a directory containing ``SKILL.md``.

    Returns:
        All diagnostics found. The skill conforms when no ``error`` is present.
    """
    from .loader import parse_skill_file  # loader imports this module

    path = Path(skill_dir)
    skill_file = path / SKILL_FILENAME
    if not path.is_dir():
        return [SkillDiagnostic(level="error", message="Not a directory", path=str(path))]
    if not skill_file.is_file():
        return [
            SkillDiagnostic(
                level="error", message=f"Missing required file: {SKILL_FILENAME}", path=str(path)
            )
        ]

    parsed = parse_skill_file(skill_file)
    issues = list(parsed.diagnostics)
    if parsed.frontmatter is not None:
        issues.extend(validate_frontmatter(parsed.frontmatter, path))
        issues.extend(validate_body(parsed.body, path))

    for issue in issues:
        if not issue.path:
            issue.path = str(skill_file)
    return issues
