"""Tests for the Agentflow Skills system (Agent Skills specification support).

Covers:
- SkillMeta / SkillConfig models
- SKILL.md parsing, malformed-YAML fallback and lenient discovery (loader)
- Spec validation (validation)
- SkillsRegistry lookup, shadowing, hot reload, catalog (registry)
- activate_skill / read_skill_resource tools and prompts (activation)
- AgentSkillsMixin._setup_skills and _build_skill_prompts (agent integration)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from tenxgraph.core.graph.agent_internal.skills import AgentSkillsMixin
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.skills import (
    SkillConfig,
    SkillMeta,
    SkillResourceError,
    SkillsRegistry,
    validate_skill,
)
from tenxgraph.core.skills.activation import (
    ACTIVE_SKILLS_KEY,
    build_catalog_prompt,
    get_active_skills,
    make_activate_skill_tool,
    make_read_skill_resource_tool,
    mark_skill_active,
    skill_content_marker,
)
from tenxgraph.core.skills.loader import (
    discover_skills,
    iter_skill_dirs,
    load_skill,
    load_skill_content,
    parse_skill_file,
    read_skill_file,
    split_frontmatter,
)
from tenxgraph.core.state import AgentState, Message, ToolResult


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _write_skill(
    root: Path,
    name: str,
    *,
    frontmatter: str | None = None,
    description: str = "A test skill. Use when testing.",
    metadata: str = "",
    body: str = "# Skill body\nSome instructions.",
    files: dict[str, str | bytes] | None = None,
    dir_name: str | None = None,
) -> Path:
    """Create ``root/<dir_name or name>/SKILL.md`` plus bundled *files*."""
    skill_dir = root / (dir_name or name)
    skill_dir.mkdir(parents=True, exist_ok=True)
    if frontmatter is None:
        frontmatter = f"name: {name}\ndescription: {description}\n{metadata}"
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")
    for rel_path, content in (files or {}).items():
        target = skill_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    return skill_dir


def _registry(root: Path) -> SkillsRegistry:
    registry = SkillsRegistry()
    registry.discover(str(root))
    return registry


def _text(result: str | ToolResult) -> str:
    return result.message if isinstance(result, ToolResult) else result


def _mixin(tool_node: ToolNode | None = None) -> AgentSkillsMixin:
    mixin = AgentSkillsMixin()
    mixin._tool_node = tool_node
    return mixin


# ════════════════════════════════════════════════════════════════════════════
# 1. Models
# ════════════════════════════════════════════════════════════════════════════


class TestSkillMeta:
    def test_valid(self):
        meta = SkillMeta(name="pdf-processing", description="Handles PDFs")
        assert meta.name == "pdf-processing"
        assert meta.allowed_tools == []
        assert meta.metadata == {}

    def test_name_kept_as_authored(self):
        # Spec-level naming problems are diagnostics, not model errors.
        assert SkillMeta(name="My_Skill", description="d").name == "My_Skill"

    @pytest.mark.parametrize("name", ["", "   ", "has space", "a/b", "a\\b", "tab\there"])
    def test_unusable_names_rejected(self, name: str):
        with pytest.raises(ValidationError):
            SkillMeta(name=name, description="d")

    def test_description_required(self):
        with pytest.raises(ValidationError):
            SkillMeta(name="x", description="  ")

    def test_description_stripped(self):
        assert SkillMeta(name="x", description="  hi  ").description == "hi"

    def test_triggers_and_tags_cleaned(self):
        meta = SkillMeta(name="x", description="d", triggers=[" a ", ""], tags={" Dev ", ""})
        assert meta.triggers == ["a"]
        assert meta.tags == {"dev"}


class TestSkillConfig:
    def test_defaults(self):
        cfg = SkillConfig()
        assert cfg.skills_dir is None
        assert cfg.skill_dirs == []
        assert cfg.inject_catalog is True
        assert cfg.hot_reload is True
        assert cfg.include_skill_path is False
        assert cfg.mode == "on-demand"

    def test_single_dir(self):
        assert SkillConfig(skills_dir=" ./skills ").skill_dirs == ["./skills"]

    def test_path_object(self, tmp_path: Path):
        assert SkillConfig(skills_dir=tmp_path).skill_dirs == [str(tmp_path)]

    def test_list_of_dirs(self, tmp_path: Path):
        cfg = SkillConfig(skills_dir=["./a", tmp_path])
        assert cfg.skill_dirs == ["./a", str(tmp_path)]

    @pytest.mark.parametrize("bad", ["", "   ", [], ["ok", ""], 5])
    def test_invalid_dirs(self, bad):
        with pytest.raises(ValidationError):
            SkillConfig(skills_dir=bad)

    def test_max_resource_bytes_positive(self):
        with pytest.raises(ValidationError):
            SkillConfig(max_resource_bytes=0)


# ════════════════════════════════════════════════════════════════════════════
# 2. Parsing
# ════════════════════════════════════════════════════════════════════════════


class TestParsing:
    def test_split_frontmatter(self):
        raw, body = split_frontmatter("---\nname: x\n---\n\n# Body\n")
        assert raw == "name: x\n"
        assert body == "# Body"

    def test_split_handles_bom_crlf_and_trailing_spaces(self):
        raw, body = split_frontmatter("\ufeff---  \r\nname: x\r\n---\r\nBody\r\n")
        assert raw is not None and "name: x" in raw
        assert body == "Body"

    def test_closing_delimiter_must_be_its_own_line(self):
        # "----" or "---foo" do not close the frontmatter.
        raw, _ = split_frontmatter("---\nname: x\n----\nmore: 1\n---\nBody")
        assert raw is not None and "more: 1" in raw

    def test_no_frontmatter(self, tmp_path: Path):
        path = tmp_path / "SKILL.md"
        path.write_text("# Just markdown")
        parsed = parse_skill_file(path)
        assert parsed.frontmatter is None
        assert parsed.diagnostics[0].level == "error"

    def test_unclosed_frontmatter(self, tmp_path: Path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: x\n")
        assert parse_skill_file(path).frontmatter is None

    def test_colon_in_value_is_recovered(self, tmp_path: Path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: x\ndescription: Use this when: the user asks\n---\nBody")
        parsed = parse_skill_file(path)
        assert parsed.frontmatter == {"name": "x", "description": "Use this when: the user asks"}
        assert any("loaded after quoting" in d.message for d in parsed.diagnostics)

    def test_unrecoverable_yaml(self, tmp_path: Path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: [unclosed\n---\nBody")
        parsed = parse_skill_file(path)
        assert parsed.frontmatter is None
        assert "Invalid YAML" in parsed.diagnostics[0].message

    def test_non_mapping_frontmatter(self, tmp_path: Path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\n- a\n- b\n---\nBody")
        assert parse_skill_file(path).frontmatter is None

    def test_missing_file(self, tmp_path: Path):
        parsed = parse_skill_file(tmp_path / "nope" / "SKILL.md")
        assert parsed.frontmatter is None
        assert "Cannot read" in parsed.diagnostics[0].message

    def test_load_skill_content_strips_frontmatter(self, tmp_path: Path):
        skill_dir = _write_skill(tmp_path, "x", body="Hello body")
        meta, _ = load_skill(skill_dir)
        assert meta is not None
        assert load_skill_content(meta) == "Hello body"


# ════════════════════════════════════════════════════════════════════════════
# 3. Loading & discovery
# ════════════════════════════════════════════════════════════════════════════


class TestLoadSkill:
    def test_all_spec_fields(self, tmp_path: Path):
        skill_dir = _write_skill(
            tmp_path,
            "pdf-processing",
            frontmatter=(
                "name: pdf-processing\n"
                "description: Extract PDF text. Use when handling PDFs.\n"
                "license: Apache-2.0\n"
                "compatibility: Requires python3\n"
                "allowed-tools: Bash(git:*) Read\n"
                "metadata:\n"
                "  author: example-org\n"
                '  version: "1.0"\n'
            ),
        )
        meta, diagnostics = load_skill(skill_dir)
        assert diagnostics == []
        assert meta is not None
        assert meta.license == "Apache-2.0"
        assert meta.compatibility == "Requires python3"
        assert meta.allowed_tools == ["Bash(git:*)", "Read"]
        assert meta.metadata == {"author": "example-org", "version": "1.0"}
        assert meta.skill_dir == str(skill_dir.resolve())
        assert meta.skill_file == str((skill_dir / "SKILL.md").resolve())

    def test_agentflow_extensions_as_strings(self, tmp_path: Path):
        skill_dir = _write_skill(
            tmp_path,
            "review",
            metadata=(
                "metadata:\n"
                '  triggers: "review my code; find bugs"\n'
                '  tags: "engineering, dev"\n'
                '  priority: "10"\n'
            ),
        )
        meta, diagnostics = load_skill(skill_dir)
        assert diagnostics == []
        assert meta is not None
        assert meta.triggers == ["review my code", "find bugs"]
        assert meta.tags == {"engineering", "dev"}
        assert meta.priority == 10

    def test_agentflow_extensions_as_lists_load_with_diagnostic(self, tmp_path: Path):
        skill_dir = _write_skill(
            tmp_path,
            "review",
            metadata="metadata:\n  triggers:\n    - a\n    - b\n  priority: 3\n",
        )
        meta, diagnostics = load_skill(skill_dir)
        assert meta is not None
        assert meta.triggers == ["a", "b"]
        assert meta.priority == 3
        assert meta.metadata["triggers"] == "a; b"
        assert any("Metadata values must be strings" in d.message for d in diagnostics)

    def test_invalid_priority_defaults_to_zero(self, tmp_path: Path):
        skill_dir = _write_skill(tmp_path, "x", metadata='metadata:\n  priority: "high"\n')
        meta, diagnostics = load_skill(skill_dir)
        assert meta is not None and meta.priority == 0
        assert any("priority" in d.message for d in diagnostics)

    def test_top_level_extension_fields_are_flagged_not_used(self, tmp_path: Path):
        skill_dir = _write_skill(
            tmp_path, "x", frontmatter="name: x\ndescription: d\ntriggers:\n  - a\n"
        )
        meta, diagnostics = load_skill(skill_dir)
        assert meta is not None
        assert meta.triggers == []
        assert any("Move triggers under 'metadata'" in d.message for d in diagnostics)

    @pytest.mark.parametrize(
        ("name", "dir_name", "expected"),
        [
            ("my_tool", None, "invalid characters"),
            ("Upper", None, "must be lowercase"),
            ("a--b", None, "consecutive hyphens"),
            ("-a", None, "start or end with a hyphen"),
            ("x" * 70, None, "exceeds 64"),
            ("other", "dir-name", "must match skill name"),
        ],
    )
    def test_spec_name_violations_still_load(self, tmp_path, name, dir_name, expected):
        skill_dir = _write_skill(tmp_path, name, dir_name=dir_name)
        meta, diagnostics = load_skill(skill_dir)
        assert meta is not None
        assert meta.name == name
        assert any(expected in d.message for d in diagnostics)
        assert all(d.path.endswith("SKILL.md") for d in diagnostics)

    def test_long_description_still_loads(self, tmp_path: Path):
        skill_dir = _write_skill(tmp_path, "x", description="d" * 1100)
        meta, diagnostics = load_skill(skill_dir)
        assert meta is not None
        assert any("exceeds 1024" in d.message for d in diagnostics)

    @pytest.mark.parametrize(
        "frontmatter",
        ["description: no name", "name: x", "name: x\ndescription: ''", "name: a b\ndescription: d"],
    )
    def test_unloadable_skills_are_skipped(self, tmp_path: Path, frontmatter: str):
        skill_dir = _write_skill(tmp_path, "x", frontmatter=frontmatter)
        meta, diagnostics = load_skill(skill_dir)
        assert meta is None
        assert any(d.message.startswith("Skipped") for d in diagnostics)


class TestDiscovery:
    def test_folder_of_skills(self, tmp_path: Path):
        _write_skill(tmp_path, "beta")
        _write_skill(tmp_path, "alpha")
        (tmp_path / "README.md").write_text("not a skill")
        (tmp_path / "no-skill-md").mkdir()
        skills, _ = discover_skills(tmp_path)
        assert [s.name for s in skills] == ["alpha", "beta"]

    def test_single_skill_directory(self, tmp_path: Path):
        skill_dir = _write_skill(tmp_path, "solo")
        skills, _ = discover_skills(skill_dir)
        assert [s.name for s in skills] == ["solo"]

    def test_hidden_and_ignored_dirs_skipped(self, tmp_path: Path):
        _write_skill(tmp_path, ".hidden")
        _write_skill(tmp_path, "node_modules")
        _write_skill(tmp_path, "visible")
        assert [p.name for p in iter_skill_dirs(tmp_path)] == ["visible"]

    def test_missing_dir_reports_warning(self, tmp_path: Path):
        skills, diagnostics = discover_skills(tmp_path / "missing")
        assert skills == []
        assert diagnostics[0].level == "warning"
        assert "not found" in diagnostics[0].message

    def test_multiple_dirs(self, tmp_path: Path):
        _write_skill(tmp_path / "a", "one")
        _write_skill(tmp_path / "b", "two")
        skills, _ = discover_skills([tmp_path / "a", tmp_path / "b"])
        assert [s.name for s in skills] == ["one", "two"]


# ════════════════════════════════════════════════════════════════════════════
# 4. Validation
# ════════════════════════════════════════════════════════════════════════════


class TestValidateSkill:
    def test_conforming_skill(self, tmp_path: Path):
        skill_dir = _write_skill(
            tmp_path,
            "pdf",
            body="See [guide](references/guide.md) and run scripts/run.py.",
            files={"references/guide.md": "g", "scripts/run.py": "print(1)"},
        )
        assert validate_skill(skill_dir) == []

    def test_not_a_directory(self, tmp_path: Path):
        issues = validate_skill(tmp_path / "missing")
        assert issues[0].message == "Not a directory"

    def test_missing_skill_md(self, tmp_path: Path):
        issues = validate_skill(tmp_path)
        assert "Missing required file: SKILL.md" in issues[0].message

    def test_unknown_field_is_error(self, tmp_path: Path):
        skill_dir = _write_skill(tmp_path, "x", frontmatter="name: x\ndescription: d\nfoo: 1")
        issues = validate_skill(skill_dir)
        assert [i.level for i in issues] == ["error"]
        assert "Unexpected frontmatter fields: foo" in issues[0].message

    def test_optional_field_types(self, tmp_path: Path):
        skill_dir = _write_skill(
            tmp_path,
            "x",
            frontmatter=(
                "name: x\ndescription: d\nlicense: 5\ncompatibility: ''\n"
                "allowed-tools: [Read]\nmetadata: not-a-map"
            ),
        )
        messages = " | ".join(i.message for i in validate_skill(skill_dir))
        assert "'license' must be a string" in messages
        assert "'compatibility' must be a non-empty string" in messages
        assert "'allowed-tools' must be a space-separated string" in messages
        assert "'metadata' must be a mapping" in messages

    def test_compatibility_too_long(self, tmp_path: Path):
        skill_dir = _write_skill(
            tmp_path, "x", frontmatter=f"name: x\ndescription: d\ncompatibility: {'c' * 501}"
        )
        assert any("Compatibility exceeds 500" in i.message for i in validate_skill(skill_dir))

    def test_body_recommendations_are_warnings(self, tmp_path: Path):
        body = "\n".join(["line"] * 501) + "\nsee references/missing.md"
        skill_dir = _write_skill(tmp_path, "x", body=body)
        issues = validate_skill(skill_dir)
        assert {i.level for i in issues} == {"warning"}
        messages = " | ".join(i.message for i in issues)
        assert "keep it under 500" in messages
        assert "references/missing.md" in messages

    def test_diagnostic_str(self, tmp_path: Path):
        skill_dir = _write_skill(tmp_path, "x", frontmatter="name: x\ndescription: d\nfoo: 1")
        assert str(validate_skill(skill_dir)[0]).startswith("error: ")


# ════════════════════════════════════════════════════════════════════════════
# 5. Registry
# ════════════════════════════════════════════════════════════════════════════


class TestSkillsRegistry:
    def test_discover_and_lookup(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha")
        _write_skill(tmp_path, "beta")
        registry = _registry(tmp_path)
        assert registry.names() == ["alpha", "beta"]
        assert len(registry) == 2
        assert "alpha" in registry
        assert registry.get("alpha") is not None
        assert registry.get("nope") is None

    def test_register_duplicate_file_is_idempotent(self, tmp_path: Path):
        meta, _ = load_skill(_write_skill(tmp_path, "x"))
        registry = SkillsRegistry()
        registry.register(meta)
        registry.register(meta)
        assert len(registry) == 1

    def test_register_conflict_raises_unless_replace(self, tmp_path: Path):
        first, _ = load_skill(_write_skill(tmp_path / "a", "x"))
        second, _ = load_skill(_write_skill(tmp_path / "b", "x"))
        registry = SkillsRegistry()
        registry.register(first)
        with pytest.raises(ValueError, match="Duplicate skill name"):
            registry.register(second)
        registry.register(second, replace=True)
        assert registry.get("x").skill_file == second.skill_file

    def test_earlier_directory_wins_and_later_is_shadowed(self, tmp_path: Path):
        _write_skill(tmp_path / "project", "shared", body="project version")
        _write_skill(tmp_path / "user", "shared", body="user version")
        registry = SkillsRegistry()
        registry.discover([tmp_path / "project", tmp_path / "user"])
        assert registry.load_content("shared") == "project version"
        assert any("shadowed" in d.message for d in registry.diagnostics)

    def test_unregister(self, tmp_path: Path):
        _write_skill(tmp_path, "x")
        registry = _registry(tmp_path)
        assert registry.unregister("x") is True
        assert registry.unregister("x") is False

    def test_tag_filter(self, tmp_path: Path):
        _write_skill(tmp_path, "a", metadata='metadata:\n  tags: "medical"\n')
        _write_skill(tmp_path, "b", metadata='metadata:\n  tags: "legal"\n')
        registry = _registry(tmp_path)
        assert [s.name for s in registry.get_all(tags={"medical"})] == ["a"]

    def test_load_content_hot_reload(self, tmp_path: Path):
        skill_dir = _write_skill(tmp_path, "x", body="v1")
        registry = _registry(tmp_path)
        assert registry.load_content("x") == "v1"

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("---\nname: x\ndescription: d\n---\nv2")
        stat = skill_file.stat()
        os.utime(skill_file, (stat.st_atime, stat.st_mtime + 10))

        assert registry.load_content("x", hot_reload=False) == "v1"
        assert registry.load_content("x", hot_reload=True) == "v2"

    def test_load_content_unknown(self):
        assert SkillsRegistry().load_content("nope") == ""

    def test_list_files(self, tmp_path: Path):
        _write_skill(
            tmp_path,
            "x",
            files={
                "scripts/run.py": "",
                "references/a.md": "",
                ".hidden": "",
                "__pycache__/c.pyc": "",
                "scripts/cache.pyc": "",
                ".agentflow-skill.json": "{}",
            },
        )
        registry = _registry(tmp_path)
        assert registry.list_files("x") == (["references/a.md", "scripts/run.py"], False)
        assert registry.list_files("x", limit=1) == (["references/a.md"], True)
        assert registry.list_files("nope") == ([], False)

    def test_read_file_unknown_skill(self):
        with pytest.raises(KeyError):
            SkillsRegistry().read_file("nope", "a.md", 100)

    def test_catalog(self, tmp_path: Path):
        _write_skill(tmp_path, "low", description="Low <priority> & stuff")
        _write_skill(
            tmp_path,
            "high",
            metadata='metadata:\n  priority: "5"\n  triggers: "review code"\n',
        )
        catalog = _registry(tmp_path).build_catalog()
        assert catalog.startswith("<available_skills>")
        assert catalog.index("<name>high</name>") < catalog.index("<name>low</name>")
        assert "Low &lt;priority&gt; &amp; stuff" in catalog
        assert "<triggers>review code</triggers>" in catalog
        # The description is always shown, even when triggers exist.
        assert catalog.count("<description>") == 2

    def test_catalog_empty(self):
        assert SkillsRegistry().build_catalog() == ""


# ════════════════════════════════════════════════════════════════════════════
# 6. Reading bundled files
# ════════════════════════════════════════════════════════════════════════════


class TestReadSkillFile:
    @pytest.fixture
    def meta(self, tmp_path: Path) -> SkillMeta:
        skill_dir = _write_skill(
            tmp_path / "skills",
            "x",
            files={
                "scripts/run.py": "print('hi')\n",
                "scripts/deploy": "#!/bin/sh\necho deploy\n",
                "scripts/latin.sh": b"caf\xe9\n",
                "assets/logo.png": b"\x89PNG\x00\x00binary",
                "references/big.md": "x" * 100,
            },
        )
        (tmp_path / "secret.txt").write_text("secret")
        meta, _ = load_skill(skill_dir)
        assert meta is not None
        return meta

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("scripts/run.py", "print('hi')"),
            ("scripts/deploy", "echo deploy"),
            ("./scripts/run.py", "print('hi')"),
            ("scripts\\run.py", "print('hi')"),
            ("SKILL.md", "name: x"),
        ],
    )
    def test_reads_any_text_file(self, meta, path, expected):
        assert expected in read_skill_file(meta, path, 10_000)

    def test_invalid_utf8_is_replaced(self, meta):
        assert read_skill_file(meta, "scripts/latin.sh", 10_000) == "caf\ufffd\n"

    def test_binary_file_described(self, meta):
        text = read_skill_file(meta, "assets/logo.png", 10_000)
        assert "binary file" in text

    def test_truncation(self, meta):
        text = read_skill_file(meta, "references/big.md", 10)
        assert text.startswith("x" * 10)
        assert "truncated: showing the first 10 of 100 bytes" in text

    def test_directory_listing(self, meta):
        text = read_skill_file(meta, "scripts", 10_000)
        assert "scripts/run.py" in text and "scripts/deploy" in text

    @pytest.mark.parametrize(
        ("path", "message"),
        [
            ("", "must not be empty"),
            ("/etc/passwd", "must be relative"),
            ("C:/Windows", "must be relative"),
            ("../../secret.txt", "outside the skill directory"),
            ("scripts/../../../secret.txt", "outside the skill directory"),
            ("missing.md", "not found"),
        ],
    )
    def test_rejected_paths(self, meta, path, message):
        with pytest.raises(SkillResourceError, match=message):
            read_skill_file(meta, path, 10_000)

    def test_symlink_escape_blocked(self, meta, tmp_path: Path):
        link = Path(meta.skill_dir) / "link.txt"
        try:
            link.symlink_to(tmp_path / "secret.txt")
        except OSError:
            pytest.skip("symlinks not supported")
        with pytest.raises(SkillResourceError, match="outside the skill directory"):
            read_skill_file(meta, "link.txt", 10_000)


# ════════════════════════════════════════════════════════════════════════════
# 7. Activation tools
# ════════════════════════════════════════════════════════════════════════════


class TestActivateSkillTool:
    @pytest.fixture
    def registry(self, tmp_path: Path) -> SkillsRegistry:
        _write_skill(
            tmp_path,
            "pdf",
            frontmatter="name: pdf\ndescription: PDFs\ncompatibility: Needs pypdf",
            body="Use scripts/extract.py",
            files={"scripts/extract.py": "print(1)"},
        )
        _write_skill(tmp_path, "plain", body="Plain body")
        return _registry(tmp_path)

    def test_returns_wrapped_content_with_resources(self, registry):
        tool = make_activate_skill_tool(
            registry, SkillConfig(), embed_catalog=False, can_read_files=True
        )
        result = tool("pdf")
        assert isinstance(result, str)
        assert result.startswith(skill_content_marker("pdf"))
        assert result.rstrip().endswith("</skill_content>")
        assert "Use scripts/extract.py" in result
        assert "Compatibility: Needs pypdf" in result
        assert "<file>scripts/extract.py</file>" in result
        assert 'read_skill_resource(skill_name="pdf"' in result
        assert "Skill directory:" not in result

    def test_include_skill_path(self, registry):
        tool = make_activate_skill_tool(
            registry,
            SkillConfig(include_skill_path=True),
            embed_catalog=False,
            can_read_files=True,
        )
        assert f"Skill directory: {registry.get('pdf').skill_dir}" in tool("pdf")

    def test_no_resources_section_without_files(self, registry):
        tool = make_activate_skill_tool(
            registry, SkillConfig(), embed_catalog=False, can_read_files=True
        )
        assert "<skill_resources>" not in tool("plain")

    def test_unknown_skill_is_error(self, registry):
        tool = make_activate_skill_tool(
            registry, SkillConfig(), embed_catalog=False, can_read_files=False
        )
        result = tool("nope")
        assert isinstance(result, ToolResult) and result.is_error
        assert "Available skills: pdf, plain" in result.message

    def test_empty_body_is_error(self, tmp_path: Path):
        _write_skill(tmp_path, "empty", body="")
        registry = _registry(tmp_path)
        tool = make_activate_skill_tool(
            registry, SkillConfig(), embed_catalog=False, can_read_files=False
        )
        result = tool("empty")
        assert isinstance(result, ToolResult) and result.is_error

    def test_records_activation_and_deduplicates(self, registry):
        tool = make_activate_skill_tool(
            registry, SkillConfig(), embed_catalog=False, can_read_files=False
        )
        state = AgentState()
        content = tool("plain", state=state)
        assert get_active_skills(state) == ["plain"]
        assert state.execution_meta.internal_data[ACTIVE_SKILLS_KEY] == ["plain"]

        state.context.append(Message.text_message(content, role="tool"))
        again = tool("plain", state=state)
        assert "already active" in again
        assert get_active_skills(state) == ["plain"]

    def test_reloads_when_content_left_context(self, registry):
        tool = make_activate_skill_tool(
            registry, SkillConfig(), embed_catalog=False, can_read_files=False
        )
        state = AgentState()
        mark_skill_active(state, "plain")
        assert tool("plain", state=state).startswith(skill_content_marker("plain"))

    def test_schema_uses_enum_and_hides_state(self, registry):
        tool = make_activate_skill_tool(
            registry, SkillConfig(), embed_catalog=False, can_read_files=False
        )
        spec = ToolNode([tool]).get_local_tool()[0]["function"]
        assert spec["name"] == "activate_skill"
        assert spec["parameters"]["properties"] == {
            "skill_name": {"type": "string", "enum": ["pdf", "plain"]}
        }
        assert spec["parameters"]["required"] == ["skill_name"]
        assert "Available skills" not in spec["description"]

    def test_embedded_catalog(self, registry):
        tool = make_activate_skill_tool(
            registry, SkillConfig(), embed_catalog=True, can_read_files=False
        )
        assert "- pdf: PDFs" in tool.__doc__


class TestReadSkillResourceTool:
    @pytest.fixture
    def tool(self, tmp_path: Path):
        _write_skill(tmp_path, "pdf", files={"scripts/extract.py": "print(1)"})
        registry = _registry(tmp_path)
        return make_read_skill_resource_tool(registry, SkillConfig(max_resource_bytes=1000))

    def test_reads_script(self, tool):
        result = tool("pdf", "scripts/extract.py")
        assert result == (
            '<skill_resource skill="pdf" path="scripts/extract.py">\nprint(1)\n</skill_resource>'
        )

    def test_missing_file_lists_bundled_files(self, tool):
        result = tool("pdf", "nope.md")
        assert isinstance(result, ToolResult) and result.is_error
        assert "Bundled files: scripts/extract.py" in result.message

    def test_unknown_skill(self, tool):
        result = tool("nope", "a.md")
        assert isinstance(result, ToolResult) and result.is_error

    def test_schema(self, tool):
        spec = ToolNode([tool]).get_local_tool()[0]["function"]
        assert spec["name"] == "read_skill_resource"
        assert spec["parameters"]["properties"]["skill_name"]["enum"] == ["pdf"]
        assert spec["parameters"]["required"] == ["skill_name", "path"]


class TestCatalogPrompt:
    def test_mentions_tools(self, tmp_path: Path):
        _write_skill(tmp_path, "x")
        registry = _registry(tmp_path)
        with_files = build_catalog_prompt(registry, can_read_files=True)
        assert "`activate_skill`" in with_files
        assert "`read_skill_resource`" in with_files
        assert "<available_skills>" in with_files
        assert "`read_skill_resource`" not in build_catalog_prompt(
            registry, can_read_files=False
        )

    def test_empty_registry(self):
        assert build_catalog_prompt(SkillsRegistry(), can_read_files=True) == ""


# ════════════════════════════════════════════════════════════════════════════
# 8. AgentSkillsMixin
# ════════════════════════════════════════════════════════════════════════════


class TestAgentSkillsMixin:
    def test_setup_none(self):
        mixin = _mixin()
        mixin._setup_skills(None)
        assert mixin._skills_config is None
        assert mixin._skills_registry is None
        assert mixin._skill_catalog_prompt is None

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="Expected SkillConfig"):
            _mixin()._setup_skills("not-a-config")

    def test_requires_tool_node(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha")
        with pytest.raises(RuntimeError, match="Skills require an existing ToolNode"):
            _mixin()._setup_skills(SkillConfig(skills_dir=str(tmp_path)))

    def test_no_skills_registers_nothing(self, tmp_path: Path):
        mixin = _mixin()  # no ToolNode needed when nothing was found
        mixin._setup_skills(SkillConfig(skills_dir=str(tmp_path)))
        assert mixin._skill_catalog_prompt is None
        assert mixin._build_skill_prompts(AgentState(), []) == []

    def test_registers_only_activate_without_files(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha")
        node = ToolNode([])
        _mixin(node)._setup_skills(SkillConfig(skills_dir=str(tmp_path)))
        assert set(node._funcs) == {"activate_skill"}

    def test_registers_both_tools_with_files(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha", files={"references/a.md": "a"})
        node = ToolNode([])
        _mixin(node)._setup_skills(SkillConfig(skills_dir=str(tmp_path)))
        assert set(node._funcs) == {"activate_skill", "read_skill_resource"}

    def test_defers_to_named_tool_node(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha", files={"references/a.md": "a"})
        mixin = _mixin()
        mixin.tool_node_name = "TOOL"
        mixin._setup_skills(SkillConfig(skills_dir=str(tmp_path)))
        assert [t.__name__ for t in mixin._extra_tools] == [
            "activate_skill",
            "read_skill_resource",
        ]

    def test_catalog_prompt(self, tmp_path: Path):
        _write_skill(tmp_path, "review")
        mixin = _mixin(ToolNode([]))
        mixin._setup_skills(SkillConfig(skills_dir=str(tmp_path)))
        base = [{"role": "system", "content": "Be helpful"}]
        result = mixin._build_skill_prompts(AgentState(), base)
        assert result[0] == base[0]
        assert "<name>review</name>" in result[1]["content"]
        assert len(base) == 1  # not mutated

    def test_catalog_disabled_moves_it_to_tool(self, tmp_path: Path):
        _write_skill(tmp_path, "review")
        node = ToolNode([])
        mixin = _mixin(node)
        mixin._setup_skills(SkillConfig(skills_dir=str(tmp_path), inject_catalog=False))
        assert mixin._build_skill_prompts(AgentState(), []) == []
        assert "- review:" in node._funcs["activate_skill"].__doc__

    def test_reinjects_active_skill_missing_from_context(self, tmp_path: Path):
        _write_skill(tmp_path, "review", body="Review rules")
        mixin = _mixin(ToolNode([]))
        mixin._setup_skills(SkillConfig(skills_dir=str(tmp_path)))

        state = AgentState()
        mark_skill_active(state, "review")
        prompts = mixin._build_skill_prompts(state, [])
        assert prompts[-1]["content"].startswith(skill_content_marker("review"))
        assert "Review rules" in prompts[-1]["content"]

        # Once the content is back in the conversation it is not duplicated.
        state.context.append(Message.text_message(prompts[-1]["content"], role="tool"))
        assert len(mixin._build_skill_prompts(state, [])) == 1

    def test_ignores_active_skills_from_other_registries(self, tmp_path: Path):
        _write_skill(tmp_path, "review")
        mixin = _mixin(ToolNode([]))
        mixin._setup_skills(SkillConfig(skills_dir=str(tmp_path)))
        state = AgentState()
        mark_skill_active(state, "someone-elses-skill")
        assert len(mixin._build_skill_prompts(state, [])) == 1
