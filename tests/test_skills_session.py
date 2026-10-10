"""Tests for the session-preloaded skills mode in AgentFlow.

Covers:
- SkillConfig validation for mode and preload_from fields
- AgentSkillsMixin._setup_skills in session mode (no activate_skill tool, no catalog)
- AgentSkillsMixin._build_skill_prompts in session mode (reads state, loads content)
- Edge cases: missing skill, empty skill_name, hot_reload behaviour, caching
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tenxgraph.core.skills.models import SkillConfig, SkillMeta
from tenxgraph.core.skills.registry import SkillsRegistry


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _make_skill_dir(
    tmp_path: Path,
    name: str,
    *,
    description: str = "A test skill",
    body: str = "# Skill body\nSome instructions.",
    triggers: list[str] | None = None,
) -> Path:
    """Create a minimal skill directory with a SKILL.md."""
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    yaml_lines = [f"name: {name}", f"description: {description}"]
    if triggers:
        yaml_lines.append("metadata:")
        yaml_lines.append("  triggers:")
        for t in triggers:
            yaml_lines.append(f"    - {t}")

    yaml_section = "\n".join(yaml_lines)
    content = f"---\n{yaml_section}\n---\n{body}"
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def _make_mixin(tool_node=None):
    """Return a fresh AgentSkillsMixin with a pre-set _tool_node."""
    from tenxgraph.core.graph.agent_internal.skills import AgentSkillsMixin

    mixin = AgentSkillsMixin()
    mixin._tool_node = tool_node
    return mixin


# ────────────────────────────────────────────────────────────────────────────
# 1. SkillConfig model validation
# ────────────────────────────────────────────────────────────────────────────


class TestSkillConfigSessionMode:
    def test_default_mode_is_on_demand(self):
        cfg = SkillConfig()
        assert cfg.mode == "on-demand"

    def test_session_mode_accepted(self):
        cfg = SkillConfig(mode="session", preload_from="SKILL_NAME")
        assert cfg.mode == "session"
        assert cfg.preload_from == "SKILL_NAME"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValidationError):
            SkillConfig(mode="unknown")

    def test_preload_from_none_by_default(self):
        cfg = SkillConfig()
        assert cfg.preload_from is None

    def test_preload_from_strips_whitespace(self):
        cfg = SkillConfig(preload_from="  MY_FIELD  ")
        assert cfg.preload_from == "MY_FIELD"

    def test_preload_from_empty_string_raises(self):
        with pytest.raises(ValidationError, match="preload_from must not be an empty string"):
            SkillConfig(preload_from="")

    def test_preload_from_non_identifier_raises(self):
        with pytest.raises(ValidationError, match="not a valid Python identifier"):
            SkillConfig(preload_from="my-field")  # hyphens not allowed

    def test_preload_from_valid_identifier_accepted(self):
        cfg = SkillConfig(preload_from="skill_name_123")
        assert cfg.preload_from == "skill_name_123"


# ────────────────────────────────────────────────────────────────────────────
# 2. _setup_skills — session mode
# ────────────────────────────────────────────────────────────────────────────


class TestSetupSkillsSessionMode:
    def test_session_mode_does_not_require_tool_node(self, tmp_path: Path):
        """Session mode should not raise even when no ToolNode is configured."""
        _make_skill_dir(tmp_path, "fashion")

        mixin = _make_mixin(tool_node=None)
        cfg = SkillConfig(skills_dir=str(tmp_path), mode="session", preload_from="SKILL_NAME")
        # Should not raise RuntimeError about missing ToolNode
        mixin._setup_skills(cfg)

        assert mixin._skills_config is not None
        assert mixin._skills_registry is not None

    def test_session_mode_no_catalog_prompt(self, tmp_path: Path):
        """The skill catalog must not be built in session mode."""
        _make_skill_dir(tmp_path, "fashion", triggers=["show me clothes"])

        mixin = _make_mixin(tool_node=None)
        mixin._setup_skills(
            SkillConfig(skills_dir=str(tmp_path), mode="session", preload_from="SKILL_NAME")
        )

        assert mixin._skill_catalog_prompt is None

    def test_session_mode_no_activate_tool_added(self, tmp_path: Path):
        """activate_skill must not be queued or attached in session mode."""
        _make_skill_dir(tmp_path, "fashion")

        from tenxgraph.core.graph.tool_node import ToolNode

        tool_node = ToolNode([])
        mixin = _make_mixin(tool_node=tool_node)
        mixin._setup_skills(
            SkillConfig(skills_dir=str(tmp_path), mode="session", preload_from="SKILL_NAME")
        )

        assert "activate_skill" not in tool_node._funcs
        assert not getattr(mixin, "_extra_tools", [])

    def test_session_mode_no_extra_tools_queued_without_tool_node(self, tmp_path: Path):
        """When no ToolNode is configured, no _extra_tools list should be created."""
        _make_skill_dir(tmp_path, "fashion")

        mixin = _make_mixin(tool_node=None)
        mixin._setup_skills(
            SkillConfig(skills_dir=str(tmp_path), mode="session", preload_from="SKILL_NAME")
        )

        assert getattr(mixin, "_extra_tools", None) is None

    def test_session_mode_registers_resource_tool_for_bundled_files(self, tmp_path: Path):
        from tenxgraph.core.graph.tool_node import ToolNode

        skill_dir = _make_skill_dir(tmp_path, "fashion")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "catalog.md").write_text("items")

        tool_node = ToolNode([])
        mixin = _make_mixin(tool_node=tool_node)
        mixin._setup_skills(
            SkillConfig(skills_dir=str(tmp_path), mode="session", preload_from="SKILL_NAME")
        )

        assert set(tool_node._funcs) == {"read_skill_resource"}
        result = mixin._build_skill_prompts(SimpleNamespace(SKILL_NAME="fashion"), [])
        assert "<file>references/catalog.md</file>" in result[0]["content"]
        assert "read_skill_resource(" in result[0]["content"]

    def test_session_mode_bundled_files_without_tool_node(self, tmp_path: Path):
        """Without a ToolNode, files are listed but no read hint is given."""
        skill_dir = _make_skill_dir(tmp_path, "fashion")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "catalog.md").write_text("items")

        mixin = _make_mixin(tool_node=None)
        mixin._setup_skills(
            SkillConfig(skills_dir=str(tmp_path), mode="session", preload_from="SKILL_NAME")
        )

        assert getattr(mixin, "_extra_tools", None) is None
        result = mixin._build_skill_prompts(SimpleNamespace(SKILL_NAME="fashion"), [])
        assert "<file>references/catalog.md</file>" in result[0]["content"]
        assert "read_skill_resource(" not in result[0]["content"]

    def test_on_demand_mode_still_works_with_named_tool_node(self, tmp_path: Path):
        """Ensure on-demand mode is unaffected by the new session-mode branching."""
        _make_skill_dir(tmp_path, "support")

        mixin = _make_mixin(tool_node=None)
        mixin.tool_node_name = "TOOL"
        mixin._setup_skills(SkillConfig(skills_dir=str(tmp_path), mode="on-demand"))

        assert getattr(mixin, "_extra_tools", None) is not None
        assert len(mixin._extra_tools) == 1


# ────────────────────────────────────────────────────────────────────────────
# 3. _build_skill_prompts — session mode
# ────────────────────────────────────────────────────────────────────────────


class TestBuildSkillPromptsSessionMode:
    def _setup(self, tmp_path: Path, skill_name: str, body: str, preload_from: str = "SKILL_NAME"):
        _make_skill_dir(tmp_path, skill_name, body=body)
        mixin = _make_mixin(tool_node=None)
        mixin._setup_skills(
            SkillConfig(
                skills_dir=str(tmp_path),
                mode="session",
                preload_from=preload_from,
            )
        )
        return mixin

    def test_injects_skill_content_as_system_message(self, tmp_path: Path):
        body = "# Fashion Expert\nYou are an expert fashion advisor."
        mixin = self._setup(tmp_path, "fashion", body=body)

        state = SimpleNamespace(SKILL_NAME="fashion")
        base = [{"role": "system", "content": "Be helpful"}]
        result = mixin._build_skill_prompts(state, base)

        assert len(result) == 2
        assert result[0] == base[0]
        assert result[1]["role"] == "system"
        assert "Fashion Expert" in result[1]["content"]
        assert result[1]["content"].startswith('<skill_content name="fashion">')

    def test_no_skill_content_when_state_field_empty(self, tmp_path: Path):
        _make_skill_dir(tmp_path, "fashion")
        mixin = _make_mixin(tool_node=None)
        mixin._setup_skills(
            SkillConfig(skills_dir=str(tmp_path), mode="session", preload_from="SKILL_NAME")
        )

        state = SimpleNamespace(SKILL_NAME="")
        base = [{"role": "system", "content": "Be helpful"}]
        result = mixin._build_skill_prompts(state, base)

        assert result == base

    def test_no_skill_content_when_state_is_none(self, tmp_path: Path):
        _make_skill_dir(tmp_path, "fashion")
        mixin = _make_mixin(tool_node=None)
        mixin._setup_skills(
            SkillConfig(skills_dir=str(tmp_path), mode="session", preload_from="SKILL_NAME")
        )

        base = [{"role": "system", "content": "Be helpful"}]
        result = mixin._build_skill_prompts(None, base)

        assert result == base

    def test_no_skill_content_when_state_field_missing(self, tmp_path: Path):
        _make_skill_dir(tmp_path, "fashion")
        mixin = _make_mixin(tool_node=None)
        mixin._setup_skills(
            SkillConfig(skills_dir=str(tmp_path), mode="session", preload_from="SKILL_NAME")
        )

        # State has no SKILL_NAME attribute
        state = SimpleNamespace(OTHER_FIELD="something")
        base = [{"role": "system", "content": "Be helpful"}]
        result = mixin._build_skill_prompts(state, base)

        assert result == base

    def test_no_skill_content_when_skill_not_found_in_registry(self, tmp_path: Path):
        """Unknown skill names return empty content — no system message appended."""
        _make_skill_dir(tmp_path, "fashion")
        mixin = _make_mixin(tool_node=None)
        mixin._setup_skills(
            SkillConfig(skills_dir=str(tmp_path), mode="session", preload_from="SKILL_NAME")
        )

        state = SimpleNamespace(SKILL_NAME="nonexistent-skill")
        base = [{"role": "system", "content": "Be helpful"}]
        result = mixin._build_skill_prompts(state, base)

        assert result == base

    def test_does_not_mutate_original_list(self, tmp_path: Path):
        body = "# Fashion Expert\nInstructions here."
        mixin = self._setup(tmp_path, "fashion", body=body)

        state = SimpleNamespace(SKILL_NAME="fashion")
        base = [{"role": "system", "content": "Be helpful"}]
        original_len = len(base)
        mixin._build_skill_prompts(state, base)

        assert len(base) == original_len

    def test_preload_from_reads_custom_field_name(self, tmp_path: Path):
        body = "# Bridal Expert\nSpecialize in bridal wear."
        _make_skill_dir(tmp_path, "bridal", body=body)
        mixin = _make_mixin(tool_node=None)
        mixin._setup_skills(
            SkillConfig(
                skills_dir=str(tmp_path),
                mode="session",
                preload_from="AI_PERSONA",
            )
        )

        state = SimpleNamespace(AI_PERSONA="bridal")
        base = [{"role": "system", "content": "You are helpful"}]
        result = mixin._build_skill_prompts(state, base)

        assert len(result) == 2
        assert "Bridal Expert" in result[1]["content"]

    def test_caching_stores_content_after_first_call(self, tmp_path: Path):
        body = "# Fashion Expert\nInstructions."
        mixin = self._setup(tmp_path, "fashion", body=body)

        state = SimpleNamespace(SKILL_NAME="fashion")
        base = [{"role": "system", "content": "Be helpful"}]

        # First call should populate the registry's body cache
        mixin._build_skill_prompts(state, base)
        assert "Fashion Expert" in mixin._skills_registry._bodies["fashion"][1]

    def test_multiple_skills_cached_independently(self, tmp_path: Path):
        _make_skill_dir(tmp_path, "fashion", body="# Fashion skill body")
        _make_skill_dir(tmp_path, "bridal", body="# Bridal skill body")

        mixin = _make_mixin(tool_node=None)
        mixin._setup_skills(
            SkillConfig(skills_dir=str(tmp_path), mode="session", preload_from="SKILL_NAME")
        )

        base = [{"role": "system", "content": "Be helpful"}]
        mixin._build_skill_prompts(SimpleNamespace(SKILL_NAME="fashion"), base)
        mixin._build_skill_prompts(SimpleNamespace(SKILL_NAME="bridal"), base)

        bodies = mixin._skills_registry._bodies
        assert "Fashion skill body" in bodies["fashion"][1]
        assert "Bridal skill body" in bodies["bridal"][1]

    def test_no_preload_from_returns_base_prompt(self, tmp_path: Path):
        """mode='session' without preload_from raises at config creation time."""
        with pytest.raises(ValidationError, match="preload_from.*must be set"):
            SkillConfig(skills_dir=str(tmp_path), mode="session", preload_from=None)

    def test_session_mode_no_catalog_in_output(self, tmp_path: Path):
        """The skill catalog must never appear in session-mode prompts."""
        _make_skill_dir(tmp_path, "fashion", triggers=["show me dresses"])
        mixin = _make_mixin(tool_node=None)
        mixin._setup_skills(
            SkillConfig(
                skills_dir=str(tmp_path),
                mode="session",
                preload_from="SKILL_NAME",
                inject_catalog=True,  # ignored in session mode
            )
        )

        state = SimpleNamespace(SKILL_NAME="fashion")
        base = [{"role": "system", "content": "Be helpful"}]
        result = mixin._build_skill_prompts(state, base)

        combined = " ".join(m.get("content", "") for m in result)
        assert "<available_skills>" not in combined
        assert "activate_skill" not in combined


# ────────────────────────────────────────────────────────────────────────────
# 4. on-demand mode still works (regression guard)
# ────────────────────────────────────────────────────────────────────────────


class TestOnDemandModeRegression:
    def test_catalog_still_appended_in_on_demand(self, tmp_path: Path):
        from tenxgraph.core.graph.tool_node import ToolNode

        _make_skill_dir(tmp_path, "support", triggers=["need help"])

        mixin = _make_mixin(tool_node=ToolNode([]))
        mixin._setup_skills(
            SkillConfig(skills_dir=str(tmp_path), mode="on-demand", inject_catalog=True)
        )

        base = [{"role": "system", "content": "Be helpful"}]
        result = mixin._build_skill_prompts(None, base)

        assert len(result) == 2
        assert "<available_skills>" in result[1]["content"]

    def test_on_demand_does_not_use_state_for_skill_selection(self, tmp_path: Path):
        """State.SKILL_NAME is irrelevant in on-demand mode."""
        from tenxgraph.core.graph.tool_node import ToolNode

        _make_skill_dir(tmp_path, "support", triggers=["need help"])

        mixin = _make_mixin(tool_node=ToolNode([]))
        mixin._setup_skills(SkillConfig(skills_dir=str(tmp_path), mode="on-demand"))

        state = SimpleNamespace(SKILL_NAME="support")
        base = [{"role": "system", "content": "Be helpful"}]
        result = mixin._build_skill_prompts(state, base)

        # on-demand only adds the catalog, not the skill body directly
        content = " ".join(m.get("content", "") for m in result)
        assert "<available_skills>" in content
        assert "Skill body" not in content
