"""Tests for EvalSetBuilder fluent builder."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from tenxgraph.qa.evaluation.dataset.builder import EvalSetBuilder
from tenxgraph.qa.evaluation.dataset.eval_set import EvalSet, ToolCall


class TestEvalSetBuilderInit:
    def test_default_name(self):
        b = EvalSetBuilder()
        assert b.name == "eval_set"

    def test_custom_name(self):
        b = EvalSetBuilder("my_tests")
        assert b.name == "my_tests"

    def test_unique_eval_set_id(self):
        b1 = EvalSetBuilder()
        b2 = EvalSetBuilder()
        assert b1.eval_set_id != b2.eval_set_id

    def test_cases_empty_on_init(self):
        b = EvalSetBuilder()
        assert b.cases == []


class TestAddCase:
    def test_add_single_case(self):
        b = EvalSetBuilder("test")
        b.add_case(query="Hello", expected="Hi")
        assert len(b.cases) == 1

    def test_case_id_auto_generated(self):
        b = EvalSetBuilder("test")
        b.add_case(query="q", expected="r")
        assert b.cases[0].eval_id == "case_1"

    def test_case_id_custom(self):
        b = EvalSetBuilder("test")
        b.add_case(query="q", expected="r", case_id="custom-id")
        assert b.cases[0].eval_id == "custom-id"

    def test_multiple_cases_numbered(self):
        b = EvalSetBuilder("test")
        b.add_case("q1", "r1")
        b.add_case("q2", "r2")
        b.add_case("q3", "r3")
        assert [c.eval_id for c in b.cases] == ["case_1", "case_2", "case_3"]

    def test_method_chaining(self):
        b = EvalSetBuilder("test")
        result = b.add_case("q", "r")
        assert result is b  # returns self

    def test_add_case_with_expected_tools_as_strings(self):
        b = EvalSetBuilder("test")
        b.add_case("q", "r", expected_tools=["get_weather", "send_email"])
        case = b.cases[0]
        # The case's first invocation should have tool trajectory
        tools = case.conversation[0].expected_tool_trajectory
        assert len(tools) == 2
        assert tools[0].name == "get_weather"

    def test_add_case_with_expected_tools_as_tool_calls(self):
        b = EvalSetBuilder("test")
        tc = ToolCall(name="my_tool", args={"key": "val"})
        b.add_case("q", "r", expected_tools=[tc])
        tools = b.cases[0].conversation[0].expected_tool_trajectory
        assert tools[0].name == "my_tool"

    def test_add_case_with_expected_node_order(self):
        b = EvalSetBuilder("test")
        b.add_case("q", "r", expected_node_order=["MAIN", "TOOL", "MAIN"])
        nodes = b.cases[0].conversation[0].expected_node_order
        assert nodes == ["MAIN", "TOOL", "MAIN"]

    def test_add_case_with_name_and_description(self):
        b = EvalSetBuilder("test")
        b.add_case("q", "r", name="Test Name", description="test desc")
        assert b.cases[0].name == "Test Name"
        assert b.cases[0].description == "test desc"


class TestAddMultiTurn:
    def test_add_multi_turn_basic(self):
        b = EvalSetBuilder("test")
        b.add_multi_turn([("Hi", "Hello"), ("How are you?", "Fine")])
        assert len(b.cases) == 1
        assert len(b.cases[0].conversation) == 2

    def test_multi_turn_case_id_auto(self):
        b = EvalSetBuilder("test")
        b.add_multi_turn([("q", "r")])
        assert b.cases[0].eval_id == "case_1"

    def test_multi_turn_with_custom_id(self):
        b = EvalSetBuilder("test")
        b.add_multi_turn([("q", "r")], case_id="multi-1")
        assert b.cases[0].eval_id == "multi-1"

    def test_method_chaining(self):
        b = EvalSetBuilder("test")
        result = b.add_multi_turn([("q", "r")])
        assert result is b


class TestAddToolTest:
    def test_add_tool_test(self):
        b = EvalSetBuilder("test")
        b.add_tool_test(query="What's the weather?", tool_name="get_weather")
        assert len(b.cases) == 1
        tools = b.cases[0].conversation[0].expected_tool_trajectory
        assert len(tools) == 1
        assert tools[0].name == "get_weather"

    def test_add_tool_test_with_args(self):
        b = EvalSetBuilder("test")
        b.add_tool_test("query", "my_tool", tool_args={"city": "Paris"})
        tools = b.cases[0].conversation[0].expected_tool_trajectory
        assert tools[0].args == {"city": "Paris"}

    def test_add_tool_test_custom_expected_response(self):
        b = EvalSetBuilder("test")
        b.add_tool_test("q", "tool", expected_response="custom response")
        # Check it uses the custom response
        invocation = b.cases[0].conversation[0]
        assert invocation.expected_final_response is not None

    def test_add_tool_test_default_response(self):
        b = EvalSetBuilder("test")
        b.add_tool_test("q", "my_tool")
        invocation = b.cases[0].conversation[0]
        assert invocation.expected_final_response is not None
        assert "my_tool" in invocation.expected_final_response.get_text()


class TestBuild:
    def test_build_returns_eval_set(self):
        b = EvalSetBuilder("my_set")
        b.add_case("q", "r")
        eval_set = b.build()
        assert isinstance(eval_set, EvalSet)

    def test_build_preserves_name(self):
        b = EvalSetBuilder("my_name")
        b.add_case("q", "r")
        eval_set = b.build()
        assert eval_set.name == "my_name"

    def test_build_preserves_cases(self):
        b = EvalSetBuilder("test")
        b.add_case("q1", "r1")
        b.add_case("q2", "r2")
        eval_set = b.build()
        assert len(eval_set.eval_cases) == 2


class TestSave:
    def test_save_and_load(self, tmp_path):
        b = EvalSetBuilder("saved_set")
        b.add_case("q", "r")
        path = str(tmp_path / "eval_set.json")
        eval_set = b.save(path)
        assert isinstance(eval_set, EvalSet)
        assert os.path.exists(path)

        loaded = EvalSet.from_file(path)
        assert loaded.name == "saved_set"
        assert len(loaded.eval_cases) == 1


class TestClassMethods:
    def test_from_conversations(self):
        convs = [
            {"user": "Hello", "assistant": "Hi"},
            {"user": "Bye", "assistant": "Goodbye"},
        ]
        eval_set = EvalSetBuilder.from_conversations(convs, name="conv_tests")
        assert eval_set.name == "conv_tests"
        assert len(eval_set.eval_cases) == 2
        assert eval_set.eval_cases[0].eval_id == "conv_1"

    def test_from_conversations_default_name(self):
        convs = [{"user": "q", "assistant": "a"}]
        eval_set = EvalSetBuilder.from_conversations(convs)
        assert eval_set.name == "conversation_tests"

    def test_from_file(self, tmp_path):
        b = EvalSetBuilder("file_test")
        b.add_case("q", "r", case_id="case_1")
        path = str(tmp_path / "set.json")
        b.save(path)

        loaded_b = EvalSetBuilder.from_file(path)
        assert loaded_b.name == "file_test"
        assert len(loaded_b.cases) == 1

    def test_quick_builder(self):
        eval_set = EvalSetBuilder.quick(
            ("Hello", "Hi!"),
            ("Bye", "Goodbye"),
        )
        assert isinstance(eval_set, EvalSet)
        assert len(eval_set.eval_cases) == 2
        assert eval_set.eval_cases[0].eval_id == "test_1"
        assert eval_set.eval_cases[1].eval_id == "test_2"

    def test_quick_builder_single(self):
        eval_set = EvalSetBuilder.quick(("query", "response"))
        assert len(eval_set.eval_cases) == 1


class TestNormaliseTools:
    def test_none_returns_empty(self):
        result = EvalSetBuilder._normalise_tools(None)
        assert result == []

    def test_empty_list_returns_empty(self):
        result = EvalSetBuilder._normalise_tools([])
        assert result == []

    def test_string_converted_to_tool_call(self):
        result = EvalSetBuilder._normalise_tools(["tool_name"])
        assert len(result) == 1
        assert result[0].name == "tool_name"
        assert result[0].args == {}

    def test_tool_call_passed_through(self):
        tc = ToolCall(name="my_tool", args={"k": "v"})
        result = EvalSetBuilder._normalise_tools([tc])
        assert len(result) == 1
        assert result[0] is tc

    def test_mixed_list(self):
        tc = ToolCall(name="tool2", args={})
        result = EvalSetBuilder._normalise_tools(["tool1", tc])
        assert len(result) == 2
        assert result[0].name == "tool1"
        assert result[1] is tc
