"""The deprecated ``agentflow`` import name aliases ``tenxgraph`` until 2.0."""

import subprocess
import sys
import textwrap


def _run(code: str, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *flags, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_submodule_import_works():
    res = _run("from agentflow.core.graph import StateGraph; print(StateGraph.__name__)")
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "StateGraph"


def test_objects_are_identical_to_tenxgraph():
    res = _run(
        """
        import sys
        import agentflow
        import tenxgraph
        from agentflow.core.graph import StateGraph as Old
        from tenxgraph.core.graph import StateGraph as New
        import agentflow.core.state as old_state
        import tenxgraph.core.state as new_state
        assert agentflow is tenxgraph
        assert Old is New
        assert old_state is new_state
        assert sys.modules["agentflow.core.state"] is sys.modules["tenxgraph.core.state"]
        assert new_state.__spec__.name == "tenxgraph.core.state"
        print("ok")
        """
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"


def test_deprecation_warning_emitted_once():
    res = _run(
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import agentflow
            import agentflow.core.graph
            import agentflow.utils
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)
               and "tenxgraph" in str(w.message)]
        assert len(dep) == 1, dep
        print("ok")
        """
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"


def test_unknown_submodule_raises():
    res = _run("import agentflow.does_not_exist")
    assert res.returncode != 0
    assert "ModuleNotFoundError" in res.stderr


def test_tenxgraph_lazy_top_level_exports():
    res = _run(
        """
        import sys
        import tenxgraph
        assert "tenxgraph.core.graph" not in sys.modules
        from tenxgraph import StateGraph, Agent, ToolNode, AgentState, Message, START, END
        from tenxgraph.core.graph import StateGraph as SG
        assert StateGraph is SG
        print("ok")
        """
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"
