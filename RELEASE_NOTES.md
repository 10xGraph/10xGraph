# 10xscale-agentflow 0.10.0

## This is the final release of `10xscale-agentflow`

Agentflow is now **10xGraph**. The project continues under a new name because
"Agentflow" is shared by several unrelated projects, which made it hard to find.
Nothing about the framework, its license or its maintainers changes.

No further versions of `10xscale-agentflow` will be published to PyPI. Existing
installs keep working; pin `10xscale-agentflow==0.10.0` if you need to stay on it.

| | Before | After |
|---|---|---|
| PyPI package | `10xscale-agentflow` | `10xgraph` |
| Import | `import agentflow` | `import tenxgraph` |
| Website | agentflow.10xscale.ai | [10xgraph.com](https://10xgraph.com) |
| GitHub | github.com/10xHub | [github.com/10xGraph](https://github.com/10xGraph) |

The import name is `tenxgraph` because a Python identifier cannot start with a digit.
`10xgraph` keeps `agentflow` importable as a deprecated alias until 2.0, so existing
code runs unchanged while you migrate:

```bash
pip uninstall 10xscale-agentflow
pip install 10xgraph
```

```python
# before
from agentflow.core.graph import StateGraph
# after
from tenxgraph import StateGraph
```

The API server/CLI (`10xscale-agentflow-cli`) and the TypeScript client
(`@10xscale/agentflow-client`) are renamed too. Their new package names are announced
in the 10xGraph repositories.

## Highlights

- **`interrupt()` from inside a node or tool.** Pause a run, save the thread, and resume
  with `ainvoke({"resume": value}, config)`; the node re-runs and `interrupt()` returns
  `value`. Works inside tools, including parallel tool calls.
- **Per-run client tools** via `config["remote_tools"]`, without mutating the graph.
- **Skills follow the Agent Skills specification (agentskills.io).** Skills written for
  Claude Code, Codex or GitHub Copilot load unchanged. `activate_skill` and
  `read_skill_resource` replace `set_skill`; activated skills survive context trimming.
- **Client-side tool calls now pause the graph.** Previously the check never matched and
  the graph kept running past a remote tool call.
- **Checkpointers no longer import a class named by stored data.** Rows are rebuilt into
  the state class bound at `compile()`. Old rows still load.
- **`Inject[...]` defaults resolve per call**, so the first graph to run no longer decides
  which checkpointer, publisher or store every later graph uses.

## Breaking changes

- **Skills API.** `set_skill` is replaced by `activate_skill` / `read_skill_resource`;
  `SkillConfig.inject_trigger_table` is renamed `inject_catalog`; `triggers`, `tags` and
  `priority` move under `metadata`.
- **One checkpointer instance serves one state class.** Give each graph with its own state
  class its own checkpointer, or call `checkpointer.bind_state_type(MyState)` when reading
  threads outside a compiled graph.

See `CHANGELOG.md` for the full list with migration steps.
