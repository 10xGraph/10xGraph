"""Root conftest for examples/evaluation — ensures agentflow is importable."""

import sys
from pathlib import Path

# Add the project root (two levels up) to sys.path so that
# ``import tenxgraph`` works regardless of where pytest is invoked.
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
