"""Deprecated alias for :mod:`tenxgraph`.

``agentflow`` is the old import name of 10xGraph and is kept until 2.0. Every
``agentflow`` and ``agentflow.<sub>`` import resolves to the *same module object*
as ``tenxgraph`` / ``tenxgraph.<sub>``, so ``isinstance`` checks and module-level
singletons are shared rather than duplicated. New code should use ``tenxgraph``.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings


_OLD = "agentflow"
_NEW = "tenxgraph"


class _AliasLoader(importlib.abc.Loader):
    """Hand back the already-importable ``tenxgraph`` module under the old name."""

    def __init__(self, real_name: str) -> None:
        self._real_name = real_name
        self._real_spec = None

    def create_module(self, spec):
        module = importlib.import_module(self._real_name)
        # The import machinery overwrites ``module.__spec__`` with the alias spec.
        self._real_spec = module.__spec__
        return module

    def exec_module(self, module) -> None:
        # Already executed under its real name; restore the real spec.
        if self._real_spec is not None:
            module.__spec__ = self._real_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        real_name = _NEW + fullname[len(_OLD) :]
        try:
            real_spec = importlib.util.find_spec(real_name)
        except (ImportError, ValueError):
            return None
        if real_spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(real_name),
            is_package=real_spec.submodule_search_locations is not None,
        )


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

warnings.warn(
    "The 'agentflow' import name is deprecated and will be removed in 2.0; "
    "import 'tenxgraph' instead (for example 'from tenxgraph import StateGraph').",
    DeprecationWarning,
    stacklevel=2,
)

# Make ``import agentflow`` itself return the real package object.
sys.modules[_OLD] = importlib.import_module(_NEW)
