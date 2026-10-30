"""Mock-based Pg tests are not about schema setup: skip the first-use schema creation.

``PgCheckpointer`` now creates its schema on first query. Tests that exercise that path
(``test_pg_storage_v1_fixes.py``) opt out with the ``real_schema_init`` marker.
"""

import pytest

from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer


@pytest.fixture(autouse=True)
def _skip_lazy_schema_init(request, monkeypatch):
    if request.node.get_closest_marker("real_schema_init"):
        return
    original = PgCheckpointer.__init__

    def _init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._schema_initialized = True

    monkeypatch.setattr(PgCheckpointer, "__init__", _init)
