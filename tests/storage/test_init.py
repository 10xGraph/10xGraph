import pytest
import tenxgraph.storage


def test_storage_lazy_exports():
    assert tenxgraph.storage.make_agent_memory_tool is not None
    assert tenxgraph.storage.make_user_memory_tool is not None
    assert tenxgraph.storage.memory_tool is not None

    with pytest.raises(AttributeError):
        _ = tenxgraph.storage.invalid_attribute_name_xxx
