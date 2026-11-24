import tenxgraph.storage.store as store_module


def test_all_exports():
    """Every name in __all__ resolves, including the lazily loaded memory tools."""
    for item in store_module.__all__:
        assert hasattr(store_module, item), f"Missing export: {item}"
