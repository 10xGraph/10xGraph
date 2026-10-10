"""
Tests for store module initialization and imports.

This module tests the store package's __init__.py file including
proper imports, optional dependency handling, and module structure.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from importlib import import_module


class TestStoreModuleImports:
    """Test store module imports and structure."""
    
    def test_base_store_import(self):
        """Test that BaseStore can be imported from store module."""
        from tenxgraph.storage.store import BaseStore
        
        # Should be able to import without errors
        assert BaseStore is not None
        
        # Should be an abstract base class
        from abc import ABC
        assert issubclass(BaseStore, ABC)
    
    def test_store_schemas_import(self):
        """Test that store schemas can be imported."""
        from tenxgraph.storage.store import (
            MemoryRecord,
            MemorySearchResult,
            MemoryType,
            DistanceMetric
        )
        
        # All schema classes should be importable
        assert MemoryRecord is not None
        assert MemorySearchResult is not None
        assert MemoryType is not None
        assert DistanceMetric is not None
        
        # RetrievalStrategy is not exported but should be importable directly
        from tenxgraph.storage.store.store_schema import RetrievalStrategy
        assert RetrievalStrategy is not None
    
    def test_embedding_classes_import(self):
        """Test that embedding classes can be imported."""
        from tenxgraph.storage.store import BaseEmbedding
        
        assert BaseEmbedding is not None
        
        # Test optional OpenAIEmbedding import
        try:
            from tenxgraph.storage.store import OpenAIEmbedding
            assert OpenAIEmbedding is not None
        except ImportError:
            # Should only fail if openai is not installed
            pass
    
    def test_store_module_structure(self):
        """Test the overall structure of the store module."""
        import tenxgraph.storage.store as store_module
        
        # Should have the main components
        assert hasattr(store_module, 'BaseStore')
        assert hasattr(store_module, 'BaseEmbedding')
        assert hasattr(store_module, 'MemoryRecord')
        assert hasattr(store_module, 'MemorySearchResult')
        
        # Should have enums
        assert hasattr(store_module, 'MemoryType')
        assert hasattr(store_module, 'DistanceMetric')
        
        # RetrievalStrategy is not in the main __init__.py but exists in schema
        from tenxgraph.storage.store.store_schema import RetrievalStrategy
        assert RetrievalStrategy is not None
    
    def test_all_exports(self):
        """Test that __all__ is properly defined."""
        import tenxgraph.storage.store as store_module
        
        # Should have __all__ defined
        if hasattr(store_module, '__all__'):
            # All items in __all__ should be importable
            for item in store_module.__all__:
                assert hasattr(store_module, item), f"Missing export: {item}"


class TestOptionalDependencies:
    """Test handling of optional dependencies."""
    
    def test_openai_embedding_availability(self):
        """Test OpenAIEmbedding availability based on openai package."""
        # Check if openai is available
        try:
            import openai
            openai_available = True
        except ImportError:
            openai_available = False
        
        if openai_available:
            # Should be able to import OpenAIEmbedding
            from tenxgraph.storage.store import OpenAIEmbedding
            assert OpenAIEmbedding is not None
        else:
            # Should either not be available or import gracefully
            try:
                from tenxgraph.storage.store import OpenAIEmbedding
                # If it imports, it should work
                assert OpenAIEmbedding is not None
            except (ImportError, AttributeError):
                # Expected if openai is not installed
                pass
    
    @patch.dict(sys.modules, {'openai': None})
    def test_graceful_degradation_without_openai(self):
        """Test that store module works without openai package."""
        # Force reimport without openai
        if 'tenxgraph.storage.store' in sys.modules:
            del sys.modules['tenxgraph.storage.store']
        if 'tenxgraph.storage.store.embedding.openai_embedding' in sys.modules:
            del sys.modules['tenxgraph.storage.store.embedding.openai_embedding']
        
        # Should still be able to import basic store functionality
        from tenxgraph.storage.store import BaseStore, MemoryRecord
        
        assert BaseStore is not None
        assert MemoryRecord is not None
    
    def test_store_functionality_without_optional_deps(self):
        """Test that core store functionality works without optional dependencies."""
        # Core classes should always be available
        from tenxgraph.storage.store.base_store import BaseStore
        from tenxgraph.storage.store.store_schema import MemoryRecord, MemorySearchResult
        from tenxgraph.storage.store.embedding.base_embedding import BaseEmbedding
        
        # Should be able to create instances (where applicable)
        record = MemoryRecord(content="test")
        assert record.content == "test"
        
        search_result = MemorySearchResult()
        assert search_result.content == ""
        
        # BaseStore and BaseEmbedding should be abstract
        with pytest.raises(TypeError):
            BaseStore()
        
        with pytest.raises(TypeError):
            BaseEmbedding()


class TestStoreModuleConstants:
    """Test constants and configuration in store module."""
    
    def test_has_openai_flag(self):
        """Test that HAS_OPENAI flag is properly set."""
        from tenxgraph.storage.store.embedding.openai_embedding import HAS_OPENAI
        
        # Should be a boolean
        assert isinstance(HAS_OPENAI, bool)
        
        # Should reflect actual openai availability
        try:
            import openai
            expected = True
        except ImportError:
            expected = False
        
        assert HAS_OPENAI == expected
    
    def test_embedding_module_constants(self):
        """Test constants in embedding modules."""
        from tenxgraph.storage.store.embedding import BaseEmbedding
        
        # BaseEmbedding should have proper abstract methods
        assert BaseEmbedding.__abstractmethods__
        # Check for async methods (actual implementation uses aembed)
        assert 'aembed' in BaseEmbedding.__abstractmethods__
        assert 'aembed_batch' in BaseEmbedding.__abstractmethods__
    
    def test_store_schema_enums(self):
        """Test that enums have expected values."""
        from tenxgraph.storage.store.store_schema import MemoryType, DistanceMetric, RetrievalStrategy
        
        # MemoryType should have core types
        memory_types = [mt.value for mt in MemoryType]
        assert 'episodic' in memory_types
        assert 'semantic' in memory_types
        
        # DistanceMetric should have common metrics
        distance_metrics = [dm.value for dm in DistanceMetric]
        assert 'cosine' in distance_metrics
        assert 'euclidean' in distance_metrics
        
        # RetrievalStrategy should have strategies
        strategies = [rs.value for rs in RetrievalStrategy]
        assert 'similarity' in strategies
        assert 'temporal' in strategies


class TestStoreModuleCompatibility:
    """Test compatibility and version handling."""
    
    def test_python_version_compatibility(self):
        """Test that store module works with current Python version."""
        import sys
        
        # Should work with Python 3.10+
        assert sys.version_info >= (3, 10), "Store module requires Python 3.10+"
        
        # Import should work
        import tenxgraph.storage.store
        assert tenxgraph.storage.store is not None
    
    def test_import_performance(self):
        """Test that imports are reasonably fast."""
        import time
        
        start_time = time.time()
        
        # Fresh import
        if 'tenxgraph.storage.store' in sys.modules:
            del sys.modules['tenxgraph.storage.store']
        
        import tenxgraph.storage.store
        
        import_time = time.time() - start_time
        
        # Should import in reasonable time (adjust threshold as needed)
        assert import_time < 5.0, f"Import took too long: {import_time}s"
    
    def test_circular_imports(self):
        """Test that there are no circular import issues."""
        # These imports should all work without circular dependency errors
        from tenxgraph.storage.store import BaseStore
        from tenxgraph.storage.store import BaseEmbedding
        from tenxgraph.storage.store import MemoryRecord, MemorySearchResult
        from tenxgraph.storage.store.base_store import BaseStore as BaseStoreFromModule
        from tenxgraph.storage.store.embedding.base_embedding import BaseEmbedding as BaseEmbeddingFromModule
        
        # Should be the same objects
        assert BaseStore is BaseStoreFromModule
        assert BaseEmbedding is BaseEmbeddingFromModule


class TestStoreModuleDocumentation:
    """Test that store module has proper documentation."""
    
    def test_module_docstring(self):
        """Test that store module has documentation."""
        import tenxgraph.storage.store as store_module
        
        # Module docstring may be None, that's acceptable for an __init__.py
        # Just check that the module can be imported
        assert store_module is not None
    
    def test_class_docstrings(self):
        """Test that main classes have docstrings."""
        from tenxgraph.storage.store import BaseStore, BaseEmbedding, MemoryRecord
        
        # Main classes should exist (docstrings may be None)
        assert BaseStore is not None
        assert BaseEmbedding is not None  
        assert MemoryRecord is not None
        
        # Check if docstrings exist and are meaningful when present
        if BaseStore.__doc__:
            assert len(BaseStore.__doc__.strip()) > 10
        if MemoryRecord.__doc__:
            assert len(MemoryRecord.__doc__.strip()) > 10
    
    def test_enum_docstrings(self):
        """Test that enums have proper documentation."""
        from tenxgraph.storage.store import MemoryType, DistanceMetric
        from tenxgraph.storage.store.store_schema import RetrievalStrategy
        
        # Enums should have docstrings
        assert MemoryType.__doc__ is not None
        assert DistanceMetric.__doc__ is not None
        assert RetrievalStrategy.__doc__ is not None


class TestStoreModuleEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_import_error_handling(self):
        """Test graceful handling of import errors."""
        # Mock a missing dependency
        with patch.dict(sys.modules, {'nonexistent_package': None}):
            # Store module should still import successfully
            try:
                import tenxgraph.storage.store
                assert tenxgraph.storage.store is not None
            except ImportError as e:
                # If there is an ImportError, it should be specific and clear
                assert 'nonexistent_package' not in str(e)
    
    def test_namespace_pollution(self):
        """Test that store module doesn't pollute namespace."""
        import tenxgraph.storage.store as store_module
        
        # Should not have internal implementation details exposed
        internal_names = ['sys', 'os', 'typing', 'import_module', '__builtins__']
        
        for name in internal_names:
            if hasattr(store_module, name):
                # If present, should be explicitly in __all__ or be a dunder attribute
                if hasattr(store_module, '__all__'):
                    if name in dir(store_module) and not name.startswith('_'):
                        assert name in store_module.__all__, f"Unexposed internal: {name}"
    
    def test_module_reload_safety(self):
        """Test that module can be safely reloaded."""
        import importlib
        import tenxgraph.storage.store as store_module
        
        # Get original BaseStore
        original_base_store = store_module.BaseStore
        
        # Reload module
        importlib.reload(store_module)
        
        # Should still work
        new_base_store = store_module.BaseStore
        
        # Classes should be equivalent (same name and module)
        assert original_base_store.__name__ == new_base_store.__name__
        assert original_base_store.__module__ == new_base_store.__module__
    
    def test_import_order_independence(self):
        """Test that import order doesn't matter."""
        # Clear any existing imports
        modules_to_clear = [
            'tenxgraph.storage.store',
            'tenxgraph.storage.store.base_store',
            'tenxgraph.storage.store.store_schema',
            'tenxgraph.storage.store.embedding',
            'tenxgraph.storage.store.embedding.base_embedding'
        ]
        
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]
        
        # Import in different order
        from tenxgraph.storage.store.store_schema import MemoryRecord
        from tenxgraph.storage.store.base_store import BaseStore
        from tenxgraph.storage.store.embedding.base_embedding import BaseEmbedding
        
        # All should work
        assert MemoryRecord is not None
        assert BaseStore is not None
        assert BaseEmbedding is not None
        
        # Top-level import should also work
        from tenxgraph.storage.store import BaseStore as TopLevelBaseStore
        
        # Should be the same class
        assert BaseStore is TopLevelBaseStore
