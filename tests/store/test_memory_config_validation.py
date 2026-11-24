"""MemoryConfig and scope configs validate limits and retrieval_mode through pydantic fields."""

import pytest
from pydantic import ValidationError

from tenxgraph.storage.store import AgentMemoryConfig, MemoryConfig, UserMemoryConfig
from tenxgraph.storage.store.long_term_memory import ReadMode


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": -1}, {"score_threshold": -0.1}])
def test_memory_config_rejects_out_of_range(kwargs):
    with pytest.raises(ValidationError):
        MemoryConfig(**kwargs)


@pytest.mark.parametrize("cls", [UserMemoryConfig, AgentMemoryConfig])
@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"score_threshold": -1}])
def test_scope_config_rejects_out_of_range(cls, kwargs):
    with pytest.raises(ValidationError):
        cls(**kwargs)


def test_defaults_and_boundaries_are_accepted():
    cfg = MemoryConfig(limit=1, score_threshold=0)
    assert (cfg.limit, cfg.score_threshold, cfg.retrieval_mode) == (1, 0, ReadMode.POSTLOAD)
    assert UserMemoryConfig().limit is None
    assert UserMemoryConfig(limit=1, score_threshold=0).score_threshold == 0


def test_retrieval_mode_string_is_coerced():
    assert MemoryConfig(retrieval_mode="preload").retrieval_mode is ReadMode.PRELOAD
    with pytest.raises(ValidationError):
        MemoryConfig(retrieval_mode="nope")
