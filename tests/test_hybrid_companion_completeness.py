"""A restored hybrid prefix needs one companion for every recurrent slot."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mlx_lm.models.cache import ArraysCache, KVCache
from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator


def _state():
    cache = ArraysCache(size=2)
    cache[0] = np.zeros((1, 2, 2), dtype=np.float32)
    cache[1] = np.zeros((1, 2, 2), dtype=np.float32)
    return cache


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_companion_must_fill_exactly_the_recurrent_slots(count):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._is_hybrid = True
    gen._hybrid_kv_positions = [0]
    gen._hybrid_num_layers = 3
    gen._cache_model = SimpleNamespace(make_cache=lambda: [KVCache(), ArraysCache(size=2), ArraysCache(size=2)])
    states = [_state() for _ in range(count)]
    fetch = Mock(return_value=(states, True))
    gen._ssm_state_cache = SimpleNamespace(fetch=fetch)
    kv = KVCache()
    kv.keys = np.zeros((1, 1, 4, 2), dtype=np.float16)
    kv.values = np.zeros((1, 1, 4, 2), dtype=np.float16)
    kv.offset = 4
    result = gen._complete_hybrid_base_from_companion([kv], [1, 2, 3, 4], 4, cache_extra_keys=["image-id"])
    fetch.assert_called_once_with([1, 2, 3, 4], 4, cache_extra_keys=["image-id"])
    if count == 2:
        assert result[0] is kv
        assert all(result[i + 1] is states[i] for i in range(2))
        assert kv.offset == 4
    else:
        assert result is None
