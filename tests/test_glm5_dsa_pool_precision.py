"""Reference activation rounding must not depend on cache storage precision."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from vmlx_engine.models.glm5_next.glm5_next import Glm5NextIndexer


def bf16_round(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_pool_reference_rounding_with_fp32_cache(dtype, monkeypatch):
    monkeypatch.delenv("VMLX_GLM5_DSA_BF16", raising=False)
    args = SimpleNamespace(index_n_heads=2, index_head_dim=4, index_topk=8,
                           index_kpool=4, index_kpool_always_select_tail=True,
                           q_lora_rank=4, hidden_size=8)
    indexer = Glm5NextIndexer(args)
    indexer.update(tree_map(lambda x: x.astype(dtype), indexer.parameters()))
    # Establish the projection dtype as happens before every pool-cache update.
    packed = indexer.packed_states(mx.ones((1, 4, 8), dtype=dtype))
    assert packed.dtype == mx.float32
    round_value = (bf16_round if dtype == mx.bfloat16 else
                   lambda x: np.asarray(x, dtype=np.float16 if dtype == mx.float16
                                       else np.float32).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(packed), round_value(np.asarray(packed)))
    values = np.array([1.00390625, 1.01171875, 1.02734375, 1.04296875], np.float32)
    keys = np.broadcast_to(values[None, :, None], (1, 4, 4)).copy()
    # Zero pool gates give exactly representable 1/4 probabilities. The oracle
    # uses IEEE rounding, independent of MLX softmax/reduction implementation.
    source = mx.array(np.concatenate([keys, np.zeros_like(keys)], axis=-1))
    result = indexer.compress_pool_keys(source)
    expected = round_value(round_value(round_value(keys) * 0.25).sum(axis=1))
    assert result.dtype == mx.float32
    np.testing.assert_array_equal(np.asarray(result), expected[:, None, :])

