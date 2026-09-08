"""The real Qwen4 selector must satisfy the native kernel's prefix ABI."""

import mlx.core as mx
import pytest

from vmlx_engine.metal.qwen4_prefill_direct import qsa_prefill_direct_topk_buffer
from vmlx_engine.models.qwen4_exp.language import QSAIndexer, Qwen4ExpTextArgs


@pytest.mark.parametrize("rows", [2052, 4096, 8192])
def test_real_selector_matches_dense_mask_and_valid_prefix(rows):
    mx.random.seed(61)
    indexer = QSAIndexer(Qwen4ExpTextArgs(hidden_size=64))
    indexer.eval()
    x = mx.random.normal((1, rows, 64))
    dense = indexer(x, None, offset=0, return_blocks=False)
    ids, valid = indexer(x, None, offset=0, return_blocks=True)
    native = qsa_prefill_direct_topk_buffer(ids, valid, pos_start=0, validate=True)
    assert native.shape == (1, 1, rows, 512)
    blocks = mx.put_along_axis(
        mx.zeros((rows, rows // 4), dtype=mx.bool_), ids, mx.array(True), axis=-1
    )
    selected = mx.repeat(blocks, 4, axis=-1)
    positions = mx.arange(rows)[None, :]
    queries = mx.arange(rows)[:, None]
    complete = (queries + 1) // 4
    candidate_mask = (selected | (positions >= complete * 4)) & (positions <= queries)
    reference_mask = (dense[0, 0] == 0) & (positions <= queries)
    assert bool(mx.array_equal(candidate_mask, reference_mask))
