"""Same-shape GLM DSA materialization: no selection or precision change."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.models.glm5_next.glm5_next import MLAAttention


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("rows,rank,selected,tile", [(9, 16, 7, 3), (33, 512, 129, 4)])
def test_materialization_is_exact_with_same_tiles_and_causal_selection(monkeypatch, dtype, batch, rows, rank, selected, tile):
    rng = np.random.default_rng(314)
    past, heads = 7, 4
    queries = mx.array(rng.normal(size=(batch, heads, rows, rank)).astype(np.float32)).astype(dtype)
    latent = mx.array(rng.normal(size=(batch, 1, past + rows, rank)).astype(np.float32)).astype(dtype)
    chosen = rng.integers(0, past + rows, size=(batch, rows, selected), dtype=np.int32)
    chosen[:, :, 0] = 0
    mask = rng.random(size=chosen.shape) > 0.1
    mask[:, :, 0] = True
    indices, valid = mx.array(chosen), mx.array(mask)
    owner = SimpleNamespace(gather_element_budget=tile * selected * rank, scale=rank**-0.5)
    mx.eval(queries, latent, indices, valid)
    latent_before = mx.array(latent)
    monkeypatch.delenv("VMLX_GLM5_DSA_PREFILL_MATERIALIZE", raising=False)
    baseline = MLAAttention._gather_absorbed_attention(owner, queries, latent, indices, valid, past=past)
    mx.eval(baseline)

    calls = []
    real_eval = mx.eval

    def observed_eval(*arrays):
        calls.append(tuple(arrays[0].shape))
        return real_eval(*arrays)

    monkeypatch.setenv("VMLX_GLM5_DSA_PREFILL_MATERIALIZE", "1")
    monkeypatch.setattr(mx, "eval", observed_eval)
    candidate = MLAAttention._gather_absorbed_attention(owner, queries, latent, indices, valid, past=past)
    real_eval(candidate)
    assert len(calls) == (rows + tile - 1) // tile
    assert all(shape[0] <= batch * tile and shape[-2:] == (heads, rank) for shape in calls)
    assert bool(mx.array_equal(baseline, candidate).item())
    assert bool(mx.array_equal(latent, latent_before).item())
    assert bool(mx.array_equal(indices, mx.array(chosen)).item())


@pytest.mark.parametrize("enabled,rows", [(False, 9), (True, 1), (True, 3)])
def test_off_decode_and_single_tile_do_not_add_a_fence(monkeypatch, enabled, rows):
    monkeypatch.setenv("VMLX_GLM5_DSA_PREFILL_MATERIALIZE", "1" if enabled else "0")
    owner = SimpleNamespace(gather_element_budget=3 * 4 * 16, scale=0.25)
    q, latent = mx.ones((1, 2, rows, 16)), mx.ones((1, 1, rows + 3, 16))
    indices, valid = mx.zeros((1, rows, 4), mx.int32), mx.ones((1, rows, 4), mx.bool_)
    real_eval = mx.eval
    calls = []
    monkeypatch.setattr(mx, "eval", lambda *arrays: calls.append(arrays))
    result = MLAAttention._gather_absorbed_attention(owner, q, latent, indices, valid, past=3)
    assert not calls
    real_eval(result)
    assert result.shape == (1, 2, rows, 16)
