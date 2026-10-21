"""The paired verify optimization must preserve the caller's mask contract."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def patched_attention(monkeypatch):
    from mlx_lm.models import nemotron_h
    from vmlx_engine.metal import row_exact_qmv
    from vmlx_engine.models import nemotron_mamba_rollback as rollback

    calls = []
    sentinel = object()

    def original(self, x, mask=None, cache=None):
        calls.append((x, mask, cache))
        return sentinel

    attention = nemotron_h.NemotronHAttention
    mixer = nemotron_h.NemotronHMamba2Mixer
    # apply() patches both classes. Register every mutation for fixture cleanup,
    # including the rollback module's process-wide installation flag.
    monkeypatch.setattr(rollback, "_patched", False)
    monkeypatch.setattr(attention, "__call__", original)
    monkeypatch.setattr(mixer, "__call__", mixer.__call__)
    for cls in (attention, mixer):
        monkeypatch.setattr(cls, "_vmlx_original_call", getattr(cls, "_vmlx_original_call", None), raising=False)
    monkeypatch.setattr(row_exact_qmv, "row_exact_scope_active", lambda: True)
    rollback.apply()
    return attention.__call__, calls, sentinel


@pytest.mark.parametrize("mask", [
    None,
    np.ones((1, 1, 4, 8), dtype=np.bool_),
    np.zeros((1, 1, 4, 8), dtype=np.float32),
    np.zeros((1, 1, 1, 8), dtype=np.float32),
    "custom-mask",
])
def test_noncausal_mask_uses_original_contract(patched_attention, mask):
    call, calls, sentinel = patched_attention
    x, cache = SimpleNamespace(shape=(1, 4, 8)), object()
    # No projections or GPU work should happen before delegating the mask.
    assert call(object(), x, mask=mask, cache=cache) is sentinel
    assert calls == [(x, mask, cache)]


def test_causal_verify_keeps_paired_fast_path(patched_attention):
    call, calls, _ = patched_attention

    class PairedPathReached(Exception):
        pass

    def q_proj(x):
        raise PairedPathReached

    with pytest.raises(PairedPathReached):
        call(SimpleNamespace(q_proj=q_proj), SimpleNamespace(shape=(1, 4, 8)),
             mask="causal", cache=object())
    assert calls == []


def test_depth_one_verify_matches_decode_past_gqa_kernel_boundary(monkeypatch):
    """Exercise the model wrapper, not just the SDPA helper it can bypass."""
    import mlx.core as mx
    from mlx_lm.models import nemotron_h
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.metal import row_exact_qmv
    from vmlx_engine.models import nemotron_mamba_rollback as rollback

    attention, mixer = nemotron_h.NemotronHAttention, nemotron_h.NemotronHMamba2Mixer
    original = getattr(attention, "_vmlx_original_call", attention.__call__)
    monkeypatch.setattr(rollback, "_patched", False)
    monkeypatch.setattr(attention, "__call__", original)
    monkeypatch.setattr(mixer, "__call__", mixer.__call__)
    for cls in (attention, mixer):
        monkeypatch.setattr(cls, "_vmlx_original_call",
                            getattr(cls, "_vmlx_original_call", None), raising=False)
    monkeypatch.setattr(row_exact_qmv, "row_exact_scope_active", lambda: True)
    rollback.apply()

    # Identity/slice projections isolate the actual attention dispatch from QMM.
    layer = SimpleNamespace(
        num_heads=32, num_key_value_heads=2, scale=128 ** -0.5,
        q_proj=lambda x: x, k_proj=lambda x: x[..., :256],
        v_proj=lambda x: x[..., 256:512], o_proj=lambda x: x,
    )
    mx.random.seed(9001)
    keys = mx.random.normal((1, 2, 9000, 128)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, 9000, 128)).astype(mx.bfloat16)
    x = mx.random.normal((1, 2, 4096)).astype(mx.bfloat16)
    verify_cache, decode_cache = KVCache(), KVCache()
    verify_cache.update_and_fetch(keys, values)
    decode_cache.update_and_fetch(keys, values)
    got = attention.__call__(layer, x, mask="causal", cache=verify_cache)
    expected = mx.concatenate([
        original(layer, x[:, r:r + 1], mask=None, cache=decode_cache)
        for r in range(2)
    ], axis=1)
    mx.eval(got, expected)
    assert mx.array_equal(got, expected).item()
