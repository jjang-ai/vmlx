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
