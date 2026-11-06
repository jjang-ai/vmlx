# SPDX-License-Identifier: Apache-2.0
"""Ricursive (Ripple) runtime: mixed sliding/full cache, compacted rotating state, rollback, drafter window."""
from __future__ import annotations

import os
import types

# TF32 matmuls (the M5 default) round cached-vs-full comparisons to ~1e-3; with it off they agree to ~1e-6. A window or
# mask error is O(0.1) either way, so the tolerance below separates the two whether or not MLX saw this setting first.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.models.ricursive import register_ricursive_runtime  # noqa: E402

register_ricursive_runtime()
import mlx_lm.models.ricursive as R  # noqa: E402

WINDOW = 8


def _model(seed: int = 0):
    mx.random.seed(seed)
    args = R.ModelArgs(hidden_size=64, intermediate_size=128, num_hidden_layers=5, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=16, vocab_size=97, sliding_window=WINDOW, embed_scale=3.7)
    model = R.Model(args)
    mx.eval(model.parameters())
    return model


def _prefill(model, ids, cache, step):
    out = None
    for s in range(0, len(ids), step):
        out = model(mx.array([ids[s:s + step]]), cache=cache)
    return out[:, -1]


def _decode(model, cache, ids):
    rows = []
    for t in ids:
        rows.append(model(mx.array([[t]]), cache=cache)[:, -1])
    return mx.stack(rows)


def _clone(model, cache):
    fresh = model.make_cache()
    for dst, src in zip(fresh, cache):
        dst.state = src.state
        dst.meta_state = src.meta_state
    return fresh


# Lengths around the window, never only multiples of it (a misaligned restore is the hard shape).
@pytest.mark.parametrize("n,step", [(5, 3), (13, 5), (19, 7), (23, 23), (31, 4)])
def test_cached_prefill_matches_full_forward(n, step):
    model = _model()
    ids = [(i * 7 + 3) % 97 for i in range(n)]
    full = model(mx.array([ids]))[:, -1]
    cache = model.make_cache()
    got = _prefill(model, ids, cache, step)
    assert mx.allclose(full, got, atol=1e-2).item()


@pytest.mark.parametrize("n,step", [(13, 5), (19, 7), (21, 10), (30, 11), (27, 9)])
def test_compacted_state_restores_bit_exact(n, step):
    model = _model(1)
    ids = [(i * 11 + 5) % 97 for i in range(n)]
    cache = model.make_cache()
    _prefill(model, ids, cache, step)
    sliding = [c for c in cache if isinstance(c, R.RotatingKVCache)]
    assert sliding and all(c.offset == n for c in cache)
    # A multi-row last chunk left max_size + S - 1 rows; state carries only the newest max_size.
    if sliding[0].keys.shape[2] > WINDOW and sliding[0]._idx == sliding[0].keys.shape[2] and n >= WINDOW:
        assert sliding[0].state[0].shape[2] == WINDOW
        assert int(sliding[0].meta_state[3]) == WINDOW
    restored = _clone(model, cache)
    tail = [(i * 5 + 1) % 97 for i in range(6)]
    a = _decode(model, cache, tail)
    b = _decode(model, restored, tail)
    assert mx.array_equal(a, b).item()
    # A multi-row continuation after the restore reads the same window as well.
    cache2, restored2 = model.make_cache(), None
    _prefill(model, ids, cache2, step)
    restored2 = _clone(model, cache2)
    x = mx.array([tail])
    assert mx.array_equal(model(x, cache=cache2), model(x, cache=restored2)).item()


def test_compaction_actually_shrinks_a_wide_chunk():
    model = _model(3)
    cache = model.make_cache()
    _prefill(model, list(range(1, 22)), cache, 7)            # 21 rows: chunks 7/7/7, last buffer 7 + 7 = 14 rows
    layer = next(c for c in cache if isinstance(c, R.RotatingKVCache))
    assert layer.keys.shape[2] == WINDOW - 1 + 7
    assert layer.state[0].shape[2] == WINDOW and layer.meta_state[2:4] == ("21", str(WINDOW))


@pytest.mark.parametrize("n,rows,drop", [(12, 4, 3), (20, 6, 5), (9, 3, 1)])
def test_rollback_recent_keeps_the_window(n, rows, drop):
    model = _model(2)
    ids = [(i * 3 + 2) % 97 for i in range(n)]
    verify = [(i * 13 + 7) % 97 for i in range(rows)]
    keep = verify[: rows - drop]
    cache = model.make_cache()
    _prefill(model, ids, cache, 5)
    model(mx.array([verify]), cache=cache)
    for c in cache:
        if hasattr(c, "rollback_recent") and not c.is_trimmable():
            c.rollback_recent(drop)
        else:
            c.trim(drop)
    ref = model.make_cache()
    _prefill(model, ids + keep, ref, 5)
    nxt = [(i * 17 + 1) % 97 for i in range(4)]
    assert mx.allclose(_decode(model, cache, nxt), _decode(model, ref, nxt), atol=1e-2).item()


def test_draft_context_window_bounds_full_layers(monkeypatch):
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from vmlx_engine.dflash2_loader import DRAFT_CONTEXT_WINDOWS, apply_draft_context_window

    def draft(types_):
        d = types.SimpleNamespace(config=types.SimpleNamespace(layer_types=types_, sliding_window=4096))
        d.make_cache = lambda: [RotatingKVCache(max_size=4095) if t == "sliding_attention" else KVCache() for t in types_]
        return d

    monkeypatch.delenv("VMLX_DFLASH2_DRAFT_WINDOW", raising=False)
    mixed = draft(("sliding_attention",) * 4 + ("full_attention",))
    assert apply_draft_context_window(mixed, "ricursive") == DRAFT_CONTEXT_WINDOWS["ricursive"] == 16384
    caches = mixed.make_cache()
    assert [c.max_size for c in caches] == [4095] * 4 + [16383]
    assert mixed._vmlx_context_window == 16384
    # All-sliding drafters and unknown families are untouched.
    sliding = draft(("sliding_attention",) * 5)
    assert apply_draft_context_window(sliding, "ricursive") is None and not hasattr(sliding, "_vmlx_context_window")
    other = draft(("sliding_attention",) * 4 + ("full_attention",))
    assert apply_draft_context_window(other, "qwen3_5") is None
    # The env override; 0 restores the unbounded drafter; a window is never below the sliding window.
    monkeypatch.setenv("VMLX_DFLASH2_DRAFT_WINDOW", "0")
    assert apply_draft_context_window(draft(("sliding_attention", "full_attention")), "ricursive") is None
    monkeypatch.setenv("VMLX_DFLASH2_DRAFT_WINDOW", "1000")
    assert apply_draft_context_window(draft(("sliding_attention", "full_attention")), "ricursive") == 4096


def test_generic_speculative_loop_never_drives_a_dflash2_drafter(monkeypatch):
    import vmlx_engine.speculative as spec

    monkeypatch.setattr(spec, "is_speculative_enabled", lambda: True)
    monkeypatch.setattr(spec, "_spec_kind", "dflash2")
    assert spec.should_use_speculative() is False


def test_snapshot_shells_do_not_alias_a_rotating_buffer():
    """A clone taken past the window must not change when the live cache then takes single-row steps."""
    from mlx_lm.models.cache import RotatingKVCache

    from vmlx_engine.dflash2_runtime import _clone_cache_shells

    for cls in (RotatingKVCache, R.RotatingKVCache):
        live = cls(max_size=4, keep=0)
        live.update_and_fetch(mx.ones((1, 1, 6, 2)), mx.ones((1, 1, 6, 2)))
        live.update_and_fetch(mx.ones((1, 1, 1, 2)) * 2, mx.ones((1, 1, 1, 2)) * 2)   # rotated, past the window
        mx.eval(live.keys)
        (snap,) = _clone_cache_shells([live], lambda: [cls(max_size=4, keep=0)])
        before = mx.array(snap.keys)
        for v in (3.0, 4.0, 5.0):
            live.update_and_fetch(mx.ones((1, 1, 1, 2)) * v, mx.ones((1, 1, 1, 2)) * v)
        mx.eval(live.keys)
        assert mx.array_equal(snap.keys, before).item(), cls


@pytest.mark.parametrize("n,rows", [(5, 4), (29, 6), (45, 5), (70, 9)])
def test_verify_rows_equal_their_decode_steps(n, rows):
    """Inside a row-exact verify scope, row r of an L-row forward equals the decode step for the same token, bit for
    bit, past the window and at lengths that do not divide it (so temp-0 output does not depend on verify widths)."""
    from vmlx_engine.metal.row_exact_qmv import row_exact_verify_scope

    # Ripple's attention geometry (head_dim 128, GQA 10) in bf16: small toys never reach MLX's row-count kernel
    # switches (GEMV/GEMM, vector/full SDPA), and this test passed with the exact path disabled.
    mx.random.seed(4)
    model = R.Model(R.ModelArgs(hidden_size=512, intermediate_size=1024, num_hidden_layers=5, num_attention_heads=20,
                                num_key_value_heads=2, head_dim=128, vocab_size=512, sliding_window=WINDOW * 4,
                                embed_scale=3.7))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    ids = [(i * 13 + 7) % 512 for i in range(n)]
    nxt = [(i * 11 + 3) % 512 for i in range(rows)]
    a, b = model.make_cache(), model.make_cache()
    _prefill(model, ids, a, 7)
    _prefill(model, ids, b, 7)
    steps = _decode(model, a, nxt)                                      # (rows, 1, vocab)
    with row_exact_verify_scope(rows=rows, required=True):
        verify = model(mx.array([nxt]), cache=b)[0]                     # (rows, vocab)
    mx.eval(steps, verify)
    for r in range(rows):
        assert mx.array_equal(verify[r], steps[r, 0]).item(), r


@pytest.mark.parametrize("prefix_len,new_rows", [(30, 5), (19, 3), (30, 12)])
def test_ssd_turn_entry_stores_sliding_rows_as_a_delta_and_restores_exactly(tmp_path, prefix_len, new_rows):
    """Boundary entry, then a turn entry on it: the sliding windows go to SSD as only the rows added since the boundary
    (when the boundary's end is inside the window), and the restored cache decodes bit-equal to the live one."""
    from vmlx_engine.dflash2_runtime import _clone_cache_shells
    from vmlx_engine.dflash2_session_disk import DFlash2SessionSSD

    im_start, im_end = 90, 91
    model = _model(5)
    ssd = DFlash2SessionSSD(root=tmp_path, max_size_bytes=1 << 30, model_key="ricursive-unit")
    try:
        ids = [(i * 7 + 2) % 89 for i in range(prefix_len)]
        live = model.make_cache()
        _prefill(model, ids, live, 7)
        shells = _clone_cache_shells(live, model.make_cache)
        rc = ssd.put({"kind": "boundary", "tokens": ids, "cache_len": len(ids), "target_cache": shells})
        assert ssd.flush() and rc.result()["durable"]
        parent = dict(rc.vmlx_ssd_ref, receipt=rc)
        more = [(i * 5 + 1) % 89 for i in range(new_rows)]
        _decode(model, live, more)
        tokens = ids + more
        rc2 = ssd.put({"kind": "turn", "tokens": tokens, "cache_len": len(tokens), "target_cache": live,
                       "ssd_parent": parent})
        assert ssd.flush() and rc2.result()["durable"]
        import json

        from vmlx_engine.dflash2_session_disk import signature_for

        _arrays, meta = ssd.store.load(signature_for(tokens), lambda p: mx.load(str(p), return_metadata=True))
        layers = json.loads(meta["dflash2"])["target"]
        sliding = [layer for layer in layers if layer["cls"] == "RotatingKVCache"]
        # Rows inside the window go as a delta (only the new rows on disk); past it the window is stored whole.
        assert all(bool(layer.get("rot_delta")) == (new_rows < WINDOW) for layer in sliding)
        if new_rows < WINDOW:
            assert all(_arrays[f"t{i}.0"].shape[2] == new_rows for i, layer in enumerate(layers) if layer.get("rot_delta"))
        got = ssd.take_matching(tokens + [im_start, 3], im_start_id=im_start, eos_ids=[im_end],
                                make_target=model.make_cache, make_draft=lambda: [])
        assert got is not None and got["cache_len"] == len(tokens)
        restored = got["target_cache"]
        for a, b in zip(live, restored):
            assert a.offset == b.offset
            assert all(mx.array_equal(x, y).item() for x, y in zip(a.state, b.state))
        tail = [(i * 3 + 4) % 89 for i in range(5)]
        assert mx.array_equal(_decode(model, live, tail), _decode(model, restored, tail)).item()
    finally:
        ssd.store.close()
