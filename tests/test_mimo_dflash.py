"""MiMo-V2.6 DFlash lane mechanics on a tiny synthetic JANGHT model (no real weights).

* verify rollback: a multi-row verify followed by Target.rollback(r) must leave every cache (full KVCache and the SWA
  RotatingKVCache, wrapped: prompt longer than the window) equivalent to forwarding only the committed tokens -- the
  next step's logits must match;
* rollback fails closed on a rotating cache that is not in concat order;
* the drafter: context window trimming, finite proposals of the configured block width, taps of the right width.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.mimo_v2.v26_model")

from tests.test_mimo_v26_janght_loader import _bundle, D, V  # noqa: E402
from vmlx_engine import mimo_dflash as MD  # noqa: E402
from vmlx_engine.models.mimo_v26_janght import build_mimo_v26_janght_model  # noqa: E402


def _rel(a, b):
    a = a.astype(mx.float32); b = b.astype(mx.float32)
    return float((mx.linalg.norm(a - b) / mx.linalg.norm(b)).item())


@pytest.mark.parametrize("r", [0, 1, 5, 7])
def test_verify_rollback_equals_committed_prefix(tmp_path, r):
    _bundle(tmp_path)
    model = build_mimo_v26_janght_model(tmp_path)
    tgt = MD.Target(model, [0, 1])
    rng = np.random.default_rng(2)
    prompt = rng.integers(0, V, size=22).tolist()                  # > sliding window 16: rotating caches wrapped
    block = rng.integers(0, V, size=8).tolist()
    nxt = int(rng.integers(0, V))
    ca = model.make_cache(); tgt.forward(prompt, ca); tgt.forward(block, ca); MD.Target.rollback(ca, r)
    kept = block[: 8 - r]
    cb = model.make_cache(); tgt.forward(prompt, cb)
    for t in kept:                                                  # committed tokens one at a time (decode steps)
        tgt.forward([t], cb)
    la, _ = tgt.forward([nxt], ca)
    lb, _ = tgt.forward([nxt], cb)
    assert _rel(la[-1], lb[-1]) < 3e-2
    for c1, c2 in zip(ca, cb):
        assert c1.offset == c2.offset
    tgt.close()


def test_rollback_refuses_rotated_cache(tmp_path):
    _bundle(tmp_path)
    model = build_mimo_v26_janght_model(tmp_path)
    tgt = MD.Target(model, [0, 1])
    c = model.make_cache()
    tgt.forward(list(range(20)), c)
    for t in range(5):
        tgt.forward([t], c)                                          # single-row updates rotate in place
    with pytest.raises(RuntimeError):
        MD.Target.rollback(c, 2)
    tgt.close()


def test_drafter_shapes_and_window():
    c = {"hidden_size": D, "intermediate_size": 512, "num_hidden_layers": 2, "num_attention_heads": 8, "num_key_value_heads": 2,
         "head_dim": 128, "partial_rotary_factor": 0.5, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "sliding_window": 32,
         "block_size": 8, "dflash_config": {"target_layer_ids": [0, 1], "mask_token_id": 7, "block_size": 8, "attention_value_scale": 0.612}}
    m = MD.MiMoDFlashDrafter(c)
    mx.eval(m.parameters())
    taps = mx.random.normal((40, 2 * D)).astype(mx.bfloat16)
    m.add_context(taps, 0)
    assert m.kv[0][0].shape[2] == 32 and m.ctx_end == 40
    m.add_context(taps[:3], 40)
    assert m.kv[0][0].shape[2] == 32 and m.ctx_end == 43
    h = m.draft(mx.random.normal((1, 8, D)).astype(mx.bfloat16), 43)
    assert h.shape == (1, 8, D) and bool(mx.all(mx.isfinite(h.astype(mx.float32))).item())
    with pytest.raises(AssertionError):
        m.draft(mx.random.normal((1, 8, D)).astype(mx.bfloat16), 44)


def _lane(tmp_path):
    _bundle(tmp_path)
    model = build_mimo_v26_janght_model(tmp_path)
    c = {"hidden_size": D, "intermediate_size": 512, "num_hidden_layers": 1, "num_attention_heads": 8, "num_key_value_heads": 2,
         "head_dim": 128, "partial_rotary_factor": 0.5, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "sliding_window": 64,
         "block_size": 6, "dflash_config": {"target_layer_ids": [0, 1], "mask_token_id": 7, "block_size": 6, "attention_value_scale": 0.612}}
    drafter = MD.MiMoDFlashDrafter(c); mx.eval(drafter.parameters())
    return model, MD.Target(model, [0, 1]), drafter


def _ar_greedy(model, prompt, n):
    cache = model.make_cache()
    lg = model(mx.array([prompt]), cache=cache)[0, -1]
    out = [int(mx.argmax(lg).item())]
    while len(out) < n:
        lg = model(mx.array([[out[-1]]]), cache=cache)[0, -1]; out.append(int(mx.argmax(lg).item()))
    return out


def test_oracle_drafts_fully_accepted_and_equal_ar(tmp_path):
    model, tgt, drafter = _lane(tmp_path)
    prompt = np.random.default_rng(4).integers(0, V, size=30).tolist()
    ref = _ar_greedy(model, prompt, 25)
    oracle = lambda n_emitted, k: ref[n_emitted: n_emitted + k] + [0] * max(0, n_emitted + k - len(ref))
    st = MD.LaneStats(); out = []
    for new, _ in MD.mimo_rounds(tgt, drafter, None, prompt, 25, set(), 0.0, stats=st, adaptive=False, propose=oracle):
        out.extend(new)
    assert out == ref
    assert st.rounds > 0 and all(a == 5 for a in st.accepted[:-1]), st.accepted
    tgt.close()


def test_random_drafts_still_equal_ar_and_prefix_store(tmp_path):
    model, tgt, drafter = _lane(tmp_path)
    prompt = np.random.default_rng(6).integers(0, V, size=40).tolist()
    ref = _ar_greedy(model, prompt, 20)
    store = MD.MiMoPrefixStore(2)
    out = [t for new, _ in MD.mimo_rounds(tgt, drafter, None, prompt, 20, set(), 0.0, store=store, adaptive=False) for t in new]
    assert out == ref
    st = MD.LaneStats()
    out2 = [t for new, _ in MD.mimo_rounds(tgt, drafter, None, prompt + [5, 9], 5, set(), 0.0, stats=st, store=store, adaptive=False) for t in new]
    assert st.cached_tokens == len(prompt)
    assert out2 == _ar_greedy(model, prompt + [5, 9], 5)
    tgt.close()


_THREAD_SCRIPT = r'''
import json, sys, threading
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from vmlx_engine import mimo_dflash as MD
d = Path(sys.argv[1]); D = int(sys.argv[2])
c = {"hidden_size": D, "intermediate_size": 512, "num_hidden_layers": 1, "num_attention_heads": 8, "num_key_value_heads": 2,
     "head_dim": 128, "partial_rotary_factor": 0.5, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "sliding_window": 64,
     "block_size": 6, "dflash_config": {"target_layer_ids": [0, 1], "mask_token_id": 7, "block_size": 6, "attention_value_scale": 0.612}}
m = MD.MiMoDFlashDrafter(c); mx.eval(m.parameters())
(d / "config.json").write_text(json.dumps(c))
mx.save_safetensors(str(d / "dflash_draft_model.safetensors"), dict(tree_flatten(m.parameters())))
mx.save_safetensors(str(d / "mask_embedding.safetensors"), {"embedding": mx.ones((D,), dtype=mx.bfloat16),
                                                             "mask_token_id": mx.array([7], dtype=mx.int32)})
drafter, mask_vec = MD.load_drafter(d)          # loader thread (as the server's model load)
out = {}
def gen():                                     # the server generates on another thread with its own streams
    s = mx.new_stream(mx.gpu)
    with mx.stream(s):
        out["mask"] = float((mask_vec.astype(mx.float32) * 2).sum().item())
        taps = mx.zeros((4, 2 * D), dtype=mx.bfloat16); drafter.add_context(taps, 0)
        h = drafter.draft(mx.zeros((1, 6, D), dtype=mx.bfloat16), 4); mx.eval(h); out["ok"] = True
t = threading.Thread(target=gen); t.start(); t.join()
assert out.get("ok") and out["mask"] == 2.0 * D, out
print("THREAD OK")
'''


def test_drafter_loaded_on_one_thread_runs_on_another(tmp_path):
    """Served lanes load on the loader thread and generate on a worker thread. Every array load_drafter returns must be
    materialized at load: a lazy mx.load array evaluated later from another thread aborts the process ("There is no
    Stream(cpu, 1) in current thread", 2026-10-15 MiMo JANGHT serve)."""
    import subprocess, sys
    r = subprocess.run([sys.executable, "-c", _THREAD_SCRIPT, str(tmp_path), str(D)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "THREAD OK" in r.stdout, (r.returncode, r.stdout[-500:], r.stderr[-1500:])
