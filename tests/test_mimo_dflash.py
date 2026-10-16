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
    # the prompt's head on its last row only, as the lane's prefill does (a full-chunk head can break a bf16 tie
    # differently on this tiny random model)
    cache = model.make_cache()
    lg = model.lm_head(model.model(mx.array([prompt]), cache)[:, -1:])[0, -1]
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


def _run(tgt, drafter, prompt, n, **kw):
    return [t for new, _ in MD.mimo_rounds(tgt, drafter, None, prompt, n, set(), 0.0, adaptive=False, **kw) for t in new]


def test_random_drafts_equal_ar_and_turn_state_resumes(tmp_path):
    """The end-of-turn session state (cache + drafter taps) resumes the next turn: greedy output == AR."""
    model, tgt, drafter = _lane(tmp_path)
    prompt = np.random.default_rng(6).integers(0, V, size=40).tolist()
    session = {}
    out = _run(tgt, drafter, prompt, 20, session=session)
    assert out == _ar_greedy(model, prompt, 20)
    # the cache holds the prompt + every emitted token but the last (sampled, not forwarded)
    assert session["forwarded"] == len(prompt) + len(out) - 1
    assert int(session["taps"].shape[0]) == min(drafter.window, session["forwarded"])
    prompt2 = prompt + out + [5, 9]
    st = MD.LaneStats()
    resume = {"target_cache": session["target_cache"], "draft_context": session["taps"], "cache_len": session["forwarded"]}
    out2 = _run(tgt, drafter, prompt2, 6, stats=st, resume=resume)
    assert st.cached_tokens == session["forwarded"]
    assert out2 == _ar_greedy(model, prompt2, 6)
    tgt.close()


@pytest.mark.parametrize("max_tokens", [3, 4, 7])
def test_round_cut_short_by_max_tokens_keeps_only_emitted_rows(tmp_path, max_tokens):
    """Oracle drafts (5 accepted per round) cut by max_tokens mid-round: the stored state must hold exactly the
    emitted tokens, so a resumed turn still equals AR (the ported lane kept rows of tokens never emitted)."""
    model, tgt, drafter = _lane(tmp_path)
    prompt = np.random.default_rng(8).integers(0, V, size=30).tolist()
    ref = _ar_greedy(model, prompt, 40)
    oracle = lambda n_emitted, k: ref[n_emitted: n_emitted + k] + [0] * max(0, n_emitted + k - len(ref))
    session = {}
    out = [t for new, _ in MD.mimo_rounds(tgt, drafter, None, prompt, max_tokens, set(), 0.0, adaptive=False,
                                          propose=oracle, session=session) for t in new]
    assert out == ref[:max_tokens]
    assert session["forwarded"] == len(prompt) + max_tokens - 1
    prompt2 = prompt + out
    resume = {"target_cache": session["target_cache"], "draft_context": session["taps"], "cache_len": session["forwarded"]}
    # resuming re-forwards only the last emitted token (a 1-row step, as AR decoding does), so it continues the AR
    # sequence itself (a one-shot prefill of prompt2 can flip near-tie argmaxes on this tiny random model)
    assert _run(tgt, drafter, prompt2, 5, resume=resume) == ref[max_tokens: max_tokens + 5]
    tgt.close()


def test_snapshot_clone_isolated_from_later_rotating_writes(tmp_path):
    """A boundary snapshot of the SWA RotatingKVCache must not change while the request keeps decoding (single-token
    rotating updates write into the live buffer in place); resuming it equals AR."""
    from vmlx_engine import mimo_session

    model, tgt, drafter = _lane(tmp_path)
    prompt = np.random.default_rng(9).integers(0, V, size=60).tolist()   # longer than the SWA window: wrapped
    snaps = []
    on_snap = lambda cut, cache, taps: snaps.append((cut, mimo_session.clone(cache, tgt.model.make_cache), taps))
    out = _run(tgt, drafter, prompt, 12, snapshot_cuts=[37], on_snapshot=on_snap)
    assert out == _ar_greedy(model, prompt, 12)
    (cut, snap, taps), = snaps
    from mlx_lm.models.cache import RotatingKVCache
    rings = [c for c in snap if isinstance(c, RotatingKVCache)]
    assert rings and all(int(c.keys.shape[2]) <= int(c.max_size) for c in rings)   # dead concat rows trimmed
    before = [np.array(c.keys.astype(mx.float32)) for c in snap if c.keys is not None]
    _run(tgt, drafter, prompt + out, 10)                   # unrelated decoding on a fresh cache must not touch it
    resume = {"target_cache": snap, "draft_context": taps, "cache_len": cut}
    assert all(np.array_equal(a, np.array(c.keys.astype(mx.float32))) for a, c in zip(before, [c for c in snap if c.keys is not None]))
    assert _run(tgt, drafter, prompt, 8, resume=resume) == _ar_greedy(model, prompt, 8)
    tgt.close()


def test_session_ssd_round_trip_resumes_equal_to_ar(tmp_path, monkeypatch):
    """Store a snapshot through mimo_session with an SSD tier, drop RAM, restore it from SSD: output == AR."""
    from vmlx_engine import mimo_session
    from vmlx_engine.dflash2_session_disk import DFlash2SessionSSD

    model, tgt, drafter = _lane(tmp_path)
    E = V - 1                                               # an "end of turn" id: the SSD cut finder keys on it
    prompt = np.random.default_rng(10).integers(0, V - 2, size=50).tolist()
    prompt[29] = E
    monkeypatch.setattr(mimo_session, "STORE", None)
    monkeypatch.setattr(mimo_session, "SSD", DFlash2SessionSSD(root=str(tmp_path / "ssd"), max_size_bytes=0, model_key="t"))
    snaps = []
    _run(tgt, drafter, prompt, 6, snapshot_cuts=[30],
         on_snapshot=lambda cut, cache, taps: snaps.append((cut, mimo_session.clone(cache, tgt.model.make_cache), taps)))
    (cut, snap, taps), = snaps
    receipt = mimo_session.store({"kind": "system", "tokens": prompt[:cut], "cache_len": cut, "target_cache": snap,
                                  "draft_context": taps, "model_key": "k"})
    assert receipt.result(timeout=30)["durable"]
    mimo_session.clear()                                    # RAM tier gone: the restore comes from SSD
    tok = type("T", (), {"convert_tokens_to_ids": lambda self, t: -1})()
    resume = mimo_session.lookup("k", prompt, tok, tgt.model.make_cache, eos_ids=[E])
    assert resume is not None and resume["source"] == "mimo-dflash-ssd" and resume["cache_len"] == cut
    assert _run(tgt, drafter, prompt, 8, resume=resume) == _ar_greedy(model, prompt, 8)
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


def test_adaptive_ar_bursts_defer_drafter_context_and_stay_equal_to_ar(tmp_path):
    """With the adaptive policy (AR bursts + draft rounds), deferred drafter context still yields AR-equal greedy output
    and a session state aligned with the emitted tokens."""
    model, tgt, drafter = _lane(tmp_path)
    prompt = np.random.default_rng(12).integers(0, V, size=35).tolist()
    session = {}
    out = [t for new, _ in MD.mimo_rounds(tgt, drafter, None, prompt, 30, set(), 0.0, adaptive=True, session=session)
           for t in new]
    ref = _ar_greedy(model, prompt, 30)
    assert out == ref
    assert session["forwarded"] == len(prompt) + len(out) - 1
    assert int(session["taps"].shape[0]) == min(drafter.window, session["forwarded"])
    resume = {"target_cache": session["target_cache"], "draft_context": session["taps"], "cache_len": session["forwarded"]}
    assert _run(tgt, drafter, prompt + out, 5, resume=resume) == _ar_greedy(model, prompt, 35)[30:35]
    tgt.close()


def test_pipelined_ar_burst_eos_keeps_eos_row_and_resumes(tmp_path):
    """A pipelined AR burst that samples EOS has already launched the next step with the EOS as input: the cache then
    holds prompt + every emitted token (EOS included), flagged eos_forwarded, and that state resumes the next turn."""
    model, tgt, drafter = _lane(tmp_path)
    prompt = np.random.default_rng(14).integers(0, V, size=33).tolist()
    ref = _ar_greedy(model, prompt, 12)
    eos = {ref[3]}                                          # inside the first AR burst (the policy starts with AR)
    session, st = {}, MD.LaneStats()
    out = [t for new, _ in MD.mimo_rounds(tgt, drafter, None, prompt, 12, eos, 0.0, adaptive=True, session=session,
                                          stats=st) for t in new]
    assert out == ref[:4] and out[-1] in eos
    assert st.eos_forwarded and session["forwarded"] == len(prompt) + len(out)
    assert int(session["target_cache"][0].offset) == session["forwarded"]
    assert int(session["taps"].shape[0]) == min(drafter.window, session["forwarded"])
    tgt.close()


def test_prefill_stops_at_a_chunk_boundary_when_the_request_is_cancelled(tmp_path):
    """SimpleEngine's cancellation scope (client gone) must stop the MiMo lane's prompt prefill at the next chunk
    boundary instead of running every chunk (a dropped 30k prompt held the engine 106-108 s, 2026-10-16), and the lane
    must then serve the next request normally (output == AR)."""
    from vmlx_engine.utils.generation_cancellation import GenerationCancelled, generation_cancellation_scope

    model, tgt, drafter = _lane(tmp_path)
    prompt = np.random.default_rng(11).integers(0, V, size=40).tolist()
    forwards = {"n": 0}
    real_forward = tgt.forward

    def counting_forward(*a, **k):
        forwards["n"] += 1
        return real_forward(*a, **k)

    tgt.forward = counting_forward
    with pytest.raises(GenerationCancelled):
        with generation_cancellation_scope(lambda: forwards["n"] >= 2):
            _run(tgt, drafter, prompt, 10, prefill_step=8)       # 5 chunks of 8: cancel after the second
    assert forwards["n"] == 2
    tgt.forward = real_forward
    assert _run(tgt, drafter, prompt, 10) == _ar_greedy(model, prompt, 10)
    tgt.close()
