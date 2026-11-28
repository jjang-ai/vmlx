"""Unit tests for ERNIE 5.5 runtime/parsers/cache/kernels, Inkling 2 fused looping kernels, and ARFallback outlier guard."""

from __future__ import annotations

import math
import tempfile

import mlx.core as mx
import numpy as np
import pytest


def test_ernie_reasoning_parser_extract_and_stream():
    from vmlx_engine.reasoning import get_parser

    cls = get_parser("ernie")
    assert cls is not None
    parser = cls()
    reasoning, content = parser.extract_reasoning(
        "<ernie:think>Step 1: 17 * 23 = 391.</ernie:think>The answer is 391."
    )
    assert reasoning == "Step 1: 17 * 23 = 391."
    assert content == "The answer is 391."

    # Streaming chunks split mid-tag
    p_stream = cls()
    chunks = ["<ernie:", "think>", "Plan ", "done", "</ernie:", "think>", "Final ", "text."]
    prev = ""
    r_acc, c_acc = [], []
    for ch in chunks:
        cur = prev + ch
        res = p_stream.extract_reasoning_streaming(prev, cur, ch)
        if res and res.reasoning:
            r_acc.append(res.reasoning)
        if res and res.content:
            c_acc.append(res.content)
        prev = cur
    assert "".join(r_acc) == "Plan done"
    assert "".join(c_acc) == "Final text."

    # Prompt-injected <ernie:think> (end-only reasoning in model output)
    p_implicit = cls()
    p_implicit.reset_state(think_in_prompt=True)
    r_imp, c_imp = p_implicit.extract_reasoning(
        "Compute 17 * 23 = 391.</ernie:think>The answer is 391."
    )
    assert r_imp == "Compute 17 * 23 = 391."
    assert c_imp == "The answer is 391."

    p_imp_stream = cls()
    p_imp_stream.reset_state(think_in_prompt=True)
    imp_chunks = ["Compute ", "17*23", "</ernie:", "think>", "The answer ", "is 391."]
    prev = ""
    r_acc, c_acc = [], []
    for ch in imp_chunks:
        cur = prev + ch
        res = p_imp_stream.extract_reasoning_streaming(prev, cur, ch)
        if res and res.reasoning:
            r_acc.append(res.reasoning)
        if res and res.content:
            c_acc.append(res.content)
        prev = cur
    assert "".join(r_acc) == "Compute 17*23"
    assert "".join(c_acc) == "The answer is 391."

    # Direct rail (enable_thinking=False -> closed sentinel in prompt)
    p_direct = cls()
    p_direct.reset_state(think_in_prompt=False)
    res_direct = p_direct.extract_reasoning_streaming("", "Tokyo is sunny.", "Tokyo is sunny.")
    assert res_direct is not None
    assert res_direct.reasoning is None
    assert res_direct.content == "Tokyo is sunny."


def test_ernie_prompt_reasoning_seed_and_template_kwargs():
    from vmlx_engine.server import _rendered_prompt_starts_in_reasoning
    from vmlx_engine.utils.chat_template_kwargs import build_chat_template_kwargs

    open_prompt = "<|ernie:im_start|>user\n__test__<|ernie:im_end|>\n<|ernie:im_start|>assistant\n<ernie:think>"
    closed_prompt = "<|ernie:im_start|>user\n__test__<|ernie:im_end|>\n<|ernie:im_start|>assistant\n<ernie:think></ernie:think>"
    assert _rendered_prompt_starts_in_reasoning(open_prompt) is True
    assert _rendered_prompt_starts_in_reasoning(closed_prompt) is False

    # ERNIE's native Python renderer rejects the generic `thinking` alias;
    # build_chat_template_kwargs must omit it so tools are never stripped on retry.
    kw = build_chat_template_kwargs(
        enable_thinking=True,
        tokenize=False,
        add_generation_prompt=True,
        model_type="ernie5_5_moe",
    )
    assert kw["enable_thinking"] is True
    assert "thinking" not in kw

    from vmlx_engine.server import _TOOL_CALL_MARKERS
    from vmlx_engine.models.ernie5_5_moe.cache import ErnieKVCache, ErnieLayerCache
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    assert "<ernie:tool_call>" in _TOOL_CALL_MARKERS
    assert "</ernie:tool_call>" in _TOOL_CALL_MARKERS
    assert SingleBatchGenerator._cache_uses_ernie55([ErnieLayerCache(ErnieKVCache(), ErnieKVCache(), is_mla=True)]) is True


def test_ernie_tool_parser_single_and_multi_call():
    from vmlx_engine.tool_parsers import ToolParserManager

    cls = ToolParserManager.get_tool_parser("ernie")
    parser = cls()
    raw = (
        "<ernie:think>Need both cities.</ernie:think>"
        '<ernie:tool_call>\n'
        '<invoke name="get_weather"><parameter name="city">Tokyo</parameter></invoke>\n'
        '<invoke name="get_weather"><parameter name="city">Paris</parameter></invoke>\n'
        '</ernie:tool_call>'
    )
    info = parser.extract_tool_calls(raw)
    assert info.tools_called is True
    assert len(info.tool_calls) == 2
    assert info.tool_calls[0]["name"] == "get_weather"
    assert '"Tokyo"' in info.tool_calls[0]["arguments"]
    assert info.tool_calls[1]["name"] == "get_weather"
    assert '"Paris"' in info.tool_calls[1]["arguments"]

    # End-only reasoning before tool call (prompt-injected <ernie:think>)
    end_only = (
        "Let me check weather.</ernie:think>"
        '<ernie:tool_call><invoke name="get_weather"><parameter name="city">Tokyo</parameter></invoke></ernie:tool_call>'
    )
    info2 = parser.extract_tool_calls(end_only)
    assert info2.tools_called is True
    assert len(info2.tool_calls) == 1
    assert info2.tool_calls[0]["name"] == "get_weather"
    assert info2.content is None

    # Plain response with no tool call
    plain = parser.extract_tool_calls("Tokyo is 22C and sunny.")
    assert plain.tools_called is False
    assert plain.tool_calls == []


def test_ernie_layer_cache_ssd_roundtrip_and_copy():
    from vmlx_engine.disk_cache import DiskCacheManager
    from vmlx_engine.models.ernie5_5_moe.cache import (
        ErnieKVCache,
        ErnieLayerCache,
        ErniePoolingCache,
        ErnieRotatingCache,
        register_ernie_cache_class,
    )

    register_ernie_cache_class()
    mx.random.seed(7)

    # Build one HCA cache (past 128-token boundary so local + pooled + pending are all populated)
    # and one MLA cache (kv_latent + k_pe + idx_k).
    rot = ErnieRotatingCache(max_size=128)
    pool = ErniePoolingCache(compress_ratio=128)
    rot.update_and_fetch(mx.random.normal((1, 4, 150, 128)).astype(mx.bfloat16))
    win_kv, win_gate, _ = pool.accumulate_windows(
        mx.random.normal((1, 150, 256)).astype(mx.float32),
        mx.random.normal((1, 150, 256)).astype(mx.float32),
        offset=0,
    )
    assert win_kv.shape == (1, 128, 256)
    pool.update_and_fetch(mx.random.normal((1, 1, 256)).astype(mx.bfloat16))
    hca = ErnieLayerCache(rot, pool, is_mla=False)

    lat = ErnieKVCache()
    idx = ErnieKVCache()
    lat.update_and_fetch(
        mx.random.normal((1, 1, 150, 512)).astype(mx.bfloat16),
        mx.random.normal((1, 1, 150, 64)).astype(mx.bfloat16),
    )
    idx.update_and_fetch(
        mx.random.normal((1, 1, 150, 128)).astype(mx.bfloat16),
        mx.zeros((1, 1, 150, 0), dtype=mx.bfloat16),
    )
    mla = ErnieLayerCache(lat, idx, is_mla=True)

    caches = [hca, mla]
    assert hca.offset == 150 and mla.offset == 150
    assert hca.nbytes > 0 and mla.nbytes > 0

    # Deep copy isolation check
    cloned = [c.copy() for c in caches]
    assert cloned[0].offset == 150 and cloned[1].offset == 150
    cloned[0][0].update_and_fetch(mx.random.normal((1, 4, 2, 128)).astype(mx.bfloat16))
    assert cloned[0].offset == 152 and hca.offset == 150

    # Full DiskCacheManager SSD round-trip
    tokens = list(range(1000, 1150))
    with tempfile.TemporaryDirectory() as td:
        dc = DiskCacheManager(
            cache_dir=td,
            max_size_gb=1.0,
            required_cache_class="ErnieLayerCache",
            expected_num_layers=2,
        )
        assert dc.store(tokens, caches)
        assert dc.flush_pending_writes(tokens)
        loaded = dc.fetch(tokens)
        assert loaded is not None and len(loaded) == 2
        assert isinstance(loaded[0], ErnieLayerCache) and not loaded[0].is_mla
        assert isinstance(loaded[1], ErnieLayerCache) and loaded[1].is_mla
        assert loaded[0].offset == 150 and loaded[1].offset == 150
        for orig_arr, load_arr in zip(hca.state, loaded[0].state):
            mx.eval(orig_arr, load_arr)
            assert float(mx.max(mx.abs(orig_arr.astype(mx.float32) - load_arr.astype(mx.float32))).item()) == 0.0
        for orig_arr, load_arr in zip(mla.state, loaded[1].state):
            mx.eval(orig_arr, load_arr)
            assert float(mx.max(mx.abs(orig_arr.astype(mx.float32) - load_arr.astype(mx.float32))).item()) == 0.0
        dc.shutdown()


def test_ar_fallback_cold_compile_outlier_guard():
    from vmlx_engine.dspark.generate import ARFallback

    fb = ARFallback()
    fb.record_ar(0.025)
    assert fb.ar_seconds == pytest.approx(0.025)

    # First-seen verify width with a 500ms Metal shader compile spike must NOT poison draft_rate
    fb.record_draft(4, 0.500, rows=6)
    assert fb.draft_rate is None

    # Normal winning round at rows=6 establishes a healthy draft_rate
    fb.record_draft(4, 0.040, rows=6)
    assert fb.draft_rate is not None
    assert fb.draft_rate * fb.ar_seconds > 1.0


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU required")
def test_inkling2_fused_looping_kernels_fp64_parity():
    from vmlx_engine.models.inkling2 import kernels as K
    from vmlx_engine.models.inkling2.language import l2_clip

    mx.random.seed(123)

    # 1. attention_residual_fused (Q=1 and Q=3) vs FP64 NumPy
    for Q in (1, 3):
        blocks = [mx.random.normal((1, 4, 2304)).astype(mx.bfloat16) for _ in range(3)]
        bbuf = mx.stack(blocks, axis=-2)
        partial = mx.random.normal((1, 4, 2304)).astype(mx.float32)
        query = mx.random.normal((Q, 2304)).astype(mx.bfloat16)
        got = K.attention_residual_fused(bbuf, partial, query, True)
        mx.eval(got)
        st64 = np.array(mx.stack([*blocks, partial], axis=-2).astype(mx.float32), dtype=np.float64)
        q64 = np.array(query.astype(mx.float32), dtype=np.float64)
        inv_rms = 1.0 / np.sqrt(np.mean(st64 * st64, axis=-1, keepdims=True) + 1e-6)
        keys64 = st64 * inv_rms
        sc64 = np.matmul(keys64, q64.T) * (2304.0 ** -0.5)
        sc64 -= np.max(sc64, axis=-2, keepdims=True)
        w64 = np.exp(sc64) / np.sum(np.exp(sc64), axis=-2, keepdims=True)
        ref64 = np.matmul(np.swapaxes(w64, -1, -2), keys64)
        assert float(np.max(np.abs(ref64 - np.array(got, dtype=np.float64)))) < 1e-5

    # 2. short_convolution_fused (L=1 and L=8, modes 0..3)
    for L in (1, 8):
        st = mx.random.normal((1, 3, 2304)).astype(mx.float32)
        w = mx.random.normal((2304, 4, 1)).astype(mx.bfloat16)
        x = mx.random.normal((1, L, 2304)).astype(mx.bfloat16)
        res_bf = mx.random.normal((1, L, 2304)).astype(mx.bfloat16)
        res_fp = mx.random.normal((1, L, 2304)).astype(mx.float32)
        xf = x.astype(mx.float32)
        ctx = mx.concatenate([st, xf], axis=1)
        ref_nst = ctx[:, -3:]
        ref_raw = ((mx.conv1d(ctx, w.astype(mx.float32), groups=2304) + xf) / math.sqrt(5.0)).astype(mx.bfloat16)
        for rm, r_in, ref_out in [
            (0, None, ref_raw),
            (3, None, ref_raw.astype(mx.float32)),
            (1, res_bf, ref_raw.astype(mx.float32) + res_bf.astype(mx.float32)),
            (2, res_fp, (res_fp + ref_raw.astype(mx.float32)).astype(mx.bfloat16)),
        ]:
            got_out, got_nst = K.short_convolution_fused(st, w, x, res=r_in, res_mode=rm)
            mx.eval(ref_out, got_out, ref_nst, got_nst)
            assert float(mx.max(mx.abs(ref_nst - got_nst)).item()) == 0.0
            assert float(mx.max(mx.abs(ref_out.astype(mx.float32) - got_out.astype(mx.float32))).item()) < 2e-2

    # 3. loop_controller_norm_down + loop_controller_up_blend (L=1 and L=5)
    for L in (1, 5):
        x = mx.random.normal((1, L, 2304)).astype(mx.float32)
        msg = mx.random.normal((1, L, 2304)).astype(mx.float32)
        eps = mx.array([1e-6], mx.float32)
        dw = mx.random.normal((64, 4608)).astype(mx.bfloat16) * 0.05
        fw = mx.random.normal((2304, 64)).astype(mx.bfloat16) * 0.05
        gw = mx.random.normal((2304, 64)).astype(mx.bfloat16) * 0.05
        packed = mx.concatenate([fw, gw], axis=0)
        pair = mx.concatenate(
            [mx.fast.rms_norm(x, None, 1e-6), mx.fast.rms_norm(msg, None, 1e-6)], axis=-1
        ).astype(mx.bfloat16)
        h_ref = (pair @ dw.T).astype(mx.float32)
        out_ref = (1.0 - mx.tanh(h_ref @ fw.T)) * x + mx.tanh(h_ref @ gw.T) * msg
        h_got = K.loop_controller_norm_down(x, msg, eps, dw)
        out_got = K.loop_controller_up_blend(h_got, packed, x, msg)
        mx.eval(h_ref, h_got, out_ref, out_got)
        assert float(mx.max(mx.abs(h_ref - h_got)).item()) == 0.0
        assert float(mx.max(mx.abs(out_ref - out_got)).item()) < 1e-5

    # 4. prep_qkrv_fused (with and without tau)
    for use_tau in (False, True):
        q = mx.random.normal((1, 5, 24 * 128)).astype(mx.bfloat16)
        k = mx.random.normal((1, 5, 8 * 128)).astype(mx.bfloat16)
        v = mx.random.normal((1, 5, 8 * 128)).astype(mx.bfloat16)
        r = mx.random.normal((1, 5, 24 * 16)).astype(mx.bfloat16)
        qs = mx.random.normal((24, 128)).astype(mx.bfloat16)
        rs = mx.random.normal((24, 16)).astype(mx.bfloat16)
        tau = (1.0 + 0.1 * mx.log(mx.maximum((mx.arange(5) + 10000) / 8192.0, 1.0))) if use_tau else None
        q_r = l2_clip(q.reshape(1, 5, 24, 128), 22.627416997969522, qs)
        k_r = l2_clip(k.reshape(1, 5, 8, 128), 22.627416997969522)
        v_r = mx.fast.rms_norm(v.reshape(1, 5, 8, 128).astype(mx.float32), None, 1e-6).astype(mx.bfloat16)
        r_r = l2_clip(r.reshape(1, 5, 24, 16), 8.0, rs)
        if tau is not None:
            q_r = (q_r.astype(mx.float32) * tau[None, :, None, None]).astype(mx.bfloat16)
            r_r = (r_r.astype(mx.float32) * tau[None, :, None, None]).astype(mx.bfloat16)
        q_r, k_r, v_r = (a.transpose(0, 2, 1, 3) for a in (q_r, k_r, v_r))
        q_g, k_g, v_g, r_g = K.prep_qkrv_fused(
            q, k, v, r, qs, rs, 24, 8, 128, 16, 22.627416997969522, 8.0, True, 1e-6, tau=tau
        )
        mx.eval(q_r, k_r, v_r, r_r, q_g, k_g, v_g, r_g)
        for a, b in ((q_r, q_g), (k_r, k_g), (v_r, v_g), (r_r, r_g)):
            assert float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()) < 5e-2


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU required")
def test_ernie55_phase2_fused_kernels_parity():
    from vmlx_engine.models.ernie5_5_moe.kernels import (
        ernie_multi_router_topk,
        ernie_shared_situ_latent_norm_concat,
        fused_situ,
    )

    mx.random.seed(2026)

    # 1. ernie_multi_router_topk vs reference argpartition + normalize + scale
    for norm in (True, False):
        sc = mx.random.uniform(0.0, 2.0, (2, 7, 512), dtype=mx.float32)
        bi = mx.random.normal((512,), dtype=mx.float32) * 0.25
        sp = mx.random.uniform(0.5, 2.0, (512,), dtype=mx.float32)
        ids_k, w_k = ernie_multi_router_topk(sc, bi, sp, top_k=10, normalize=norm)
        ids_r = mx.argpartition(-(sc + bi), kth=9, axis=-1)[..., :10]
        w_r = mx.take_along_axis(sc, ids_r, axis=-1)
        if norm:
            w_r = w_r / mx.maximum(w_r.sum(axis=-1, keepdims=True), 1e-20)
        w_r = w_r * sp[ids_r]
        mx.eval(ids_k, w_k, ids_r, w_r)
        assert ids_k.shape == (2, 7, 10) and w_k.shape == (2, 7, 10)
        for b in range(2):
            for t in range(7):
                sk = set(ids_k[b, t].tolist())
                sr = set(ids_r[b, t].tolist())
                assert sk == sr
                mk = dict(zip(ids_k[b, t].tolist(), w_k[b, t].tolist()))
                mr = dict(zip(ids_r[b, t].tolist(), w_r[b, t].tolist()))
                assert max(abs(mk[e] - mr[e]) for e in sk) < 1e-6

    # 2. ernie_shared_situ_latent_norm_concat vs reference fused_situ + rms_norm + concat
    proj = mx.random.normal((1, 6, 5120)).astype(mx.bfloat16)
    comb = mx.random.normal((1, 6, 1024)).astype(mx.bfloat16)
    nw = mx.random.normal((1024,)).astype(mx.bfloat16)
    got = ernie_shared_situ_latent_norm_concat(proj, comb, nw, 2048, 4.0, 25.0, 1e-6)
    ref = mx.concatenate(
        [fused_situ(proj[..., 2048:4096], proj[..., :2048], 4.0, 25.0), mx.fast.rms_norm(comb, nw, 1e-6)],
        axis=-1,
    )
    mx.eval(got, ref)
    assert got.shape == ref.shape == (1, 6, 3072)
    assert float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))).item()) < 5e-2

    # 3. Split-simdgroup JANGH gather_qmv (fused=True) exact bit-parity vs separate single projections
    from vmlx_engine.jangh import kernels as JK
    from vmlx_engine.jangt import moe_kernels as MK

    for bits in (2, 3, 4):
        E, N, K = 4, 36, 512  # N=36 tests both aligned tiles and tail-guarded rows
        wg = mx.random.randint(0, 2**31 - 1, (E, N, K * bits // 32)).astype(mx.uint32)
        wu = mx.random.randint(0, 2**31 - 1, (E, N, K * bits // 32)).astype(mx.uint32)
        sg = (mx.random.normal((E, N)) * 0.1).astype(mx.float16)
        su = (mx.random.normal((E, N)) * 0.1).astype(mx.float16)
        xh = mx.random.normal((6, K)).astype(mx.float32)
        idx = mx.array([0, 1, 2, 3, 1, 2, 0, 3, 2, 1, 0, 3], dtype=mx.uint32)
        for yz in (False, True):
            got_gu = JK.gather_qmv(
                xh, wg, sg, None, idx, bits, x_per_dispatch=False,
                packed_u=wu, scales_u=su, slot_fastest=yz, act="inkling", alpha=(4.0, 25.0),
            )
            g_s = JK.gather_qmv(xh, wg, sg, None, idx, bits, x_per_dispatch=False, slot_fastest=yz)
            u_s = JK.gather_qmv(xh, wu, su, None, idx, bits, x_per_dispatch=False, slot_fastest=yz)
            ref_gu = MK.act_mlx(g_s, u_s, "inkling", 0.0, (4.0, 25.0))
            mx.eval(got_gu, ref_gu)
            assert float(mx.max(mx.abs(got_gu - ref_gu)).item()) < 1e-5


def test_attach_text_lane_idempotent(monkeypatch):
    """Multi-stage loaders (tokenizer + BatchedEngine) must never double-load multi-GB DSpark weights."""
    from types import SimpleNamespace
    from vmlx_engine.dspark import runtime as dspark_rt

    sentinel = {"drafter": object(), "target": object(), "model_type": "ernie5_5_moe"}
    model = SimpleNamespace(_vmlx_dspark_runtime=sentinel)
    called = []
    monkeypatch.setattr(dspark_rt, "load_dspark_runtime", lambda *a, **kw: called.append((a, kw)) or {"drafter": object()})

    got = dspark_rt.attach_text_lane(model, "/nonexistent/ernie")
    assert got is sentinel
    assert called == []


def test_dspark_ar_fallback_hysteresis_and_probe_burst():
    from vmlx_engine.dspark.generate import ARFallback

    fb = ARFallback(
        alpha=0.15,
        margin=0.03,
        probe_every=4,
        probe_burst=2,
        ar_switch_streak=2,
    )
    fb.record_ar(0.030)  # AR rate = 33.3 tok/s
    # Establish healthy draft rate: 4 tok / 0.060s = 66.7 tok/s (2x AR!)
    fb.record_draft(4, 0.060, rows=6)  # warm-up ignored
    fb.record_draft(4, 0.060, rows=6)
    fb.record_draft(4, 0.060, rows=6)
    assert not fb.next_is_ar()

    # Single bad sampled round (1 tok / 0.060s = 16.7 tok/s) must NOT trigger AR fallback!
    fb.record_draft(1, 0.060, rows=6)
    assert not fb.next_is_ar()

    # Sustained low acceptance (below AR * 1.03 for >= ar_switch_streak consecutive rounds) DOES switch to AR
    for _ in range(12):
        fb.record_draft(1, 0.060, rows=6)
    assert fb.next_is_ar()

    # After probe_every AR steps, ARFallback enters a 2-step probe burst before deciding
    for _ in range(4):
        fb.record_ar(0.030)
    assert not fb.next_is_ar()  # probe step 1
    fb.record_draft(4, 0.055, rows=6)
    assert not fb.next_is_ar()  # probe step 2 / recovered


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU required")
def test_metal_radix_topk_exact_with_recent_window():
    from vmlx_engine.metal.sparse_topk import metal_radix_topk

    mx.random.seed(11)
    for S, L, K_top, W in ((256, 4, 32, 16), (4096, 6, 128, 32)):
        scores = mx.random.normal((1, L, S)).astype(mx.float32)
        q_end = mx.arange(S - L + 1, S + 1, dtype=mx.uint32)
        older_end = mx.maximum(q_end.astype(mx.int32) - W, 0).astype(mx.uint32)
        sel = metal_radix_topk(scores, K_top, valid_end=older_end, recent_window=W, recent_end=q_end)
        mx.eval(sel)
        assert sel.shape == (1, L, K_top + W)
        sc_np = np.array(scores[0])
        sel_np = np.array(sel[0])
        for row in range(L):
            oe = int(older_end[row].item())
            qe = int(q_end[row].item())
            expected_top = set(np.argsort(-sc_np[row, :oe])[:K_top].tolist())
            got_top = set(sel_np[row, :K_top].tolist())
            assert got_top == expected_top
            expected_recent = list(range(oe, qe))
            got_recent = [int(v) for v in sel_np[row, K_top:] if int(v) != 0xFFFFFFFF]
            assert got_recent == expected_recent


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU required")
def test_dsa_decode_index_scores_and_sparse_mla_prefill_parity():
    from vmlx_engine.metal.dsa_index_scores import dsa_decode_index_scores
    from vmlx_engine.models.ernie5_5_moe.kernels import ernie_sparse_mla_prefill

    mx.random.seed(23)
    # 1. DSA decode index scores vs FP32 reference
    H, D, S = 64, 128, 512
    q = mx.random.normal((1, 1, H, D)).astype(mx.bfloat16) * 0.2
    k = mx.random.normal((1, S, D)).astype(mx.bfloat16) * 0.2
    w = mx.random.uniform(0.0, 1.0, (1, 1, H)).astype(mx.float32)
    got = dsa_decode_index_scores(q, k, w, scale=1.0)
    ref = (mx.maximum(q.transpose(0, 2, 1, 3).astype(mx.float32) @ k[:, None].astype(mx.float32).swapaxes(-1, -2), 0.0)
           * w.transpose(0, 2, 1)[..., None]).sum(axis=1)
    mx.eval(got, ref)
    assert got.shape == (1, 1, S)
    assert float(mx.max(mx.abs(got - ref)).item()) < 5e-2

    # 2. ernie_sparse_mla_prefill with valid + sentinel indices + sinks vs exact gathered SDPA
    Hm, L, Sm, K_sel = 16, 4, 256, 32
    qn = mx.random.normal((1, Hm, L, 512)).astype(mx.bfloat16) * 0.1
    qp = mx.random.normal((1, Hm, L, 64)).astype(mx.bfloat16) * 0.1
    kv = mx.random.normal((1, 1, Sm, 512)).astype(mx.bfloat16) * 0.1
    kp = mx.random.normal((1, 1, Sm, 64)).astype(mx.bfloat16) * 0.1
    sinks = mx.random.normal((Hm,)).astype(mx.float32) * 0.5
    idx_list = []
    for t in range(L):
        row = list(range(t * 10, t * 10 + 28)) + [0xFFFFFFFF] * 4
        idx_list.append(row)
    idx = mx.array([idx_list], dtype=mx.uint32)
    scale = (192 + 64) ** -0.5
    out_got = ernie_sparse_mla_prefill(qn, qp, kv, kp, idx, sinks, scale)
    mx.eval(out_got)

    # Exact FP32 reference per query
    for t in range(L):
        valid_ids = mx.array([i for i in idx_list[t] if i != 0xFFFFFFFF], dtype=mx.int32)
        kv_t = kv[:, :, valid_ids, :].astype(mx.float32)
        kp_t = kp[:, :, valid_ids, :].astype(mx.float32)
        logits = (
            (qn[:, :, t : t + 1, :].astype(mx.float32) @ kv_t.swapaxes(-1, -2))
            + (qp[:, :, t : t + 1, :].astype(mx.float32) @ kp_t.swapaxes(-1, -2))
        ) * scale
        logits_sink = mx.concatenate([logits, sinks[None, :, None, None]], axis=-1)
        probs = mx.softmax(logits_sink, axis=-1)[..., :-1]
        ref_t = (probs @ kv_t).astype(mx.bfloat16)
        mx.eval(ref_t)
        diff = float(mx.max(mx.abs(out_got[:, :, t : t + 1, :].astype(mx.float32) - ref_t.astype(mx.float32))).item())
        assert diff < 1e-2


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU required")
def test_ernie_bf16_mhc_mxfp8_multi_gemv_and_mxfp4_switch_parity():
    from vmlx_engine.models.ernie5_5_moe.janght import ErnieMXFP4SwitchGLU
    from vmlx_engine.models.ernie5_5_moe.kernels import (
        ernie_hc_expand_collapse_norm,
        ernie_hc_head_forward,
        ernie_mhc_collapse_norm,
        ernie_mxfp8_multi_gemv,
        fused_situ,
    )

    mx.random.seed(99)
    # 1. BF16 mHC fn vs FP32 mHC fn parity
    x = mx.random.normal((1, 4, 4, 4096)).astype(mx.bfloat16) * 0.2
    sub = mx.random.normal((1, 4, 4096)).astype(mx.bfloat16) * 0.2
    fn_f32 = mx.random.normal((24, 16384)).astype(mx.float32) * 0.02
    fn_bf16 = fn_f32.astype(mx.bfloat16)
    sc = mx.array([0.1, 0.1, 0.1], dtype=mx.float32)
    ba = mx.zeros((24,), dtype=mx.float32)
    nw = mx.ones((4096,), dtype=mx.bfloat16)
    n32, p32, c32 = ernie_mhc_collapse_norm(x, fn_f32, sc, ba, nw)
    n16, p16, c16 = ernie_mhc_collapse_norm(x, fn_bf16, sc, ba, nw)
    x32, en32, ep32, ec32 = ernie_hc_expand_collapse_norm(sub, x, p32, c32, fn_f32, sc, ba, nw)
    x16, en16, ep16, ec16 = ernie_hc_expand_collapse_norm(sub, x, p16, c16, fn_bf16, sc, ba, nw)
    hfn32 = mx.random.normal((4, 16384)).astype(mx.float32) * 0.02
    h32 = ernie_hc_head_forward(x, hfn32, mx.array([0.1], mx.float32), mx.zeros((4,), mx.float32))
    h16 = ernie_hc_head_forward(x, hfn32.astype(mx.bfloat16), mx.array([0.1], mx.float32), mx.zeros((4,), mx.float32))
    mx.eval(n32, n16, en32, en16, h32, h16)
    # BF16 rounding tolerance: <= 2 BF16 ULPs at magnitude ~4.0 (= 0.0625)
    assert float(mx.max(mx.abs(n32.astype(mx.float32) - n16.astype(mx.float32))).item()) <= 0.0625
    assert float(mx.max(mx.abs(en32.astype(mx.float32) - en16.astype(mx.float32))).item()) <= 0.0625
    assert float(mx.max(mx.abs(h32.astype(mx.float32) - h16.astype(mx.float32))).item()) <= 0.0625

    # 2. ernie_mxfp8_multi_gemv (2D + 3D grouped, T=6) vs MLX quantized_matmul
    for grouped, wshape, xshape in [
        (False, (512, 1024), (1, 6, 1024)),
        (True, (4, 256, 512), (1, 4, 6, 512)),
    ]:
        w_raw = mx.random.normal(wshape).astype(mx.bfloat16) * 0.1
        wq, wsc = mx.quantize(w_raw, group_size=32, bits=8, mode="mxfp8")
        xin = mx.random.normal(xshape).astype(mx.bfloat16)
        y_ref = mx.quantized_matmul(xin, wq, scales=wsc, transpose=True, group_size=32, bits=8, mode="mxfp8")
        y_got = ernie_mxfp8_multi_gemv(xin, wq, wsc, grouped=grouped)
        mx.eval(y_ref, y_got)
        assert y_got is not None and y_got.shape == y_ref.shape
        rel = float(mx.max(mx.abs(y_ref.astype(mx.float32) - y_got.astype(mx.float32))).item()) / (
            float(mx.max(mx.abs(y_ref.astype(mx.float32))).item()) + 1e-9
        )
        assert rel < 5e-3

    # 3. ErnieMXFP4SwitchGLU decode (T=2) and sorted prefill (T=20) parity
    E, D_in, H_in, K_top = 8, 128, 128, 2
    mod = ErnieMXFP4SwitchGLU(E, D_in, H_in, beta=4.0, linear_beta=25.0)
    mod.SORT_THRESHOLD = 16
    w_gu = mx.random.normal((E, 2 * H_in, D_in)).astype(mx.bfloat16) * 0.1
    w_dn = mx.random.normal((E, D_in, H_in)).astype(mx.bfloat16) * 0.1
    mod.gu_weight, mod.gu_scales = mx.quantize(w_gu, group_size=32, bits=4, mode="mxfp4")
    mod.down_weight, mod.down_scales = mx.quantize(w_dn, group_size=32, bits=4, mode="mxfp4")
    # Compare decode path vs prefill path on identical inputs
    xm = mx.random.normal((1, 10, D_in)).astype(mx.bfloat16)
    ids = mx.random.randint(0, E, (1, 10, K_top)).astype(mx.uint32)
    rw = mx.full((1, 10, K_top), 0.5, dtype=mx.float32)
    mod.SORT_THRESHOLD = 1000
    y_dec = mod(xm, ids, rw)
    mod.SORT_THRESHOLD = 1
    y_pre = mod(xm, ids, rw)
    mx.eval(y_dec, y_pre)
    assert y_dec.shape == y_pre.shape == (1, 10, D_in)
    assert float(mx.max(mx.abs(y_dec.astype(mx.float32) - y_pre.astype(mx.float32))).item()) < 5e-2


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU required")
def test_inkling2_phase3_attention_gemv_router_and_cache_parity():
    from mlx_lm.models.cache import RotatingKVCache
    from vmlx_engine.models.inkling2.cache_state import start_speculative_cache
    from vmlx_engine.models.inkling2.kernels import (
        affine8_g64_multi_gemv,
        inkling_flash_attention_prefill,
        inkling_flash_decode,
        inkling_router_topk,
        inkling_verify_attention,
    )
    from vmlx_engine.models.inkling2.language import _route
    from vmlx_engine.models.inkling2.primitives import banded_additive_mask
    from vmlx_engine.inkling_dflash2 import _fused_grouped_dynamic_convolve

    mx.random.seed(314)

    # 1. affine8_g64_multi_gemv: exact vs MLX quantized_matmul + bit-exact row invariance + softcap/slice
    K, N = 2304, 1024
    w_raw = mx.random.normal((N, K)).astype(mx.bfloat16) * 0.05
    wq, sc, bi = mx.quantize(w_raw, group_size=64, bits=8, mode="affine")
    x8 = mx.random.normal((1, 8, K)).astype(mx.bfloat16)
    y8 = affine8_g64_multi_gemv(x8, wq, sc, bi, out_dtype=mx.float32, softcap=30.0, out_cols=900)
    y1 = affine8_g64_multi_gemv(x8[:, 3:4, :], wq, sc, bi, out_dtype=mx.float32, softcap=30.0, out_cols=900)
    y_ref = mx.quantized_matmul(x8, wq, sc, bi, transpose=True, group_size=64, bits=8, mode="affine").astype(mx.float32)
    y_ref = mx.tanh(y_ref / 30.0) * 30.0
    y_ref = y_ref[..., :900]
    mx.eval(y8, y1, y_ref)
    assert y8.shape == (1, 8, 900)
    assert float(mx.max(mx.abs(y8[:, 3:4, :] - y1)).item()) == 0.0
    assert float(mx.max(mx.abs(y8 - y_ref)).item()) < 5e-2

    # 2. inkling_router_topk exact parity vs _route for T=1 and T=8
    gw = mx.random.normal((128, 2304)).astype(mx.bfloat16) * 0.05
    gb = mx.random.normal((128,)).astype(mx.float32) * 0.1
    for T in (1, 8):
        xr = x8[:, :T, :]
        ids_k, w_k = inkling_router_topk(xr, gw, gb, 8, 2.5)
        ids_r, w_r = _route(xr, gw, gb, 8, 2.5, exact_rows=False)
        mx.eval(ids_k, w_k, ids_r, w_r)
        for t in range(T):
            assert set(ids_k[0, t].tolist()) == set(ids_r[0, t].tolist())
            mk = dict(zip(ids_k[0, t].tolist(), w_k[0, t].tolist()))
            mr = dict(zip(ids_r[0, t].tolist(), w_r[0, t].tolist()))
            assert max(abs(mk[e] - mr[e]) for e in mk) < 1e-4

    # 3. Zero-mask inkling_verify_attention (LQ=8, sliding & full) + inkling_flash_decode (LQ=1, S=2048)
    #    + inkling_flash_attention_prefill (LQ=64, S=256)
    H, Hkv, D, Dr = 24, 8, 128, 16
    scale = D ** -0.5
    rel_proj = mx.random.normal((Dr, 256)).astype(mx.bfloat16) * 0.1
    rel_proj_t = mx.contiguous(rel_proj.T)
    for sliding, rel_ext, LQ, S in [(128, 128, 8, 192), (0, 256, 8, 320)]:
        q = mx.random.normal((1, H, LQ, D)).astype(mx.bfloat16) * 0.2
        k = mx.random.normal((1, Hkv, S, D)).astype(mx.bfloat16) * 0.2
        v = mx.random.normal((1, Hkv, S, D)).astype(mx.bfloat16) * 0.2
        r = mx.random.normal((1, LQ, H, Dr)).astype(mx.bfloat16) * 0.2
        sk = mx.random.normal((1, H, LQ, 1)).astype(mx.bfloat16) * 0.2
        got = inkling_verify_attention(q, k, v, r, rel_proj_t[:, :rel_ext] if False else mx.contiguous(rel_proj[:, :rel_ext].T), sk, scale=scale, sliding=sliding, rel_extent=rel_ext)
        m = banded_additive_mask(r, rel_proj[:, :rel_ext], S - LQ, S, sliding, rel_ext, shape_ref=k)
        m_s = mx.concatenate([m, sk], axis=-1)
        k_s = mx.concatenate([k, mx.zeros((1, Hkv, 1, D), k.dtype)], axis=2)
        v_s = mx.concatenate([v, mx.zeros((1, Hkv, 1, D), v.dtype)], axis=2)
        ref = mx.fast.scaled_dot_product_attention(q, k_s, v_s, scale=scale, mask=m_s)
        mx.eval(got, ref)
        assert float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))).item()) < 5e-2

    # 4. Single-request RotatingKVCache zero-copy commit parity (before wrap, crossing, and after wrap!)
    for init_len, max_sz, block_len, keep in [(30, 64, 8, 5), (60, 64, 8, 5), (64, 64, 8, 5), (90, 64, 8, 3)]:
        c_fast = RotatingKVCache(max_size=max_sz, keep=0)
        c_ref = RotatingKVCache(max_size=max_sz, keep=0)
        k0 = mx.random.normal((1, 4, init_len, 16)).astype(mx.bfloat16)
        v0 = mx.random.normal((1, 4, init_len, 16)).astype(mx.bfloat16)
        k1 = mx.random.normal((1, 4, block_len, 16)).astype(mx.bfloat16)
        v1 = mx.random.normal((1, 4, block_len, 16)).astype(mx.bfloat16)
        c_fast.update_and_fetch(k0, v0)
        c_ref.update_and_fetch(k0, v0)
        tx = start_speculative_cache([c_fast], block_len)
        c_fast.update_and_fetch(k1, v1)
        tx.commit([keep])
        c_ref.update_and_fetch(k1[:, :, :keep, :], v1[:, :, :keep, :])
        kf = c_fast._temporal_order(c_fast.keys)
        kr = c_ref._temporal_order(c_ref.keys)
        vf = c_fast._temporal_order(c_fast.values)
        vr = c_ref._temporal_order(c_ref.values)
        mx.eval(kf, kr, vf, vr)
        assert c_fast.offset == c_ref.offset
        assert kf.shape == kr.shape
        assert float(mx.max(mx.abs(kf.astype(mx.float32) - kr.astype(mx.float32))).item()) == 0.0
        assert float(mx.max(mx.abs(vf.astype(mx.float32) - vr.astype(mx.float32))).item()) == 0.0

    # 5. Fused GroupedDynamicCausalConv parity
    hid = mx.random.normal((1, 8, 256)).astype(mx.bfloat16)
    dyn = mx.random.normal((1, 8, 2, 16)).astype(mx.bfloat16)
    base = mx.random.normal((2, 256)).astype(mx.bfloat16)
    got_c = _fused_grouped_dynamic_convolve(hid, dyn, base, 16)
    from dflash.model_mlx import _grouped_dynamic_convolve as ref_gdconv
    ref_c = ref_gdconv(hid, dyn, base, 16)
    mx.eval(got_c, ref_c)
    # BF16 rounding tolerance: <= 2 BF16 ULPs at magnitude ~5.0 (= 0.0625)
    assert float(mx.max(mx.abs(got_c.astype(mx.float32) - ref_c.astype(mx.float32))).item()) <= 0.0625

    # 6. Fused DFlash2 CandidateSelector (dflash2_topk + dflash2_markov_select_greedy + step_scores)
    from vmlx_engine.models.inkling2.dflash2_kernels import (
        dflash2_markov_select_greedy,
        dflash2_markov_step_scores,
        dflash2_topk,
    )

    P_sel, V_sel, K_sel, R_sel = 7, 16384, 16, 256
    lg_sel = mx.random.normal((1, P_sel, V_sel)).astype(mx.float32) * 4.0
    tk_vals, tk_idxs = dflash2_topk(lg_sel, K_sel)
    mx.eval(tk_vals, tk_idxs)
    lg_np = np.array(lg_sel[0])
    for p in range(P_sel):
        ref_idx = np.argsort(-lg_np[p])[:K_sel]
        assert np.array_equal(np.array(tk_idxs[0, p]), ref_idx)
        assert np.allclose(np.array(tk_vals[0, p]), lg_np[p, ref_idx], atol=0.0)

    hp = mx.random.normal((1, P_sel, R_sel)).astype(mx.bfloat16)
    pw = mx.random.normal((V_sel, R_sel)).astype(mx.bfloat16) * 0.1
    sw = mx.random.normal((V_sel, R_sel)).astype(mx.bfloat16) * 0.1
    anc = mx.array([42], dtype=mx.int32)
    path_fast = dflash2_markov_select_greedy(tk_vals, tk_idxs, hp, pw, sw, anc, 30.0)
    pred = anc
    ref_path = []
    for p in range(P_sel):
        ed = mx.sum(
            pw[pred][:, None].astype(mx.float32)
            * hp[:, p, None].astype(mx.float32)
            * sw[tk_idxs[:, p]].astype(mx.float32),
            axis=-1,
        )
        sc_ref = 30.0 * mx.tanh((tk_vals[:, p] + ed) / 30.0)
        sc_got = dflash2_markov_step_scores(tk_vals[:, p], tk_idxs[:, p], hp[:, p], pw, sw, pred, 30.0)
        mx.eval(sc_ref, sc_got)
        assert float(mx.max(mx.abs(sc_ref - sc_got)).item()) < 1e-4
        sel = mx.argmax(sc_ref, axis=-1)
        pred = mx.take_along_axis(tk_idxs[:, p], sel[:, None], axis=-1)[:, 0]
        ref_path.append(pred)
    ref_path_arr = mx.stack(ref_path, axis=1)
    mx.eval(path_fast, ref_path_arr)
    assert mx.array_equal(path_fast, ref_path_arr).item()

    # 7. GQA-Grouped M5 MPP Split-K Decode & Verify Attention (LQ=1 and LQ=8)
    from vmlx_engine.models.inkling2.kernels import inkling_gqa_mpp_attention

    H_g, Hkv_g, D_g, Dr_g, S_g = 24, 4, 128, 16, 1024
    rp_g = mx.random.normal((Dr_g, 256)).astype(mx.bfloat16) * 0.1
    rp_gt = mx.contiguous(rp_g.T)
    for LQ_g, sl_g in ((1, 0), (8, 0), (8, 512)):
        qg = mx.random.normal((1, H_g, LQ_g, D_g)).astype(mx.bfloat16) * 0.3
        kg = mx.random.normal((1, Hkv_g, S_g, D_g)).astype(mx.bfloat16) * 0.3
        vg = mx.random.normal((1, Hkv_g, S_g, D_g)).astype(mx.bfloat16) * 0.3
        rg = mx.random.normal((1, LQ_g, H_g, Dr_g)).astype(mx.bfloat16) * 0.2
        skg = mx.random.normal((1, H_g, LQ_g, 1)).astype(mx.bfloat16) * 0.2
        got_g = inkling_gqa_mpp_attention(qg, kg, vg, rg, rp_gt, skg, scale=D_g ** -0.5, sliding=sl_g, rel_extent=256)
        if got_g is not None:
            mg = banded_additive_mask(rg, rp_g, S_g - LQ_g, S_g, sl_g, 256, shape_ref=kg)
            m_sg = mx.concatenate([mg, skg], axis=-1)
            k_sg = mx.concatenate([kg, mx.zeros((1, Hkv_g, 1, D_g), kg.dtype)], axis=2)
            v_sg = mx.concatenate([vg, mx.zeros((1, Hkv_g, 1, D_g), vg.dtype)], axis=2)
            ref_g = mx.fast.scaled_dot_product_attention(qg, k_sg, v_sg, scale=D_g ** -0.5, mask=m_sg)
            mx.eval(got_g, ref_g)
            assert float(mx.max(mx.abs(got_g.astype(mx.float32) - ref_g.astype(mx.float32))).item()) < 3e-2


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU required")
def test_ernie_dspark_markov_topk_chain_parity():
    from vmlx_engine.models.ernie5_5_moe.kernels import (
        ernie_dspark_markov_argmax,
        ernie_dspark_markov_topk_chain,
    )

    mx.random.seed(77)
    K_pos, V, R = 6, 32768, 256
    # Spiky base logits matching real LM distributions so top-1 always lies well within Top-512
    base = mx.random.normal((K_pos, V), dtype=mx.float32) * 2.0
    for i in range(K_pos):
        base[i, (i + 1) * 1337] = 14.0
        base[i, (i + 1) * 1337 + 1] = 13.5
    emb = (mx.random.normal((V, R)) * 0.05).astype(mx.bfloat16)
    w2 = (mx.random.normal((V, R)) * 0.05).astype(mx.bfloat16)
    ranges = mx.array([-10.0, 10.0, -20.0, 20.0], dtype=mx.float32)
    ts = mx.array([0.05, -0.05, 0.01, -0.01], dtype=mx.float32)
    anc = mx.array(7, dtype=mx.int32)

    got = ernie_dspark_markov_topk_chain(base, emb, w2, anc, ranges, ts)
    tk = anc
    ref = []
    for i in range(K_pos):
        tk = ernie_dspark_markov_argmax(base[i], w2, emb[tk], ranges, ts)
        ref.append(tk)
    ref_arr = mx.stack(ref)
    mx.eval(got, ref_arr)
    assert mx.array_equal(got, ref_arr).item()


def test_inkling_dflash2_prefix_store_clone_and_isolation():
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
    from vmlx_engine.inkling_dflash2 import _InklingPrefixStore, _clone_inkling_caches

    mx.random.seed(2026)
    # Build Inkling's exact mix: RotatingKVCache (empty, pre-wrap, exact-full, post-wrap),
    # KVCache, ArraysCache(4), and nested CacheList.
    r_empty = RotatingKVCache(max_size=64, keep=0)
    r_pre = RotatingKVCache(max_size=64, keep=0)
    r_full = RotatingKVCache(max_size=64, keep=0)
    r_wrap = RotatingKVCache(max_size=64, keep=0)
    kv = KVCache()
    arr = ArraysCache(4)

    def _kv(n):
        return (
            mx.random.normal((1, 4, n, 16)).astype(mx.bfloat16),
            mx.random.normal((1, 4, n, 16)).astype(mx.bfloat16),
        )

    r_pre.update_and_fetch(*_kv(30))
    r_full.update_and_fetch(*_kv(64))
    r_wrap.update_and_fetch(*_kv(95))
    kv.update_and_fetch(*_kv(30))
    arr[0] = mx.random.normal((1, 3, 16)).astype(mx.bfloat16)
    arr[2] = mx.random.normal((1, 3, 16)).astype(mx.bfloat16)
    nested = CacheList(RotatingKVCache(max_size=32, keep=0), ArraysCache(2))
    nested.caches[0].update_and_fetch(*_kv(45))
    nested.caches[1][0] = mx.ones((1, 2, 8), dtype=mx.bfloat16)

    caches = [r_empty, r_pre, r_full, r_wrap, kv, arr, nested]
    hidden = mx.random.normal((1, 30, 32)).astype(mx.bfloat16)
    logits = mx.random.normal((1, 1, 64)).astype(mx.float32)

    store = _InklingPrefixStore(max_entries=2)
    store.put([1, 2, 3, 4], caches, hidden, logits)
    hit = store.lookup([1, 2, 3, 4, 5, 6])
    assert hit is not None
    reused, entry = hit
    assert reused == 4
    assert store.last_source == "ram"

    # Mutate every live original cache (simulating subsequent decode steps)
    next_k, next_v = _kv(7)
    r_pre_before = (1 * r_pre._temporal_order(r_pre.keys), 1 * r_pre._temporal_order(r_pre.values))
    r_full_before = (1 * r_full._temporal_order(r_full.keys), 1 * r_full._temporal_order(r_full.values))
    r_wrap_before = (1 * r_wrap._temporal_order(r_wrap.keys), 1 * r_wrap._temporal_order(r_wrap.values))
    kv_before = (1 * kv.state[0], 1 * kv.state[1])
    arr0_before = 1 * arr[0]
    mx.eval(*r_pre_before, *r_full_before, *r_wrap_before, *kv_before, arr0_before)

    r_pre.update_and_fetch(next_k, next_v)
    r_full.update_and_fetch(next_k, next_v)
    r_wrap.update_and_fetch(next_k, next_v)
    kv.update_and_fetch(next_k, next_v)
    arr[0] = mx.zeros_like(arr[0])
    nested.caches[0].update_and_fetch(next_k, next_v)

    # Restore from store and verify bit-exact pre-mutation state + parity on follow-up updates!
    restored = _clone_inkling_caches(entry.cache)
    c_empty, c_pre, c_full, c_wrap, c_kv, c_arr, c_nested = restored
    assert c_empty.keys is None and c_empty.offset == 0
    assert c_pre.offset == 30 and c_full.offset == 64 and c_wrap.offset == 95 and c_kv.offset == 30
    assert mx.array_equal(c_pre._temporal_order(c_pre.keys), r_pre_before[0]).item()
    assert mx.array_equal(c_full._temporal_order(c_full.keys), r_full_before[0]).item()
    assert mx.array_equal(c_wrap._temporal_order(c_wrap.keys), r_wrap_before[0]).item()
    assert mx.array_equal(c_kv.state[0], kv_before[0]).item()
    assert mx.array_equal(c_arr[0], arr0_before).item()
    assert c_arr[1] is None
    assert c_nested.caches[0].offset == 45

    # Advance restored caches with the exact same next_k/next_v and verify bit-exact parity vs live!
    for live_c, rest_c in ((r_pre, c_pre), (r_full, c_full), (r_wrap, c_wrap), (nested.caches[0], c_nested.caches[0])):
        rest_c.update_and_fetch(next_k, next_v)
        assert live_c.offset == rest_c.offset
        assert mx.array_equal(live_c._temporal_order(live_c.keys), rest_c._temporal_order(rest_c.keys)).item()
        assert mx.array_equal(live_c._temporal_order(live_c.values), rest_c._temporal_order(rest_c.values)).item()
    c_kv.update_and_fetch(next_k, next_v)
    assert mx.array_equal(kv.state[0], c_kv.state[0]).item()
    assert mx.array_equal(kv.state[1], c_kv.state[1]).item()




