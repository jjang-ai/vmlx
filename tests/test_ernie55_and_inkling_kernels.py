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
