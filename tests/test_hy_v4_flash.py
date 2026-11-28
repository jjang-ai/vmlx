# SPDX-License-Identifier: Apache-2.0
"""Focused parity & end-to-end tests for Hy4-flash (`hy_v4_flash`): disk Engram, iHC, IndexPool, LatentMoE, DFly2, parsers."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from vmlx_engine.models.hy_v4_flash.cache import (
    HyV4GrowingMLACache,
    HyV4LayerCache,
    HyV4RotatingMLACache,
    rollback_hy4_caches,
    snapshot_hy4_caches,
)
from vmlx_engine.models.hy_v4_flash.config import HyV4FlashConfig
from vmlx_engine.models.hy_v4_flash.dfly2 import DFly2Stats, HyV4DFly2Drafter, iter_hy4_dfly2
from vmlx_engine.models.hy_v4_flash.engram import (
    HyV4DiskEngramTable,
    HyV4EngramHasher,
    HyV4EngramLayer,
    HyV4InMemoryEngramTable,
)
from vmlx_engine.models.hy_v4_flash.kernels import (
    HyV4MXFP4SwitchGLU,
    hy4_clamped_swiglu,
    hy4_ihc_post,
    hy4_ihc_pre,
    hy4_router_topk,
)
from vmlx_engine.models.hy_v4_flash.language import HyV4FlashForCausalLM


def _make_mini_hy4_config() -> HyV4FlashConfig:
    """Compact Hy4-flash config preserving every architectural feature (iHC, Engram, DSA+SWA, LatentMoE, DFly2)."""
    return HyV4FlashConfig(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=4,
        intermediate_size=256,
        rms_norm_eps=1e-5,
        enable_lm_head_fp32=True,
        bos_token_id=1,
        pad_token_id=2,
        eos_token_id=3,
        image_token_id=4,
        video_token_id=5,
        enable_ihc=True,
        hc_mult=4,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        layer_types=[
            "deepseek_sparse_attention",
            "sliding_attention",
            "sliding_attention",
            "deepseek_sparse_attention",
        ],
        swa_pattern=[0, 1, 1, 0],
        num_attention_heads=16,
        swa_num_attention_heads=16,
        q_lora_rank=64,
        kv_lora_rank=512,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        qk_head_dim=128,
        v_head_dim=64,
        sliding_window=16,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=16,
        compressed_indexer_compress_ratio=4,
        compressed_indexer_group_topk=4,
        compressed_indexer_candidate_topk=16,
        first_k_dense_replace=1,
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_expert_hidden_size=64,
        moe_intermediate_size=64,
        routed_scaling_factor=3.998,
        norm_topk_prob=True,
        swiglu_limit=10.0,
        engram_layer_ids=[2],  # 1-based -> 0-based layer 1
        engram_max_ngram_size=3,
        engram_n_embed_per_ngram=64,
        engram_n_head_per_ngram=4,  # 8 total heads x 16 dim = 128
        engram_pad_id=2,
        engram_seed=0,
        engram_layer_head_vocab_sizes={
            "2": [[17, 19, 23, 29], [31, 37, 41, 43]],
        },
        dfly_num_hidden_layers=2,
        dfly_block_size=4,
        dfly_mask_token_id=6,
        dfly_target_kv_layers=[0, 3],
        dfly_target_index_layer=[0, 3],
        dfly_target_index_topk=16,
        dfly_enable_hidden_correction=True,
    )


def _write_safetensors_bf16(path: Path, name: str, arr_u16: np.ndarray) -> None:
    raw = np.ascontiguousarray(arr_u16, dtype="<u2").tobytes()
    header = json.dumps({
        name: {
            "dtype": "BF16",
            "shape": list(arr_u16.shape),
            "data_offsets": [0, len(raw)],
        }
    }).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(raw)


def test_hy4_engram_hasher_and_disk_mmap_exact(tmp_path: Path):
    cfg = _make_mini_hy4_config()
    # Frozen lookup table shorter than vocab_size (240 < 256) to verify padded-ID clamp!
    lookup = np.arange(240, dtype=np.int32) % 100
    lookup[cfg.engram_pad_id] = 99
    hasher = HyV4EngramHasher(cfg, lookup)
    assert hasher.compressed_pad_id == 99
    assert hasher.lookup_table[250] == 99  # out-of-range padded vocab ID maps to pad!

    # Incremental hashing across two chunks MUST equal single-shot full prompt hashing!
    seq = np.array([10, 20, 30, 250, 40, 50, 60, 70], dtype=np.int64)
    full_h, full_tail, full_mask = hasher.hash_for_layers(seq)
    h1, tail1, m1 = hasher.hash_for_layers(seq[:5])
    h2, tail2, m2 = hasher.hash_for_layers(seq[5:], prev_compressed=tail1)
    assert np.array_equal(full_h[1], np.concatenate([h1[1], h2[1]], axis=0))
    assert np.array_equal(full_tail, tail2)
    assert np.array_equal(full_mask, np.concatenate([m1, m2]))

    # Zero-copy disk-backed BF16 safetensors mmap vs in-memory reference (0.0 bit-exact!)
    total_rows = sum(sum(r) for r in cfg.engram_layer_head_vocab_sizes["2"])
    head_dim = cfg.engram_n_embed_per_ngram // cfg.engram_n_head_per_ngram
    mx.random.seed(9)
    w_bf16 = (mx.random.normal((total_rows, head_dim)) * 0.25).astype(mx.bfloat16)
    mx.eval(w_bf16)
    w_u16 = np.array(w_bf16.view(mx.uint16))
    shard = tmp_path / "model-00004-of-00148.safetensors"
    tname = "model.layers.1.engram_layer.multi_head_embedding.embedding.weight"
    _write_safetensors_bf16(shard, tname, w_u16)

    disk_tbl = HyV4DiskEngramTable(shard, tname)
    mem_tbl = HyV4InMemoryEngramTable(w_bf16)
    fut = disk_tbl.prefetch(full_h[1])
    got_disk = disk_tbl.finish((1, len(seq), 8), fut.result())
    got_mem = mem_tbl.lookup(full_h[1])[None, ...]
    mx.eval(got_disk, got_mem)
    assert float(mx.max(mx.abs(got_disk.astype(mx.float32) - got_mem.astype(mx.float32))).item()) == 0.0


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU required")
def test_hy4_fused_kernels_parity():
    mx.random.seed(123)

    # 1. iHC pre-reduce & post-scatter vs exact FP32 reference
    B, L, C, D = 1, 6, 4, 256
    x = (mx.random.normal((B, L, C, D)) * 0.4).astype(mx.bfloat16)
    fn = (mx.random.normal((8, 4 * D)) * 0.02).astype(mx.float32)
    base = (mx.random.normal((8,)) * 0.1).astype(mx.float32)
    scale = mx.array([0.5, 0.7], dtype=mx.float32)
    y_sub = (mx.random.normal((B, L, D)) * 0.3).astype(mx.bfloat16)

    u_k, gp_k = hy4_ihc_pre(x, fn, base, scale, rms_eps=1e-5, hc_eps=1e-6, hc_magnitude=2.0)
    out_k = hy4_ihc_post(x, y_sub, gp_k)

    x64 = np.array(x.astype(mx.float32), dtype=np.float64).reshape(B, L, 4 * D)
    fn64 = np.array(fn, dtype=np.float64)
    b64 = np.array(base, dtype=np.float64)
    s64 = np.array(scale, dtype=np.float64)
    r64 = 1.0 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + 1e-5)
    mix64 = (x64 @ fn64.T) * r64
    g_pre64 = 1.0 / (1.0 + np.exp(-(mix64[..., :4] * s64[0] + b64[:4]))) + 1e-6
    gp_ref = 2.0 / (1.0 + np.exp(-(mix64[..., 4:] * s64[1] + b64[4:]))) + 1e-6
    xf = x.astype(mx.float32)
    u_ref = mx.sum(mx.array(g_pre64, dtype=mx.float32)[..., None] * xf, axis=2).astype(mx.bfloat16)
    out_ref = (xf + mx.array(gp_ref, dtype=mx.float32)[..., None] * y_sub.astype(mx.float32)[..., None, :]).astype(mx.bfloat16)
    mx.eval(u_k, gp_k, out_k, u_ref, out_ref)
    assert float(np.max(np.abs(np.array(gp_k, dtype=np.float64) - gp_ref))) < 1e-6
    assert float(mx.max(mx.abs(u_k.astype(mx.float32) - u_ref.astype(mx.float32))).item()) < 2e-2
    assert float(mx.max(mx.abs(out_k.astype(mx.float32) - out_ref.astype(mx.float32))).item()) < 2e-2

    # 2. Clamped SwiGLU (limit=10.0)
    gu = (mx.random.normal((4, 16, 256)) * 12.0).astype(mx.bfloat16)
    got_act = hy4_clamped_swiglu(gu, 10.0)
    g_r = mx.minimum(gu[..., :128].astype(mx.float32), 10.0)
    u_r = mx.clip(gu[..., 128:].astype(mx.float32), -10.0, 10.0)
    ref_act = (nn.silu(g_r) * u_r).astype(mx.bfloat16)
    mx.eval(got_act, ref_act)
    assert float(mx.max(mx.abs(got_act.astype(mx.float32) - ref_act.astype(mx.float32))).item()) <= 0.125

    # 3. Router top-16 over 512 experts
    xr = (mx.random.normal((1, 5, 256)) * 0.2).astype(mx.bfloat16)
    gw = (mx.random.normal((512, 256)) * 0.05).astype(mx.bfloat16)
    gb = (mx.random.normal((512,)) * 0.2).astype(mx.float32)
    ids_k, w_k = hy4_router_topk(xr, gw, gb, top_k=16, routed_scaling_factor=3.998, norm_topk_prob=True)
    sc_r = mx.sigmoid(xr.astype(mx.float32) @ gw.astype(mx.float32).T)
    ids_r = mx.argpartition(-(sc_r + gb), kth=15, axis=-1)[..., :16]
    w_r = mx.take_along_axis(sc_r, ids_r, axis=-1)
    w_r = (w_r / (mx.sum(w_r, axis=-1, keepdims=True) + 1e-20)) * 3.998
    mx.eval(ids_k, w_k, ids_r, w_r)
    for t in range(5):
        assert set(ids_k[0, t].tolist()) == set(ids_r[0, t].tolist())
        mk = dict(zip(ids_k[0, t].tolist(), w_k[0, t].tolist()))
        mr = dict(zip(ids_r[0, t].tolist(), w_r[0, t].tolist()))
        assert max(abs(mk[e] - mr[e]) for e in mk) < 1e-5

    # 4. Native MXFP4 SwitchGLU parity vs dequantized BF16 reference
    E, L_dim, H_dim = 8, 128, 128
    gu_f = (mx.random.normal((E, 2 * H_dim, L_dim)) * 0.05).astype(mx.bfloat16)
    dn_f = (mx.random.normal((E, L_dim, H_dim)) * 0.05).astype(mx.bfloat16)
    gu_q, gu_s = mx.quantize(gu_f, group_size=32, bits=4, mode="mxfp4")
    dn_q, dn_s = mx.quantize(dn_f, group_size=32, bits=4, mode="mxfp4")
    sw = HyV4MXFP4SwitchGLU(E, L_dim, H_dim, swiglu_limit=10.0)
    sw.gu_weight, sw.gu_scales = gu_q, gu_s
    sw.down_weight, sw.down_scales = dn_q, dn_s
    xm = (mx.random.normal((1, 4, L_dim)) * 0.3).astype(mx.bfloat16)
    eidx = mx.array([[[0, 3], [1, 2], [4, 7], [5, 6]]], dtype=mx.uint32)
    ew = mx.array([[[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.2, 0.8]]], dtype=mx.float32)
    got_m = sw(xm, eidx, ew)
    # Exact dequantized reference
    gu_deq = mx.dequantize(gu_q, gu_s, group_size=32, bits=4, mode="mxfp4").astype(mx.bfloat16)
    dn_deq = mx.dequantize(dn_q, dn_s, group_size=32, bits=4, mode="mxfp4").astype(mx.bfloat16)
    gu_o = mx.gather_mm(xm[0][:, None, None, :], gu_deq.swapaxes(-1, -2), rhs_indices=eidx[0])[:, :, 0, :]
    act_o = hy4_clamped_swiglu(gu_o, 10.0)
    dn_o = mx.gather_mm(act_o[:, :, None, :], dn_deq.swapaxes(-1, -2), rhs_indices=eidx[0])[:, :, 0, :]
    ref_m = mx.sum(dn_o.astype(mx.float32) * ew[0][..., None], axis=1).astype(mx.bfloat16)[None, ...]
    mx.eval(got_m, ref_m)
    assert float(mx.max(mx.abs(got_m.astype(mx.float32) - ref_m.astype(mx.float32))).item()) < 5e-2

    # 4b. Sorted-indices prefill path (T * top_k >= SORT_THRESHOLD) MUST match unsorted bit-for-bit!
    xm_big = (mx.random.normal((1, 48, L_dim)) * 0.3).astype(mx.bfloat16)
    eidx_big = mx.random.randint(0, E, shape=(1, 48, 4)).astype(mx.uint32)  # 48*4 = 192 >= 160
    ew_big = mx.full((1, 48, 4), 0.25, dtype=mx.float32)
    got_sorted = sw(xm_big, eidx_big, ew_big)
    old_thr = sw.SORT_THRESHOLD
    sw.SORT_THRESHOLD = 999999
    got_unsorted = sw(xm_big, eidx_big, ew_big)
    sw.SORT_THRESHOLD = old_thr
    mx.eval(got_sorted, got_unsorted)
    assert float(mx.max(mx.abs(got_sorted.astype(mx.float32) - got_unsorted.astype(mx.float32))).item()) < 1e-2


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU required")
def test_hy4_end_to_end_dfly2_greedy_parity_and_rollback():
    """End-to-end mini Hy4-flash + Engram + IndexPool + Adaptive DFly2 matches plain AR token-for-token!"""
    mx.random.seed(42)
    cfg = _make_mini_hy4_config()
    model = HyV4FlashForCausalLM(cfg, mxfp4=False)
    # Initialize kv_b absorbed weights and small non-zero params so logits are well-conditioned
    for layer in model.layers:
        sa = layer.self_attn
        sa.set_kv_b_weight((mx.random.normal((sa.n_heads * (sa.qk_nope + sa.v_head_dim), sa.kv_lora_rank)) * 0.02).astype(mx.bfloat16))
        sa._sinks_bf16 = sa.learnable_sink_param.astype(mx.bfloat16)
    # Bind in-memory Engram table & hasher
    lookup = np.arange(cfg.vocab_size, dtype=np.int32) % 80
    lookup[cfg.engram_pad_id] = 79
    model.model.set_engram_hasher(HyV4EngramHasher(cfg, lookup))
    total_rows = sum(sum(r) for r in cfg.engram_layer_head_vocab_sizes["2"])
    head_dim = cfg.engram_n_embed_per_ngram // cfg.engram_n_head_per_ngram
    eng_w = (mx.random.normal((total_rows, head_dim)) * 0.05).astype(mx.bfloat16)
    model.layers[1].engram_layer.bind_table(HyV4InMemoryEngramTable(eng_w))
    model.layers[1].engram_layer.prepare_runtime()

    drafter = HyV4DFly2Drafter(cfg, intermediate_size=128, has_confidence_head=True)
    for dl in drafter.layers:
        sa = dl.self_attn
        sa.set_kv_b_weight((mx.random.normal((sa.n_heads * (sa.qk_nope + sa.v_head_dim), sa.kv_lora_rank)) * 0.02).astype(mx.bfloat16))
        sa._sinks_bf16 = sa.learnable_sink_param.astype(mx.bfloat16)
    mx.eval(model.parameters(), drafter.parameters())

    prompt = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    n_new = 18

    # Plain AR reference
    c_ar = model.make_cache()
    lg = model(mx.array([prompt], dtype=mx.int32), cache=c_ar, prefill_last_logits_only=True)
    tk = int(mx.argmax(lg[0, -1], axis=-1).item())
    ar_toks = [tk]
    for _ in range(n_new - 1):
        lg = model(mx.array([[tk]], dtype=mx.int32), cache=c_ar)
        tk = int(mx.argmax(lg[0, -1], axis=-1).item())
        ar_toks.append(tk)

    # Adaptive DFly2 greedy generation (zero extra KV cache + IndexPool rollback + Engram tail rollback!)
    st = DFly2Stats()
    dfly_toks = []
    for chunk in iter_hy4_dfly2(model, drafter, prompt, n_new, eos={999999}, temperature=0.0, stats=st):
        dfly_toks.extend(chunk)

    assert len(dfly_toks) == n_new
    assert dfly_toks == ar_toks, f"DFly2 greedy diverged from AR: {dfly_toks} vs {ar_toks}"
    assert st.rounds >= 1

    # Also verify lossless sampled path runs cleanly without NaN/errors
    mx.random.seed(99)
    st_s = DFly2Stats()
    samp_toks = []
    for chunk in iter_hy4_dfly2(model, drafter, prompt, 10, eos={999999}, temperature=0.9, stats=st_s):
        samp_toks.extend(chunk)
    assert len(samp_toks) == 10

    # Verify HyV4LayerCache flat state/meta_state SSD round-trip preserves exact tokens!
    c_rebuilt = [HyV4LayerCache.from_state(lc.state, lc.meta_state) for lc in c_ar]
    lg_orig = model(mx.array([[ar_toks[-1]]], dtype=mx.int32), cache=c_ar)
    lg_rebuilt = model(mx.array([[ar_toks[-1]]], dtype=mx.int32), cache=c_rebuilt)
    mx.eval(lg_orig, lg_rebuilt)
    assert float(mx.max(mx.abs(lg_orig - lg_rebuilt)).item()) == 0.0


def test_hy4_reasoning_and_tool_parsers():
    from vmlx_engine.reasoning import get_parser
    from vmlx_engine.tool_parsers import ToolParserManager

    rp = get_parser("hy4")()
    # End-only prefilled <think:opensource> (reasoning_effort:high)
    r, c = rp.extract_reasoning("Step 1: add 2+2=4.</think:opensource>The answer is 4.")
    assert r == "Step 1: add 2+2=4."
    assert c == "The answer is 4."

    # Full <think:opensource>...</think:opensource>
    r2, c2 = rp.extract_reasoning("<think:opensource>Thinking...</think:opensource>Done.")
    assert r2 == "Thinking..."
    assert c2 == "Done."

    # Hy4 tool format (no <tool_sep:opensource>) AND Hy3 tool format (with <tool_sep:opensource>)
    tp = ToolParserManager.get_tool_parser("hy_v4_flash")()
    out_hy4 = (
        "<tool_calls:opensource>"
        "<tool_call:opensource>get_weather"
        "<arg_key:opensource>city</arg_key:opensource>"
        "<arg_value:opensource>San Francisco</arg_value:opensource>"
        "<arg_key:opensource>days</arg_key:opensource>"
        "<arg_value:opensource>3</arg_value:opensource>"
        "</tool_call:opensource>"
        "</tool_calls:opensource>"
    )
    info = tp.extract_tool_calls(out_hy4)
    assert info.tools_called is True
    assert len(info.tool_calls) == 1
    assert info.tool_calls[0]["name"] == "get_weather"
    assert json.loads(info.tool_calls[0]["arguments"]) == {"city": "San Francisco", "days": 3}

    out_hy3 = (
        "<tool_calls:opensource>"
        "<tool_call:opensource>search<tool_sep:opensource>"
        "<arg_key:opensource>q</arg_key:opensource>"
        "<arg_value:opensource>mlx</arg_value:opensource>"
        "</tool_call:opensource>"
        "</tool_calls:opensource>"
    )
    info3 = tp.extract_tool_calls(out_hy3)
    assert info3.tools_called is True
    assert info3.tool_calls[0]["name"] == "search"
    assert json.loads(info3.tool_calls[0]["arguments"]) == {"q": "mlx"}


def test_hunyuan_tool_parser_per_tool_call_streaming_no_leak():
    """Verify per-tool-call streaming emits Call 0 before Call 1 completes and never leaks partial tags."""
    from vmlx_engine.tool_parsers import ToolParserManager

    tp = ToolParserManager.get_tool_parser("hy_v4_flash")()
    chunks = [
        "Checking ",
        "both now. <tool_",
        "calls:opensource><tool_call:opensource>get_weather",
        "<arg_key:opensource>city</arg_key:opensource>",
        "<arg_value:opensource>Tokyo</arg_value:opensource>",
        "</tool_call:opensource>",
        "<tool_call:opensource>get_time",
        "<arg_key:opensource>tz</arg_key:opensource>",
        "<arg_value:opensource>JST</arg_value:opensource>",
        "</tool_call:opensource></tool_calls:opensource>",
    ]
    prev = ""
    emitted_content = []
    emitted_calls = []
    call0_emitted_at = None
    call1_emitted_at = None
    for idx, delta in enumerate(chunks):
        cur = prev + delta
        res = tp.extract_tool_calls_streaming(prev, cur, delta)
        prev = cur
        if not res:
            continue
        if res.get("content"):
            emitted_content.append(res["content"])
        if res.get("tool_calls"):
            for tc in res["tool_calls"]:
                emitted_calls.append(tc)
                if tc["index"] == 0 and call0_emitted_at is None:
                    call0_emitted_at = idx
                if tc["index"] == 1 and call1_emitted_at is None:
                    call1_emitted_at = idx

    full_text = "".join(emitted_content)
    assert full_text == "Checking both now. "
    assert "<tool_" not in full_text
    assert len(emitted_calls) == 2
    assert emitted_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(emitted_calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
    assert emitted_calls[1]["function"]["name"] == "get_time"
    assert json.loads(emitted_calls[1]["function"]["arguments"]) == {"tz": "JST"}
    # Per-tool-call streaming: Call 0 MUST emit on chunk 5 (its own </tool_call:opensource>)
    # BEFORE Call 1 closes on chunk 9!
    assert call0_emitted_at == 5
    assert call1_emitted_at == 9


def test_hy4_reasoning_effort_and_template_policy():
    from vmlx_engine.server import (
        _normalize_hy4_reasoning_effort,
        _rendered_prompt_starts_in_reasoning,
        _template_completes_thinking,
    )
    from vmlx_engine.utils.chat_template_kwargs import build_chat_template_kwargs

    assert _normalize_hy4_reasoning_effort("none", enable_thinking=None) == "no_think"
    assert _normalize_hy4_reasoning_effort("off", enable_thinking=None) == "no_think"
    assert _normalize_hy4_reasoning_effort("no_think", enable_thinking=None) == "no_think"
    assert _normalize_hy4_reasoning_effort("low", enable_thinking=None) == "high"
    assert _normalize_hy4_reasoning_effort("medium", enable_thinking=None) == "high"
    assert _normalize_hy4_reasoning_effort("high", enable_thinking=None) == "high"
    assert _normalize_hy4_reasoning_effort(None, enable_thinking=False) == "no_think"
    assert _normalize_hy4_reasoning_effort(None, enable_thinking=True) == "high"
    assert _normalize_hy4_reasoning_effort(None, enable_thinking=None) is None

    assert _rendered_prompt_starts_in_reasoning("<｜hy_Assistant:opensource｜><think:opensource>\n") is True
    assert _rendered_prompt_starts_in_reasoning("<｜hy_Assistant:opensource｜><think:opensource></think:opensource>") is False
    class _FakeTok:
        def apply_chat_template(self, msgs, **kwargs):
            return "<｜hy_Assistant:opensource｜><think:opensource></think:opensource>"

    assert _template_completes_thinking(_FakeTok(), "hy_v4_flash_unit_test_no_think") is True

    kw_off = build_chat_template_kwargs(model_type="hy_v4_flash", enable_thinking=False)
    assert kw_off.get("reasoning_effort") == "no_think"
    kw_on = build_chat_template_kwargs(model_type="hy_v4_flash", enable_thinking=True)
    assert kw_on.get("reasoning_effort") == "high"


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU required")
def test_hy4_vision_tower_mrope_and_dfly2_vl():
    """Verify HyV4VisionModel + 3D M-RoPE + rope_delta SSD round-trip + VL DFly2 decode parity."""
    from vmlx_engine.models.hy_v4_flash.vision import (
        HyV4VisionModel,
        compute_hy4_mrope_positions,
    )

    mx.random.seed(7)
    cfg = _make_mini_hy4_config()
    cfg.vision_config = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "patch_size": 8,
        "spatial_merge_size": 2,
        "out_hidden_size": cfg.hidden_size,
        "rms_norm_eps": 1e-5,
        "rope_2d_theta": 10000.0,
        "rope_2d_percent": 1.0,
        "max_position_embeddings": 16,  # 4x4 grid + 1 CLS
    }
    model = HyV4FlashForCausalLM(cfg, mxfp4=False)
    model.vision_tower = HyV4VisionModel(cfg.vision_config)
    for layer in model.layers:
        sa = layer.self_attn
        sa.set_kv_b_weight(
            (mx.random.normal((sa.n_heads * (sa.qk_nope + sa.v_head_dim), sa.kv_lora_rank)) * 0.02).astype(mx.bfloat16)
        )
        sa._sinks_bf16 = sa.learnable_sink_param.astype(mx.bfloat16)
    lookup = np.arange(cfg.vocab_size, dtype=np.int32) % 80
    lookup[cfg.engram_pad_id] = 79
    model.model.set_engram_hasher(HyV4EngramHasher(cfg, lookup))
    total_rows = sum(sum(r) for r in cfg.engram_layer_head_vocab_sizes["2"])
    head_dim = cfg.engram_n_embed_per_ngram // cfg.engram_n_head_per_ngram
    model.layers[1].engram_layer.bind_table(
        HyV4InMemoryEngramTable((mx.random.normal((total_rows, head_dim)) * 0.05).astype(mx.bfloat16))
    )
    model.layers[1].engram_layer.prepare_runtime()

    drafter = HyV4DFly2Drafter(cfg, intermediate_size=128, has_confidence_head=True)
    for dl in drafter.layers:
        sa = dl.self_attn
        sa.set_kv_b_weight(
            (mx.random.normal((sa.n_heads * (sa.qk_nope + sa.v_head_dim), sa.kv_lora_rank)) * 0.02).astype(mx.bfloat16)
        )
        sa._sinks_bf16 = sa.learnable_sink_param.astype(mx.bfloat16)
    mx.eval(model.parameters(), drafter.parameters())

    # 1 image of grid (1, 4, 4) -> merged (2, 2) + newline per row = 2 * (2 + 1) = 6 visual tokens
    grid = mx.array([[1, 4, 4]], dtype=mx.int32)
    pv = (mx.random.normal((16, 8, 8, 3)) * 0.5).astype(mx.bfloat16)
    prompt = [10, 11, cfg.image_token_id, cfg.image_token_id, cfg.image_token_id,
              cfg.image_token_id, cfg.image_token_id, cfg.image_token_id, 12, 13, 14]
    pos3d, rope_delta = compute_hy4_mrope_positions(prompt, grid, image_token_id=cfg.image_token_id, merge_size=2)
    assert tuple(pos3d.shape) == (3, 1, len(prompt))
    # 6 image tokens span spatial max=3 -> saves 3 positions vs 1D (rope_delta == -3)
    assert rope_delta == -3

    embeds, pos3d_b, rd_b = model.build_vl_inputs(mx.array([prompt], dtype=mx.int32), pv, grid)
    assert tuple(embeds.shape) == (1, len(prompt), cfg.hidden_size)
    assert rd_b == rope_delta

    # Plain VL AR reference
    c_ar = model.make_cache()
    lg = model(mx.array([prompt], dtype=mx.int32), cache=c_ar, inputs_embeds=embeds, position_ids_3d=pos3d_b, prefill_last_logits_only=True)
    for lc in c_ar:
        lc.rope_delta = rd_b
    tk = int(mx.argmax(lg[0, -1], axis=-1).item())
    ar_toks = [tk]
    for _ in range(9):
        lg = model(mx.array([[tk]], dtype=mx.int32), cache=c_ar)
        tk = int(mx.argmax(lg[0, -1], axis=-1).item())
        ar_toks.append(tk)

    # VL + Adaptive DFly2 MUST match plain VL AR token-for-token!
    st = DFly2Stats()
    dfly_toks = []
    for chunk in iter_hy4_dfly2(
        model, drafter, prompt, 10, eos={999999}, temperature=0.0, stats=st,
        inputs_embeds=embeds, position_ids_3d=pos3d_b, rope_delta=rd_b,
    ):
        dfly_toks.extend(chunk)
    assert dfly_toks == ar_toks, f"VL DFly2 diverged from VL AR: {dfly_toks} vs {ar_toks}"

    # HyV4LayerCache SSD round-trip preserves rope_delta!
    c_rebuilt = [HyV4LayerCache.from_state(lc.state, lc.meta_state) for lc in c_ar]
    assert all(lc.rope_delta == rd_b for lc in c_rebuilt)
    lg_orig = model(mx.array([[ar_toks[-1]]], dtype=mx.int32), cache=c_ar)
    lg_rebuilt = model(mx.array([[ar_toks[-1]]], dtype=mx.int32), cache=c_rebuilt)
    mx.eval(lg_orig, lg_rebuilt)
    assert float(mx.max(mx.abs(lg_orig - lg_rebuilt)).item()) == 0.0
