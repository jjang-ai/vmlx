# SPDX-License-Identifier: Apache-2.0
"""DSpark on MiniMax-M3 / M3.1: the target adapter (exact verify, rollback), the loop, Adaptive width, serving lane.

Tiny synthetic M3 (no real weights): 4 decoder layers = full attention + dense MLP (layer 0, plain KV cache), sparse
attention + dense MLP (layer 1), sparse attention + MoE (layers 2-3, 8 experts, top-4, shared expert); trunk mixed per
module like an M3.1 JANGHT bundle (MXFP8 q / o, affine 8 k / v, affine 6 indexer and lm_head, affine 4 elsewhere,
affine 8 embeddings, float32 routers). Index blocks of 32 keys with top-6 selection, so a 773-token context has 25
blocks and the MSA selection is active (asserted); one test runs the released geometry (blocks of 128, top-16) past
2,048 tokens. Lengths are chosen NOT to divide by 32 / 64 / 128 / 256 (the K / V step) and verify windows straddle
those boundaries.

Exactness is bit-equality (mx.array_equal), no tolerance: every verify row must produce exactly the logits of the
corresponding one-token decode step. Controls show the checks bite: the stock multi-row forward is not row-exact here,
and a rollback that skips a lane is caught by the cache invariant and changes the following decode steps.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx.utils import tree_flatten  # noqa: E402

from vmlx_engine.dspark import DSparkStats, generate_ar, generate_dspark, load_drafter  # noqa: E402
from vmlx_engine.dspark.drafter import DSparkDrafter  # noqa: E402
from vmlx_engine.dspark.generate import ARFallback  # noqa: E402
from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget  # noqa: E402
from vmlx_engine.models.minimax_m3 import minimax_m3 as M  # noqa: E402

D, I, SH, DENSE, E, TOPK, VOCAB = 512, 256, 256, 512, 8, 4, 1024
HEADS, KVH, HD, ROT, LAYERS = 8, 2, 64, 32, 4
MX8 = {"mode": "mxfp8", "group_size": 32, "bits": 8}
DRAFT_BLOCK, DRAFT_TAPS = 4, (1, 3)


@pytest.fixture(autouse=True)
def _exact_defaults(monkeypatch):
    for name in ("VMLX_ROW_EXACT_VERIFY_QMV", "VMLX_DSPARK", "VMLX_DSPARK_MAX_DRAFT", "VMLX_DSPARK_WIDTH",
                 "VMLX_DSPARK_SAMPLED", "VMLX_DSPARK_PIPELINE", "VMLX_DSPARK_AR_FALLBACK"):
        monkeypatch.delenv(name, raising=False)


# ----------------------------------------------------------------------------------------------- model
def _config(block: int = 32, topk: int = 6) -> dict:
    tc = {"model_type": "minimax_m3", "hidden_size": D, "num_hidden_layers": LAYERS, "intermediate_size": I,
          "dense_intermediate_size": DENSE, "shared_intermediate_size": SH, "num_attention_heads": HEADS,
          "num_key_value_heads": KVH, "head_dim": HD, "rotary_dim": ROT, "rope_theta": 10_000.0, "rms_norm_eps": 1e-6,
          "vocab_size": VOCAB, "num_local_experts": E, "num_experts_per_tok": TOPK, "routed_scaling_factor": 2.0,
          "norm_topk_prob": True, "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "moe_layer_freq": [0, 0, 1, 1],
          "sparse_attention_config": {"sparse_num_index_heads": KVH, "sparse_index_dim": HD, "sparse_block_size": block,
                                      "sparse_topk_blocks": topk, "sparse_local_block": 1,
                                      "sparse_attention_freq": [0, 1, 1, 1]}}
    return {"model_type": "minimax_m3_vl", "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
            "text_config": tc}


def _plan(path, module):
    if path.endswith("mlp.gate"):
        return False                                    # float32 router, as released
    if "switch_mlp" in path:
        return {"group_size": 64, "bits": 4} if hasattr(module, "to_quantized") else False
    if not isinstance(module, (nn.Linear, nn.Embedding)):
        return False
    if path.endswith(("q_proj", "o_proj")) and "index" not in path:
        return MX8
    if path.endswith(("k_proj", "v_proj")) and "index" not in path:
        return {"group_size": 64, "bits": 8}
    if "index_" in path or path == "lm_head":
        return {"group_size": 64, "bits": 6}
    if path.endswith("embed_tokens"):
        return {"group_size": 64, "bits": 8}
    return {"group_size": 64, "bits": 4}


def _model(seed: int = 0, block: int = 32, topk: int = 6, quantize: bool = True):
    model = M.Model(M.ModelArgs.from_dict(_config(block, topk)))
    rng = np.random.default_rng(seed)
    weights = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith("norm.weight"):
            a, dt = rng.standard_normal(v.shape) * 0.1, mx.bfloat16
        elif k.endswith("e_score_correction_bias"):
            a, dt = rng.standard_normal(v.shape) * 0.05, mx.float32
        elif k.endswith(".mlp.gate.weight"):
            a, dt = rng.standard_normal(v.shape) * 0.3, mx.float32
        elif "embed_tokens" in k:
            a, dt = rng.standard_normal(v.shape), mx.bfloat16
        else:
            a, dt = rng.standard_normal(v.shape) * 1.5 / np.sqrt(v.shape[-1]), mx.bfloat16
        weights.append((k, mx.array(a.astype(np.float32)).astype(dt)))
    model.load_weights(weights)
    if quantize:
        nn.quantize(model, class_predicate=_plan)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def target():
    return MiniMaxM3DSparkTarget(_model())


def _ids(n: int, seed: int) -> list:
    return np.random.default_rng(seed).integers(0, VOCAB - 1, n).tolist()


def _eq(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


def _sparse_active(target, cache) -> bool:
    """True when a decode step at the cache's current length selects a strict subset of index blocks."""
    attn = target.nm.layers[1].self_attn
    c = cache[1]
    x = mx.zeros((1, 1, D), dtype=mx.bfloat16)
    idx_q, _ = attn.indexer.project(x, int(c.offset), None)
    return attn.indexer.select(idx_q, c.state[2], int(c.offset)) is not None


# ----------------------------------------------------------------------------------------------- cache invariant
def _cache_problems(cache, ref) -> list:
    """Every difference between two M3 cache lists: lane lengths vs offsets, index-lane offset, logical contents."""
    assert len(cache) == len(ref) == LAYERS
    problems = []
    for li, (a, b) in enumerate(zip(cache, ref)):
        if int(a.offset) != int(b.offset):
            problems.append(f"layer {li}: offset {a.offset} != {b.offset}")
            continue
        for c, tag in ((a, "dspark"), (b, "reference")):
            idx_off = getattr(c, "_idx_offset", None)
            if getattr(c, "idx_keys", None) is not None and int(idx_off) != int(c.offset):
                problems.append(f"layer {li} {tag}: index lane {idx_off} != K/V lane {c.offset}")
        sa, sb = a.state, b.state
        assert len(sa) == len(sb) and len(sa) == (3 if li else 2), (li, len(sa))
        for name, x, y in zip(("keys", "values", "idx_keys"), sa, sb):
            assert x is not None and y is not None and x.size > 0, (li, name)     # fail closed: real content
            if x.shape[2] != int(a.offset) or y.shape[2] != int(b.offset):
                problems.append(f"layer {li} {name}: logical rows {x.shape[2]} / {y.shape[2]} != offset {a.offset}")
            elif not _eq(x, y):
                problems.append(f"layer {li} {name}: contents differ")
    return problems


# ----------------------------------------------------------------------------------------------- ops
def test_row_independent_ops_equal_one_row_calls(target):
    """The ops verify runs on all rows at once are row-independent here (norms, RoPE at offsets, embeddings, router
    top-k, SwiGLU-OAI): checked bit-equal to one-row calls, so only projections / attention / experts need care."""
    layer = target.nm.layers[2]
    rng = np.random.default_rng(3)
    for rows in (2, 5, 9):
        x = mx.array(rng.standard_normal((1, rows, D)).astype(np.float32) * 3).astype(mx.bfloat16)
        one = lambda f, t: mx.concatenate([f(t[:, r:r + 1]) for r in range(rows)], axis=1)  # noqa: E731
        assert _eq(layer.input_layernorm(x), one(layer.input_layernorm, x))
        q = mx.array(rng.standard_normal((1, HEADS, rows, HD)).astype(np.float32)).astype(mx.bfloat16)
        for off in (0, 31, 765, 2047):
            rope = layer.self_attn.rope
            assert _eq(rope(q, offset=off), mx.concatenate(
                [rope(q[:, :, r:r + 1], offset=off + r) for r in range(rows)], axis=2))
        tok = mx.array([rng.integers(0, VOCAB, rows).tolist()])
        assert _eq(target.embed_tokens(tok), one(target.embed_tokens, tok))
        logits = mx.concatenate([layer.mlp.gate(x[:, r:r + 1]) for r in range(rows)], axis=1)
        inds, scores = M.group_expert_select(logits, layer.mlp.e_score_correction_bias, TOPK, 1, 1, 2.0, True)
        for r in range(rows):
            i1, s1 = M.group_expert_select(logits[:, r:r + 1], layer.mlp.e_score_correction_bias, TOPK, 1, 1, 2.0, True)
            assert _eq(inds[:, r:r + 1], i1) and _eq(scores[:, r:r + 1], s1)
        assert _eq(M.swiglu(x, x * 0.5), one(lambda t: M.swiglu(t, t * 0.5), x))


# ----------------------------------------------------------------------------------------------- verify
@pytest.mark.parametrize("prompt_len, width", [(773, 2), (765, 8), (509, 5), (1021, 9), (771, 15), (300, 17)])
def test_verify_rows_equal_sequential_decode_steps(target, prompt_len, width):
    """Row r of a T-row verify == the r-th of T one-token decode steps, bit for bit, at contexts where the MSA
    selection is active, windows that straddle 32 / 256 / 512 / 1,024 boundaries, widths past the drafter block and
    past the MoE sort threshold (17 rows x top-4 = 68 routed pairs: experts one row at a time)."""
    prompt = mx.array([_ids(prompt_len, prompt_len)])
    rows = mx.array([_ids(width, 7 * width)])
    taps = DRAFT_TAPS
    target.verify_paths = {k: 0 for k in target.verify_paths}
    cache = target.make_cache()
    target.prefill(prompt, cache, taps, 256)
    logits, feats, _ = target.verify(rows, cache, taps)
    mx.eval(logits, feats)
    ref = target.make_cache()
    target.prefill(prompt, ref, taps, 256)
    ref_logits, ref_feats = [], []
    for r in range(width):
        if r == 0 and prompt_len > 64:
            assert _sparse_active(target, ref)                 # the per-row MSA selection is exercised
        lg, f = target.ar_step(rows[0, r:r + 1], ref, taps)
        ref_logits.append(lg)
        ref_feats.append(f)
    assert logits.shape == (width, VOCAB) and float(mx.abs(logits).max()) > 0
    bad = [r for r in range(width) if not _eq(logits[r], ref_logits[r])]
    assert not bad, f"verify rows {bad} differ from decode steps"
    assert _eq(feats, mx.concatenate(ref_feats, axis=1))
    assert not _cache_problems(cache, ref)
    paths = target.verify_paths
    assert paths["linear_batched"] > 0 and paths["linear_rows"] > 0      # both projection routes ran
    from vmlx_engine.metal.row_exact_fp_qmv import _ADMITTED as fp_admitted

    assert any(v is not None and k[3] == "mxfp8" for k, v in fp_admitted.items())   # MXFP8 q / o on the fp kernel
    if width * TOPK >= 64:
        assert paths["routed_rows"] > 0 and paths["routed_batched"] == 0
    else:
        assert paths["routed_batched"] > 0 and paths["routed_rows"] == 0


def test_ar_step_equals_decode_and_taps_are_layer_outputs(target):
    """ar_step (the drafter's AR fallback) == the served decode call, and its taps are the residual stream after the
    drafter's target layers, in the drafter's order."""
    prompt = mx.array([_ids(517, 5)])
    a, b = target.make_cache(), target.make_cache()
    target.prefill(prompt, a, (), 256)
    target.prefill(prompt, b, (), 256)
    for t in _ids(4, 6):
        tok = mx.array([t])
        lg, feats = target.ar_step(tok, a, (3, 1))
        assert _eq(lg, target.decode(tok, b))
        assert feats.shape == (1, 1, 2 * D)
    x = target.embed_tokens(mx.array([[7]]))
    c = target.make_cache()
    outs = []
    for layer, lc in zip(target.nm.layers, c):
        x = layer(x, None, lc)
        outs.append(x)
    _, f = target.ar_step(mx.array([7]), target.make_cache(), (3, 1))
    assert _eq(f, mx.concatenate([outs[3], outs[1]], axis=-1))


def test_control_stock_multi_row_forward_is_not_row_exact(target):
    """Control: the model's own multi-row forward (multi-row qmm / MXFP8 / float kernels, one causal SDPA, one
    batched indexer score matmul) does NOT reproduce the decode steps here, so the exactness test above can fail."""
    prompt = mx.array([_ids(765, 765)])
    rows = mx.array([_ids(8, 56)])
    cache = target.make_cache()
    target.prefill(prompt, cache, (), 256)
    stock = target.model(rows, cache=cache)[0]
    ref = target.make_cache()
    target.prefill(prompt, ref, (), 256)
    ref_logits = [target.decode(rows[0, r:r + 1], ref) for r in range(8)]
    assert sum(not _eq(stock[r], ref_logits[r]) for r in range(8)) >= 4


@pytest.fixture(scope="module")
def jangt_model(tmp_path_factory):
    """The JANGHT bundle test's in-memory model: JANGT (trellis) routed experts with outlier columns, MXFP8 / affine
    4-6-8 trunk, float32 routers, 3 layers, index blocks of 8 with top-2."""
    import tests.test_minimax_m3_janght as TJ

    _, _, mem, _, _ = TJ.bundle.__wrapped__(tmp_path_factory)
    return mem


def test_verify_with_jangt_experts_equals_decode_steps(jangt_model):
    """JANGT (trellis) routed experts with outlier columns, mixed MXFP8 / affine trunk (the JANGHT bundle test's
    model): rows on the JANGT decode kernels below its sort threshold (17 x 4 = 68 < 72) and one row at a time
    above it on a NAX GPU (19 x 4 = 76)."""
    import tests.test_minimax_m3_janght as TJ
    from vmlx_engine.jangt.switch import nax_prefill

    tgt = MiniMaxM3DSparkTarget(jangt_model)
    assert any(getattr(layer.mlp.switch_mlp, "is_jangt", False) for layer in tgt.nm.layers if hasattr(layer.mlp, "switch_mlp"))
    prompt = mx.array([np.random.default_rng(5).integers(0, TJ.VOCAB, 61).tolist()])
    for width in (3, 8, 17, 19):
        rows = mx.array([np.random.default_rng(width).integers(0, TJ.VOCAB, width).tolist()])
        tgt.verify_paths = {k: 0 for k in tgt.verify_paths}
        cache = tgt.make_cache()
        tgt.prefill(prompt, cache, (), 16)
        logits, _, _ = tgt.verify(rows, cache, ())
        ref = tgt.make_cache()
        tgt.prefill(prompt, ref, (), 16)
        ref_logits = [tgt.decode(rows[0, r:r + 1], ref) for r in range(width)]
        assert all(_eq(logits[r], ref_logits[r]) for r in range(width)), width
        per_row = width * TJ.TOPK >= 72 and nax_prefill()
        assert (tgt.verify_paths["routed_rows"] > 0) == per_row, (width, tgt.verify_paths)


def test_dspark_greedy_equals_ar_with_jangt_experts(jangt_model, tmp_path):
    """Greedy DSpark == greedy AR on the JANGT model (a matching tiny MiniMax drafter, scripted partial accepts,
    Adaptive widths past the drafter block)."""
    import tests.test_minimax_m3_janght as TJ

    tgt = MiniMaxM3DSparkTarget(jangt_model)
    cfg = _drafter_config()
    cfg["text_config"].update(hidden_size=TJ.D, vocab_size=TJ.VOCAB, dspark_noise_token_id=TJ.VOCAB - 1,
                              dspark_target_layer_ids=[0, 2], sliding_window=16)
    prompt = np.random.default_rng(91).integers(0, TJ.VOCAB - 1, 61).tolist()
    reference = generate_ar(tgt, prompt, 40, set(), prefill_step=16)
    assert len(set(reference)) > 5
    drafter = _Scripted(load_drafter(_save_release_drafter(tmp_path / "dspark", cfg)), prompt + reference,
                        _every_fifth)
    stats = DSparkStats()
    out = generate_dspark(tgt, drafter, prompt, 40, set(), prefill_step=16, stats=stats, max_draft=2 * DRAFT_BLOCK,
                          ar_fallback=_WidthOnly())
    assert out == reference
    assert stats.rounds > 3 and 0 < stats.accepted < stats.drafted and max(stats.widths) - 1 > DRAFT_BLOCK


# ----------------------------------------------------------------------------------------------- rollback
ROLLBACKS = [
    (765, 8, 2),    # rows 765..772 cross 768 (32-block and 256-step boundary); keep 767: back mid-block, below it
    (765, 8, 6),    # keep 771: across 768, mid the next block
    (509, 5, 1),    # rows cross 512; keep 510
    (1021, 9, 4),   # rows cross 1,024; keep 1,025 (mid-block past the boundary)
    (771, 3, 3),    # full accept
]


def _rollback_case(target, prompt_len, width, kept, rollback):
    prompt = mx.array([_ids(prompt_len, prompt_len + 1)])
    rows = mx.array([_ids(width, prompt_len + 2)])
    follow = [mx.array([t]) for t in _ids(3, prompt_len + 3)]
    cache = target.make_cache()
    target.prefill(prompt, cache, (), 256)
    target.verify(rows, cache, ())
    rollback(cache, kept, width)
    ref = target.make_cache()
    target.prefill(prompt, ref, (), 256)
    for r in range(kept):
        target.decode(rows[0, r:r + 1], ref)
    return cache, ref, follow


@pytest.mark.parametrize("prompt_len, width, kept", ROLLBACKS)
def test_rollback_equals_a_cache_that_only_saw_the_accepted_tokens(target, prompt_len, width, kept):
    cache, ref, follow = _rollback_case(target, prompt_len, width, kept,
                                        lambda c, k, t: target.rollback(c, None, kept=k, total=t))
    assert int(cache[0].offset) == prompt_len + kept
    assert not _cache_problems(cache, ref)
    for t in follow:                                   # the next rows overwrite the trimmed tail exactly
        assert _eq(target.decode(t, cache), target.decode(t, ref))
    assert not _cache_problems(cache, ref)


def test_control_rollback_skipping_a_dense_layer_is_caught(target):
    """Control: leave one layer's K / V un-rolled-back (layer 0, plain KV cache): the invariant check reports it and
    the following decode steps change (stale keys are attended)."""
    def broken(cache, kept, total):
        for c in cache[1:]:
            c.trim(total - kept)
    cache, ref, follow = _rollback_case(target, 765, 8, 2, broken)
    assert any("layer 0" in p for p in _cache_problems(cache, ref))
    assert any(not _eq(target.decode(t, cache), target.decode(t, ref)) for t in follow)


def test_control_rollback_skipping_the_index_lane_is_caught(target, monkeypatch):
    """Control: roll back K / V but not the index-key lane of the sparse layers: the invariant check reports the lane
    desync and the next decode step fails instead of silently selecting blocks from stale index keys; the target's
    own lockstep check refuses a cache whose trim leaves the index lane behind."""
    from mlx_lm.models.cache import KVCache

    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

    def broken(cache, kept, total):
        for c in cache:
            KVCache.trim(c, total - kept)              # the K / V half only
    cache, ref, follow = _rollback_case(target, 765, 8, 2, broken)
    assert any("index lane" in p for p in _cache_problems(cache, ref))
    with pytest.raises(ValueError, match="desynchronised"):
        mx.eval(target.decode(follow[0], cache))
    monkeypatch.setattr(MiniMaxM3SparseCache, "trim", KVCache.trim)
    with pytest.raises(RuntimeError, match="index-key lane"):
        _rollback_case(target, 765, 8, 2, lambda c, k, t: target.rollback(c, None, kept=k, total=t))


# ----------------------------------------------------------------------------------------------- drafter
def _drafter_config(block: int = DRAFT_BLOCK, window: int = 64, **extra) -> dict:
    text = {"hidden_size": D, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 64, "rotary_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": VOCAB,
            "dspark_ffn_hidden_size": 256, "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "sliding_window": window,
            "dspark_block_size": block, "rope_theta": 10_000.0, "dspark_noise_token_id": VOCAB - 1,
            "dspark_target_layer_ids": list(DRAFT_TAPS), "dspark_markov_rank": 16,
            "layer_types": ["sliding_attention"] * 2, "dense_intermediate_size": 256, **extra}
    return {"architectures": ["DSparkMiniMaxDraftModel"], "enable_confidence_head": True,
            "confidence_head_with_markov": True, "text_config": text}


PRE = "language_model.model.dspark."


def _save_release_drafter(path, cfg=None, seed: int = 0):
    """A drafter directory in the release layout: language_model.model.dspark.* names, decoder linears MXFP8 (e4m3
    bytes + e8m0 weight_scale_inv), fc / norms / Markov / confidence heads BF16."""
    cfg = cfg or _drafter_config()
    t = cfg["text_config"]
    rng = np.random.default_rng(seed)
    bf = lambda *s, k=0.3: mx.array((rng.standard_normal(s) * k).astype(np.float32)).astype(mx.bfloat16)  # noqa: E731

    def mxfp8(o, i):
        sign = rng.integers(0, 2, (o, i)) << 7
        w = (sign | (rng.integers(5, 9, (o, i)) << 3) | rng.integers(0, 8, (o, i))).astype(np.uint8)
        return mx.array(w), mx.array(rng.integers(118, 121, (o, i // 32)).astype(np.uint8))

    hid, rank, ffn = t["hidden_size"], t["dspark_markov_rank"], t["dspark_ffn_hidden_size"]
    heads, kv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    w = {PRE + "fc.weight": bf(hid, len(t["dspark_target_layer_ids"]) * hid, k=0.05),
         PRE + "hidden_norm.weight": bf(hid, k=0.1), PRE + "final_norm.weight": bf(hid, k=0.1),
         PRE + "markov_head.markov_w1.weight": bf(t["vocab_size"], rank),
         PRE + "markov_head.markov_w2.weight": bf(t["vocab_size"], rank),
         PRE + "confidence_head.proj.weight": bf(1, hid + rank), PRE + "confidence_head.proj.bias": bf(1)}
    for L in range(t["num_hidden_layers"]):
        p = f"{PRE}layers.{L}.decoder_layer."
        for n in ("input_layernorm", "post_attention_layernorm"):
            w[p + n + ".weight"] = bf(hid, k=0.1)
        for n in ("self_attn.q_norm", "self_attn.k_norm"):
            w[p + n + ".weight"] = bf(hd, k=0.1)
        for n, o, i in (("self_attn.q_proj", heads * hd, hid), ("self_attn.k_proj", kv * hd, hid),
                        ("self_attn.v_proj", kv * hd, hid), ("self_attn.o_proj", hid, heads * hd),
                        ("mlp.gate_proj", ffn, hid), ("mlp.up_proj", ffn, hid), ("mlp.down_proj", hid, ffn)):
            w[p + n + ".weight"], w[p + n + ".weight_scale_inv"] = mxfp8(o, i)
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path / "model.safetensors"), w, metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps(cfg, indent=1))
    return path


@pytest.fixture(scope="module")
def release_drafter_dir(tmp_path_factory):
    return _save_release_drafter(tmp_path_factory.mktemp("m31_dspark") / "dspark")


class _Scripted:
    """A real MiniMax drafter whose proposals are replaced by a script over the AR reference sequence: draft at
    absolute position p = reference[p] unless ``wrong(p)`` (then a different token). The drafter's own forward and
    Markov draft still run every round (same context / block / sliding-window path)."""

    def __init__(self, inner: DSparkDrafter, reference: list, wrong):
        self.inner, self.reference, self.wrong = inner, reference, wrong

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, block):
        return self.inner(block)

    def draft(self, hidden, anchor, lm_head, count, sampler=None, candidates=None):
        _, conf = self.inner.draft(hidden, anchor, lm_head, count, sampler=sampler)
        start = self.inner.ctx_len + 1                 # the anchor sits at ctx_len; draft j is position start + j
        toks = []
        for p in range(start, start + count):
            tok = self.reference[p] if p < len(self.reference) else 0
            toks.append((tok + 1) % VOCAB if self.wrong(p) else tok)
        return mx.array(toks, dtype=anchor.dtype), conf


def _every_fifth(p):
    return p % 5 == 4


PROMPTS = [(773, 11), (517, 12), (1029, 13)]


class _WidthOnly(ARFallback):
    """The Adaptive width chooser (per-position acceptance history) without the wall-clock AR fallback, so a test does
    not depend on timings (on this tiny model plain decode steps are cheaper than rounds and the AR fallback would
    take over; the full policy runs in the AR-interleaving case below and has its own tests)."""

    def next_is_ar(self) -> bool:
        return False


@pytest.mark.parametrize("prompt_len, seed", PROMPTS)
@pytest.mark.parametrize("mode", ["adaptive", "adaptive+ar", 2, 7])
def test_dspark_greedy_equals_ar_with_partial_accepts(target, release_drafter_dir, prompt_len, seed, mode):
    """Greedy DSpark == greedy AR token for token: Adaptive widths, Adaptive with the measured AR fallback
    interleaving plain decode steps, and pinned widths (2; 7 > the drafter's trained block of 4); drafts wrong at every
    fifth position (full, partial and zero accepts); generation crossing 32-key block boundaries with the MSA
    selection active."""
    prompt = _ids(prompt_len, seed)
    reference = generate_ar(target, prompt, 45, set(), prefill_step=256)
    assert len(reference) == 45 and len(set(reference)) > 5        # a non-degenerate baseline
    drafter = _Scripted(load_drafter(release_drafter_dir), prompt + reference, _every_fifth)
    stats = DSparkStats()
    width = mode if isinstance(mode, int) else None
    policy = {"adaptive": _WidthOnly(), "adaptive+ar": ARFallback()}.get(mode)
    out = generate_dspark(target, drafter, prompt, 45, set(), prefill_step=256, fixed_width=width, stats=stats,
                          max_draft=2 * DRAFT_BLOCK, ar_fallback=policy)
    assert out == reference
    assert stats.rounds >= 1
    if mode != "adaptive+ar":
        assert stats.rounds > 3 and 0 < stats.accepted < stats.drafted
    if width is not None:
        assert max(stats.widths) == width + 1


def test_control_stock_verify_breaks_greedy_equality(target, release_drafter_dir):
    """Control for the greedy test: the same loop with the model's stock multi-row forward as the verify (logits of
    every row from one multi-row call) does not reproduce AR on these prompts -- the near-ties of this tiny model turn
    the multi-row rounding into different tokens within a few positions."""
    def stock(rows, cache, taps):
        logits, feats = target._stock(rows, cache, taps, logits=True)
        return logits[0], feats, None

    diverged = 0
    for prompt_len, seed in PROMPTS:
        prompt = _ids(prompt_len, seed)
        reference = generate_ar(target, prompt, 45, set(), prefill_step=256)
        drafter = _Scripted(load_drafter(release_drafter_dir), prompt + reference, _every_fifth)
        target.verify = stock
        try:
            out = generate_dspark(target, drafter, prompt, 45, set(), prefill_step=256, fixed_width=7,
                                  max_draft=2 * DRAFT_BLOCK)
        finally:
            del target.verify
        diverged += out != reference
    assert diverged == len(PROMPTS)


def test_dspark_greedy_equals_ar_with_the_real_drafter(target, release_drafter_dir, tmp_path):
    """The real (synthetic-weight) MiniMax drafter as released (MXFP8) and pre-quantized (affine 4 / 8): its own
    drafts, whatever they are, never change the output."""
    pytest.importorskip("jang_tools.format.aligned_safetensors")
    from vmlx_engine.dspark import write_quantized_minimax_drafter

    quantized = write_quantized_minimax_drafter(release_drafter_dir, tmp_path / "q")
    prompt = _ids(773, 21)
    reference = generate_ar(target, prompt, 30, set(), prefill_step=256)
    for path in (release_drafter_dir, quantized):
        stats = DSparkStats()
        out = generate_dspark(target, load_drafter(path), prompt, 30, set(), prefill_step=256, stats=stats,
                              max_draft=2 * DRAFT_BLOCK, ar_fallback=ARFallback())
        assert out == reference and stats.rounds > 0


def test_released_geometry_blocks_of_128_top_16_past_2048_tokens(release_drafter_dir):
    """M3.1's own MSA geometry (blocks of 128 keys, top-16, local block): a 2,429-token prompt (19 blocks; selection
    active) whose generation crosses 2,432 (a 128 multiple); verify rows exact, rollback mid-block exact, greedy
    DSpark == AR."""
    tgt = MiniMaxM3DSparkTarget(_model(seed=4, block=128, topk=16))
    prompt = _ids(2429, 31)
    pa = mx.array([prompt])
    rows = mx.array([_ids(8, 32)])
    cache = tgt.make_cache()
    tgt.prefill(pa, cache, (), 1024)
    assert _sparse_active(tgt, cache)
    logits, _, _ = tgt.verify(rows, cache, ())
    tgt.rollback(cache, None, kept=2, total=8)                      # 2,431: mid-block, below 2,432
    ref = tgt.make_cache()
    tgt.prefill(pa, ref, (), 1024)
    ref_logits = [tgt.decode(rows[0, r:r + 1], ref) for r in range(2)]
    assert all(_eq(logits[r], ref_logits[r]) for r in range(2))
    assert not _cache_problems(cache, ref)
    reference = generate_ar(tgt, prompt, 24, set(), prefill_step=1024)
    drafter = _Scripted(load_drafter(release_drafter_dir), prompt + reference, _every_fifth)
    stats = DSparkStats()
    # Exercise partial acceptance deterministically. The timing governor can take
    # AR after the first fully accepted block, before the scripted rejection.
    assert generate_dspark(tgt, drafter, prompt, 24, set(), prefill_step=1024, stats=stats,
                           max_draft=2 * DRAFT_BLOCK, ar_fallback=_WidthOnly()) == reference
    assert 0 < stats.accepted < stats.drafted


def test_prefill_keeps_only_the_drafter_window_of_tap_rows(target, release_drafter_dir):
    """A sliding-window drafter never reads context rows further back than its window, so the prompt's tap features
    are kept only for the last ``window`` rows (exactly those rows of the full set) and the drafter context starts
    there; the output still equals AR."""
    prompt = mx.array([_ids(773, 81)])
    full_cache, kept_cache = target.make_cache(), target.make_cache()
    lg_full, full = target.prefill(prompt, full_cache, DRAFT_TAPS, 256)
    lg_kept, kept = target.prefill(prompt, kept_cache, DRAFT_TAPS, 256, keep_last=64)
    assert full.shape[1] == 773 and kept.shape[1] == 64
    assert _eq(kept, full[:, -64:]) and _eq(lg_full, lg_kept) and not _cache_problems(kept_cache, full_cache)
    drafter = load_drafter(release_drafter_dir)
    assert drafter.sliding_window == 64
    ids = _ids(773, 82)
    reference = generate_ar(target, ids, 20, set(), prefill_step=256)
    session = {}
    from vmlx_engine.dspark import iter_dspark

    out = []
    for chunk in iter_dspark(target, drafter, ids, 20, set(), prefill_step=256, max_draft=8, session=session):
        out.extend(chunk)
        if len(out) == 1:
            assert drafter.ctx_base == 773 - 64 and drafter.ctx_len == 773
    assert out == reference and drafter.ctx_len == len(session["forwarded"])


def test_dspark_stops_at_eos_like_ar(target, release_drafter_dir):
    prompt = _ids(517, 41)
    free = generate_ar(target, prompt, 30, set(), prefill_step=256)
    eos = {free[11]}
    reference = generate_ar(target, prompt, 30, eos, prefill_step=256)
    drafter = _Scripted(load_drafter(release_drafter_dir), prompt + free, lambda p: False)
    assert generate_dspark(target, drafter, prompt, 30, eos, prefill_step=256, fixed_width=7) == reference


# ----------------------------------------------------------------------------------------------- Adaptive
def test_adaptive_width_climbs_past_the_trained_block_and_shrinks(target, release_drafter_dir):
    """Adaptive changes the width over rounds: perfect drafts for the first 50 positions climb past the drafter's
    trained block (4) towards the ladder top (8); all-wrong drafts afterwards shrink it to one draft. No stamp caps
    it; the output still equals AR."""
    prompt = _ids(517, 51)
    reference = generate_ar(target, prompt, 90, set(), prefill_step=256)
    switch = len(prompt) + 50
    drafter = _Scripted(load_drafter(release_drafter_dir), prompt + reference, lambda p: p >= switch)
    stats = DSparkStats()
    widths = []
    real_verify = target.verify

    def spy(rows, cache, taps):
        widths.append((int(cache[0].offset), int(rows.shape[1])))
        return real_verify(rows, cache, taps)

    target.verify = spy
    try:
        out = generate_dspark(target, drafter, prompt, 90, set(), prefill_step=256, stats=stats,
                              max_draft=2 * DRAFT_BLOCK, ar_fallback=_WidthOnly())
    finally:
        del target.verify
    assert out == reference
    early = [w for pos, w in widths if pos < switch]
    late = [w for pos, w in widths if pos >= switch + 8]
    assert early and late
    assert max(early) - 1 > DRAFT_BLOCK                # drafted past the trained block
    assert len({w for _, w in widths}) >= 3            # the width moved over rounds
    assert max(late) <= 2                             # one draft + the anchor row
