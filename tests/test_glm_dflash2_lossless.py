"""GLM-5.3-Flash DFlash2 lane (vmlx_engine.glm_dflash2): speculative greedy decoding is lossless up to float noise.

Target: a tiny random glm5_next (KDA + DSA layers, mHC 4 streams, MoE). Its router selects ALL experts (top-k = E):
with a real top-k, a ~1e-4 router margin flips an expert under the different summation order of an 8-row verify
forward versus 1-row decode and the outputs diverge for reasons that have nothing to do with speculation (measured
2026-10-09: margins 1.8e-4 / 2.1e-4 at the diverging rows; every layer's cached chunk / decode path matches a
cache-free forward to ~3e-3). With continuous routing the remaining difference is float noise.

Drafter: SCRIPTED. It proposes the true greedy continuation and corrupts draft j = (round mod block) (j = block-1:
nothing corrupted), so the rounds accept every count from 0 to the full block; each partial rejection must restore
the KDA recurrent/conv state of the accepted prefix (rollback_speculative) and trim the MLA/DSA caches (incl.
incomplete indexer pools; prompt lengths cover every residue mod 4).
Oracles: (1) the speculative tokens equal greedy AR up to the first divergence, and a divergence is only accepted at
a near-tie (AR logit margin < TOL between the two tokens); (2) every emitted token is the argmax of ONE cache-free
forward over prompt + emitted tokens, except near-ties. A rollback bug corrupts every later position.
Plus: the real DFlash2DraftModel (random weights, selector path) runs end to end and meets oracle (2).
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("dflash.model_mlx")

from vmlx_engine.glm_dflash2 import GLMDFlashTarget, SpecStats, generate_ar, generate_dflash2  # noqa: E402
from vmlx_engine.models.glm5_next.glm5_next import Model, ModelArgs  # noqa: E402

VOCAB = 512
TAPS = (1, 3, 4)
TOL = 0.05


def _target(seed=0):
    mx.random.seed(seed)
    args = ModelArgs(
        hidden_size=128, num_hidden_layers=6, vocab_size=VOCAB, num_attention_heads=2, q_lora_rank=64, kv_lora_rank=64,
        qk_nope_head_dim=32, v_head_dim=32, index_n_heads=2, index_head_dim=32, linear_num_heads=2, linear_head_dim=32,
        n_routed_experts=8, num_experts_per_tok=8, moe_intermediate_size=64, intermediate_size=128, first_k_dense_replace=1,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                     "linear_attention", "deepseek_sparse_attention"],
    )
    m = Model(args)
    # Random init leaves the KDA short-conv weights (and A_log / dt_bias) at ZERO: q, k, v would be zero and the
    # recurrent state would never influence the output, so a broken KDA rollback could not be detected (found by
    # mutation test 2026-10-09). Give them real values.
    for layer in m.model.layers:
        att = layer.self_attn
        if hasattr(att, "q_conv1d"):
            for n in ("q_conv1d", "k_conv1d", "v_conv1d"):
                setattr(att, n, mx.random.normal(getattr(att, n).shape) * 0.5)
            att.A_log = mx.random.normal(att.A_log.shape) * 0.5
            att.dt_bias = mx.random.normal(att.dt_bias.shape) * 0.5
    mx.eval(m.parameters())
    return m


class _Cache:
    def __init__(self):
        self.offset = 0

    def trim(self, n):
        self.offset -= n
        return n


class _Cfg:
    def __init__(self, block):
        self.block_size, self.mask_token_id, self.sliding_window = block, VOCAB - 1, 64
        self.layer_types, self.target_layer_ids = ("sliding_attention",), TAPS


class ScriptedDraft:
    """Proposes the true greedy continuation; round r corrupts draft index r % block (index block-1 = none)."""

    def __init__(self, ref, block):
        self.ref, self.config, self.calls, self.n = ref, _Cfg(block), 0, 0

    def make_cache(self):
        return [_Cache()]

    def bind(self, target):
        return self

    def __call__(self, block, hidden, cache, logits_start=1):
        assert hidden.shape[-1] == 128 * len(TAPS), hidden.shape          # mean-over-streams taps, concatenated
        # tokens generated so far: 1 after the prompt, then + the target context rows passed in (= tokens emitted by
        # the previous round: accepted drafts + bonus)
        self.n = 1 if self.calls == 0 else self.n + int(hidden.shape[1])
        bs = block.shape[1]; bad = self.calls % bs
        toks = []
        for j in range(bs - 1):
            i = self.n + j
            t = self.ref[i] if i < len(self.ref) else 0
            if j == bad:
                t = (t + 1) % (VOCAB - 1)
            toks.append(t)
        self.calls += 1
        cache[0].offset += int(hidden.shape[1]) + bs                    # emulate a draft KV cache (context + block rows)
        return mx.eye(VOCAB)[mx.array(toks)][None] * 30.0


def _ar_logits(model, prompt, n):
    c = model.make_cache(); lg = model(mx.array([prompt]), cache=c)[0, -1]; toks, rows = [], []
    for _ in range(n):
        t = int(mx.argmax(lg).item()); toks.append(t); rows.append(np.array(lg.astype(mx.float32)))
        lg = model(mx.array([[t]]), cache=c)[0, -1]
    return toks, rows


def _check_vs_ar(got, ref, rows):
    div = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), None)
    if div is not None:
        margin = float(rows[div][ref[div]] - rows[div][got[div]])
        assert margin < TOL, f"speculative token {got[div]} != AR {ref[div]} at {div} with AR margin {margin:.3f} (not a near-tie)"
    return div


def _check_teacher_forced(model, prompt, got):
    full = model(mx.array([prompt + got]))[0].astype(mx.float32)          # one cache-free forward
    lg = np.array(full[len(prompt) - 1: len(prompt) - 1 + len(got)])
    bad = [(i, float(lg[i].max() - lg[i][t])) for i, t in enumerate(got) if int(lg[i].argmax()) != t]
    # near-ties (margin < TOL) may flip under summation order; a state error makes later tokens wrong by large margins
    assert all(m < TOL for _, m in bad) and len(bad) <= 3, f"emitted tokens that are not the target's argmax: {bad}"


@pytest.mark.parametrize("block", [4, 8])
@pytest.mark.parametrize("plen", [37, 38, 39, 40])
def test_scripted_partial_rejections_are_lossless(block, plen):
    model = _target()
    target = GLMDFlashTarget(model, TAPS)
    try:
        prompt = list(range(5, 5 + plen))
        ref, rows = _ar_logits(model, prompt, 40)
        st = SpecStats()
        got = list(generate_dflash2(target, ScriptedDraft(ref, block), prompt, 40, eos=set(), stats=st))
        div = _check_vs_ar(got, ref, rows)
        _check_teacher_forced(model, prompt, got)
        upto, rounds = (div if div is not None else len(got)), 0
        emitted = 1
        while rounds < len(st.accepted) and emitted < upto:          # rounds that completed before the divergence
            emitted += st.accepted[rounds] + 1; rounds += 1
        seen = set(st.accepted[:max(1, rounds)])
        assert len(seen) >= min(block, 3), f"acceptance counts not exercised: {st.accepted}"
    finally:
        target.close()


def test_real_dflash2_drafter_runs_and_is_target_consistent():
    from dflash.model_mlx import DFlash2DraftModel, DFlashConfig

    model = _target(1)
    cfg = DFlashConfig(hidden_size=128, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                       intermediate_size=256, vocab_size=VOCAB, rms_norm_eps=1e-5, rope_theta=10000.0,
                       max_position_embeddings=4096, block_size=8, target_layer_ids=TAPS, num_target_layers=6,
                       mask_token_id=VOCAB - 1, layer_types=("sliding_attention",), sliding_window=64,
                       conv_kernel_size=2, conv_group_size=16, selector_rank=32, selector_top_k=4, is_causal=False)
    draft = DFlash2DraftModel(cfg); mx.eval(draft.parameters())
    target = GLMDFlashTarget(model, TAPS)
    try:
        prompt = list(range(7, 40))
        got = list(generate_dflash2(target, draft, prompt, 24, eos=set()))
        assert len(got) == 24
        _check_teacher_forced(model, prompt, got)
    finally:
        target.close()


def test_fixture_kda_state_matters():
    """Fail closed: the rollback tests are only meaningful if the KDA recurrent state changes the output."""
    from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache

    m = _target(); prompt = list(range(5, 45))
    c1, c2 = m.make_cache(), m.make_cache()
    m(mx.array([prompt]), cache=c1); m(mx.array([prompt]), cache=c2)
    for x in c2:
        if isinstance(x, Glm5KDACache):
            x.cache = [None if v is None else mx.zeros_like(v) for v in x.cache]
    a, b = m(mx.array([[7]]), cache=c1)[0, -1], m(mx.array([[7]]), cache=c2)[0, -1]
    assert float(mx.abs(a - b).max()) > 0.05 * float(mx.abs(a).max())


def test_taps_are_removed_on_close():
    model = _target(2)
    t = GLMDFlashTarget(model, TAPS)
    t.close()
    assert all(type(l).__name__ != "_MeanStreamTap" for l in model.model.layers)
