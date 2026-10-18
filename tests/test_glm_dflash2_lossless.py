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


def _target(seed=0, linear_head_dim=32):
    mx.random.seed(seed)
    args = ModelArgs(
        hidden_size=128, num_hidden_layers=6, vocab_size=VOCAB, num_attention_heads=2, q_lora_rank=64, kv_lora_rank=64,
        qk_nope_head_dim=32, v_head_dim=32, index_n_heads=2, index_head_dim=32, linear_num_heads=2,
        linear_head_dim=linear_head_dim,
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


@pytest.mark.parametrize("block", [8, 12])
def test_wide_verify_replay_rollback_is_lossless_and_identical(block, monkeypatch):
    """K=128 KDA heads run the fused verify kernels (the K=32 fixture above takes the vectorized path). Slabs of 8+
    rows write replay records and roll back by replaying accepted rows: the emitted stream must be the target's greedy
    stream AND token-identical to the per-row-state kernel under scripted partial rejections at every draft index."""
    from vmlx_engine.metal import glm5_kda_fused_decode as F
    monkeypatch.setenv("VMLX_GLM5_FUSED_KDA_DECODE", "1")
    model = _target(7, linear_head_dim=128)
    model.set_dtype(mx.bfloat16)                                       # the fused kernels take bf16/fp16 activations
    mx.eval(model.parameters())
    prompt = list(range(5, 5 + 41))
    ref, rows = _ar_logits(model, prompt, 40)
    import vmlx_engine.models.glm5_next.glm5_next as G
    records_calls = []
    real = G.glm5_kda_fused_verify_records

    def counting(*args, **kwargs):
        out = real(*args, **kwargs)
        records_calls.append(out is not None)
        return out
    monkeypatch.setattr(G, "glm5_kda_fused_verify_records", counting)
    got = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("VMLX_GLM5_KDA_VERIFY_REPLAY", mode)
        target = GLMDFlashTarget(model, TAPS)
        try:
            n0, r0 = F._OBSERVED, sum(records_calls)
            st = SpecStats()
            got[mode] = list(generate_dflash2(target, ScriptedDraft(ref, block), prompt, 40, eos=set(), stats=st))
            assert F._OBSERVED > n0                                    # fail closed: fused verify kernels ran
            assert (sum(records_calls) > r0) == (mode == "1")         # ... and the records kernel exactly when on
            assert len(set(st.accepted)) >= 3, st.accepted              # several partial-acceptance counts exercised
        finally:
            target.close()
    assert got["1"] == got["0"]
    _check_vs_ar(got["1"], ref, rows)
    _check_teacher_forced(model, prompt, got["1"])


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


class AlwaysWrongDraft(ScriptedDraft):
    """Every draft is wrong: a cost-aware policy must stop paying for verify rounds."""

    def __call__(self, block, hidden, cache, logits_start=1):
        out = super().__call__(block, hidden, cache, logits_start)
        return mx.roll(out, 1, axis=-1)


@pytest.mark.parametrize("plen", [37, 39])
def test_adaptive_is_lossless(plen):
    model = _target()
    target = GLMDFlashTarget(model, TAPS)
    try:
        prompt = list(range(5, 5 + plen))
        ref, rows = _ar_logits(model, prompt, 40)
        st = SpecStats()
        got = list(generate_dflash2(target, ScriptedDraft(ref, 8), prompt, 40, eos=set(), stats=st, adaptive=True))
        assert len(got) == 40
        _check_vs_ar(got, ref, rows)
        _check_teacher_forced(model, prompt, got)
        assert st.ar_steps + sum(a_ + 1 for a_ in st.accepted) + 1 >= 40
    finally:
        target.close()


def test_adaptive_falls_back_to_ar_when_drafts_fail():
    model = _target(4)
    target = GLMDFlashTarget(model, TAPS)
    try:
        prompt = list(range(5, 45))
        ref, rows = _ar_logits(model, prompt, 64)
        st = SpecStats()
        got = list(generate_dflash2(target, AlwaysWrongDraft(ref, 8), prompt, 64, eos=set(), stats=st, adaptive=True))
        _check_teacher_forced(model, prompt, got)
        assert sum(st.accepted) == 0                         # no draft was ever accepted
        assert st.ar_steps > 2 * st.rounds, (st.ar_steps, st.rounds)   # mostly plain steps, occasional probes
    finally:
        target.close()


def test_adaptive_policy_prefers_large_blocks_for_good_drafts():
    from vmlx_engine.glm_dflash2 import AdaptiveSpec
    p = AdaptiveSpec(8)
    p.update_ar(0.030, 8)
    for _ in range(30):
        p.update_spec(8, 7, 0.060)                           # all 7 drafts accepted, round = 2x a step
    mode, b = p.choose()
    assert mode == "spec" and b == 8
    q = AdaptiveSpec(8)
    q.update_ar(0.030, 8)
    for _ in range(30):
        q.update_spec(4, 0, 0.060)                           # nothing accepted
    q.ar_since_probe = 0
    assert q.choose()[0] == "ar"


@pytest.mark.parametrize("adaptive", [False, True])
def test_prefix_store_reuse_is_lossless(adaptive):
    """A follow-up prompt that extends a stored prompt restores the KDA/MLA state and prefills only the suffix; its
    tokens must equal a cold run. An identical repeat restores the whole prompt. The store must not be corrupted by
    the generations that ran on restored copies (third request checks the first prompt again)."""
    from vmlx_engine.glm_dflash2 import PrefixStore
    model = _target(5)
    target = GLMDFlashTarget(model, TAPS)

    def fresh_policy():
        # The adaptive policy's measured costs are shared by every request on a target (f1e5345f), so the AR / verify
        # schedule of a later request differs from the first one's, and verify vs AR logits round differently. This
        # test is about the prefix store, so each generation starts from the same (empty) cost table.
        target.__dict__.pop("_vmlx_adaptive_costs", None)

    try:
        p1 = list(range(5, 42)); p2 = p1 + list(range(60, 77))          # 37 then 54 tokens (not multiples of 4/64)
        cold = {}
        for name, p in (("p1", p1), ("p2", p2)):
            ref, _ = _ar_logits(model, p, 24)
            fresh_policy()
            cold[name] = list(generate_dflash2(target, ScriptedDraft(ref, 4), p, 24, eos=set(), adaptive=adaptive))
        store = PrefixStore(4)
        runs = []
        for name, p in (("p1", p1), ("p2", p2), ("p1", p1)):
            ref, _ = _ar_logits(model, p, 24)
            fresh_policy()
            st = SpecStats()
            got = list(generate_dflash2(target, ScriptedDraft(ref, 4), p, 24, eos=set(), stats=st, adaptive=adaptive,
                                        prefix_store=store))
            runs.append((name, got, st.cached_tokens))
        assert runs[0][2] == 0 and runs[1][2] == len(p1) and runs[2][2] == len(p1)
        for name, got, _ in runs:
            assert got == cold[name], (name, got, cold[name])
        assert store.hits == 2 and store.misses == 1
    finally:
        target.close()


def test_prefix_store_boundary_snapshot_partial_hit(monkeypatch):
    """Same long prefix, different tail: restores the last 16-token boundary snapshot (shrunk from 2048 for the tiny
    model) before the divergence, prefills the rest; tokens equal a cold run."""
    import vmlx_engine.glm_dflash2 as GD
    monkeypatch.setattr(GD, "PREFIX_SNAPSHOT_EVERY", 16)
    model = _target(6)
    target = GLMDFlashTarget(model, TAPS)
    try:
        common = list(range(5, 45))                                     # 40 tokens: boundaries at 16 and 32
        pa, pb = common + [70, 71, 72], common + [80, 81, 82, 83, 84]
        refb, _ = _ar_logits(model, pb, 16)
        cold = list(generate_dflash2(target, ScriptedDraft(refb, 4), pb, 16, eos=set()))
        store = GD.PrefixStore(12)
        refa, _ = _ar_logits(model, pa, 16)
        list(generate_dflash2(target, ScriptedDraft(refa, 4), pa, 16, eos=set(), prefix_store=store))
        st = SpecStats()
        got = list(generate_dflash2(target, ScriptedDraft(refb, 4), pb, 16, eos=set(), stats=st, prefix_store=store))
        assert st.cached_tokens == 32 and got == cold
    finally:
        target.close()


def test_prefix_store_byte_cap_evicts_lru():
    from vmlx_engine.glm_dflash2 import PrefixStore
    model = _target(7)
    target = GLMDFlashTarget(model, TAPS)
    try:
        store = PrefixStore(12, max_bytes=1)                        # any second snapshot exceeds the cap
        for p in (list(range(5, 30)), list(range(40, 70))):
            ref, _ = _ar_logits(model, p, 4)
            list(generate_dflash2(target, ScriptedDraft(ref, 4), p, 4, eos=set(), prefix_store=store))
        assert len(store.entries) == 1 and store.evictions == 1
        assert next(iter(store.entries)) == tuple(range(40, 70))  # the most recent prompt survives
    finally:
        target.close()


def _run(model, target, prompt, n, store, **kw):
    ref, _ = _ar_logits(model, prompt, n)
    st = SpecStats()
    got = list(generate_dflash2(target, ScriptedDraft(ref, 4), prompt, n, eos=set(), stats=st, prefix_store=store, **kw))
    return got, st


def test_prefix_disk_tier_restores_after_restart(tmp_path):
    """SSD tier: a NEW store over the same directory (a restarted server) restores the longest stored prefix from disk
    and prefills only the suffix; tokens equal a cold run, and the restored state is not shared with the file."""
    from vmlx_engine.glm_dflash2 import PrefixDiskTier, PrefixStore
    model = _target(8)
    target = GLMDFlashTarget(model, TAPS)
    try:
        p1 = list(range(5, 42)); p2 = p1 + list(range(60, 77))
        cold, _ = _run(model, target, p2, 20, None)
        first = PrefixStore(4, disk=PrefixDiskTier(tmp_path, 2 ** 30))
        _run(model, target, p1, 20, first)
        first.disk.flush()
        assert first.disk.writes == 1 and len(list(first.disk.dir.glob("*.safetensors"))) == 1
        first.disk.close()
        second = PrefixStore(4, disk=PrefixDiskTier(tmp_path, 2 ** 30))      # restart: empty RAM, same directory
        got, st = _run(model, target, p2, 20, second)
        assert st.cached_tokens == len(p1) and second.disk.hits == 1 and st.cache_source == "disk"
        _, st_ram = _run(model, target, p2, 20, second)                       # the p2 snapshot `second` stored
        # SSD caching (default): the RAM copy is dropped once its SSD copy is durable, so the repeat restores from SSD;
        # VMLX_SSD_ONLY_RAM_RELEASE=0 keeps the old RAM tier.
        from vmlx_engine.glm_dflash2 import ssd_only_ram_release_enabled
        assert st_ram.cached_tokens == len(p2)
        assert st_ram.cache_source == ("disk" if ssd_only_ram_release_enabled() else "ram")
        assert got == cold
        second.disk.close()
        third = PrefixStore(4, disk=PrefixDiskTier(tmp_path, 2 ** 30))
        again, st3 = _run(model, target, p2, 20, third)                      # the p2 snapshot written by `second`
        assert again == cold
    finally:
        for name in ("first", "second", "third"):
            if name in locals():
                locals()[name].disk.close()
        target.close()


def test_prefix_disk_tier_corrupt_file_is_a_miss(tmp_path):
    from vmlx_engine.glm_dflash2 import PrefixDiskTier, PrefixStore
    model = _target(9)
    target = GLMDFlashTarget(model, TAPS)
    try:
        p1 = list(range(5, 42)); p2 = p1 + list(range(60, 70))
        cold, _ = _run(model, target, p2, 12, None)
        first = PrefixStore(4, disk=PrefixDiskTier(tmp_path, 2 ** 30))
        _run(model, target, p1, 12, first); first.disk.flush()
        f = next(first.disk.dir.glob("*.safetensors"))
        first.disk.close()
        f.write_bytes(f.read_bytes()[: f.stat().st_size // 2])              # truncated file
        second = PrefixStore(4, disk=PrefixDiskTier(tmp_path, 2 ** 30))
        got, st = _run(model, target, p2, 12, second)
        assert st.cached_tokens == 0 and got == cold and not f.exists() and second.disk.errors == 1
    finally:
        for name in ("first", "second"):
            if name in locals():
                locals()[name].disk.close()
        target.close()


def test_prefix_disk_tier_budget_keeps_newest(tmp_path):
    from vmlx_engine.glm_dflash2 import PrefixDiskTier, PrefixStore
    model = _target(10)
    target = GLMDFlashTarget(model, TAPS)
    try:
        disk = PrefixDiskTier(tmp_path, 2 ** 30)
        store = PrefixStore(4, disk=disk)
        _run(model, target, list(range(5, 30)), 4, store); disk.flush()
        assert store.last_persistence.result()["durable"] is True
        one = disk.nbytes()
        disk.close()
        disk = PrefixDiskTier(tmp_path, one * 1.5)  # New serving owner with a smaller shared cap.
        store.disk = disk
        _run(model, target, list(range(40, 70)), 4, store); disk.flush()
        assert len(disk.index) == 1
        assert disk._catalog.budget.refresh_health().capacity_evicted_entries_total >= 1
        assert next(iter(disk.index.values()))["n"] == 30
        assert len(list(disk.dir.glob("*.safetensors"))) == 1
        disk.close()
        disk = PrefixDiskTier(tmp_path, 1)
        store.disk = disk
        _run(model, target, list(range(80, 110)), 4, store)
        outcome = store.last_persistence.result(timeout=5)
        assert outcome["outcome"] == "failed" and outcome["durable"] is False
        assert outcome["retained_tokens"] == 0
    finally:
        if "disk" in locals():
            disk.close()
        target.close()


def test_prefix_disk_fingerprint_tracks_bundle_metadata(tmp_path, monkeypatch):
    import vmlx_engine.glm_dflash2 as GD
    monkeypatch.delenv("VMLX_GLM_DFLASH2_DISK_DIR", raising=False)
    monkeypatch.delenv("VMLX_GLM_DFLASH2_DISK_MAX_GB", raising=False)
    b = tmp_path / "bundle"; b.mkdir()
    (b / "config.json").write_text('{"model_type": "glm5_next"}')
    GD.configure_prefix_disk(True, tmp_path / "blocks", 0, b)
    r1 = GD._PREFIX_DISK["model_key"]
    assert GD._PREFIX_DISK["root"] == str(tmp_path / "blocks") and GD._PREFIX_DISK["max_bytes"] == float("inf")
    (b / "config.json").write_text('{"model_type": "glm5_next", "x": 1}')
    GD.configure_prefix_disk(True, tmp_path / "blocks", 5, b)
    assert GD._PREFIX_DISK["model_key"] != r1 and GD._PREFIX_DISK["max_bytes"] == 5 * 2 ** 30
    GD.configure_prefix_disk(False, tmp_path / "blocks", 5, b)
    assert GD._PREFIX_DISK["root"] is None


def test_prefix_disk_identity_tracks_selected_drafter_and_runtime(tmp_path, monkeypatch):
    from pathlib import Path

    import vmlx_engine.glm_dflash2 as GD
    import vmlx_engine.prefix_cache as PC

    monkeypatch.delenv("VMLX_GLM_DFLASH2_DISK_DIR", raising=False)
    monkeypatch.delenv("VMLX_GLM_DFLASH2_DISK_MAX_GB", raising=False)
    target = tmp_path / "target"
    draft = tmp_path / "external drafter"
    target.mkdir(); draft.mkdir()
    (target / "config.json").write_text('{"model_type":"glm5_next"}')
    config = draft / "config.json"
    config.write_text('{"target_layer_ids":[5,14,24,33,42]}')
    monkeypatch.setattr(PC, "runtime_cache_fingerprint", lambda: "runtime-a")

    def root():
        GD.configure_prefix_disk(True, tmp_path / "blocks", 5, target, draft)
        disk = GD.PrefixDiskTier(GD._PREFIX_DISK["root"], GD._PREFIX_DISK["max_bytes"], GD._PREFIX_DISK["model_key"])
        try:
            return disk.dir
        finally:
            disk.close()

    first = root()
    assert root() == first  # Same bundles/runtime remain restart-compatible.
    old = Path(first)
    old.mkdir(parents=True, exist_ok=True)
    sentinel = old / "retained-old-entry"
    sentinel.write_text("preserve")
    config.write_text('{"target_layer_ids":[6,15,25,34,43],"revision":2}')
    second = root()
    assert second != first
    monkeypatch.setattr(PC, "runtime_cache_fingerprint", lambda: "runtime-b")
    assert root() != second
    assert sentinel.read_text() == "preserve"


def test_prefix_disk_identity_separates_external_drafter_paths(tmp_path):
    import vmlx_engine.glm_dflash2 as GD

    target = tmp_path / "target"
    target.mkdir()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir(); second.mkdir()
    for draft in (first, second):
        (draft / "config.json").write_text('{"target_layer_ids":[5,14]}')
    assert GD._bundle_fingerprint(target, first) != GD._bundle_fingerprint(target, second)


def test_stable_root_snapshot_restores_the_whole_system_turn(monkeypatch):
    """A stop at the stable root (end of the system turn, not a 16-boundary) lets a new chat with the same system prefix
    restore all of it: 21 cached tokens instead of the 16-boundary; tokens equal a cold run."""
    import vmlx_engine.glm_dflash2 as GD
    monkeypatch.setattr(GD, "PREFIX_SNAPSHOT_EVERY", 16)
    model = _target(6)
    target = GLMDFlashTarget(model, TAPS)
    try:
        system = list(range(5, 26))                                     # 21-token "system turn"
        pa, pb = system + [70, 71, 72, 73], system + [80, 81, 82, 83, 84, 85]
        refb, _ = _ar_logits(model, pb, 16)
        cold = list(generate_dflash2(target, ScriptedDraft(refb, 4), pb, 16, eos=set()))
        store = GD.PrefixStore(12)
        refa, _ = _ar_logits(model, pa, 16)
        list(generate_dflash2(target, ScriptedDraft(refa, 4), pa, 16, eos=set(), prefix_store=store, stable_root=21))
        assert tuple(system) in store.entries
        st = SpecStats()
        got = list(generate_dflash2(target, ScriptedDraft(refb, 4), pb, 16, eos=set(), stats=st, prefix_store=store,
                                    stable_root=21))
        assert st.cached_tokens == 21 and got == cold
    finally:
        target.close()


def test_glm_stable_root_finder():
    from types import SimpleNamespace

    from vmlx_engine.glm_dflash2 import _glm_stable_root

    tok = SimpleNamespace(unk_token_id=0, convert_tokens_to_ids=lambda t: 7 if t == "<|user|>" else 0)
    assert _glm_stable_root(tok, [1] * 300 + [7, 2, 3]) == 300
    assert _glm_stable_root(tok, [1] * 100 + [7]) == 0                # short system turn: 2048 boundaries suffice
    assert _glm_stable_root(tok, [1] * 300) == 0                      # no user turn
    bad = SimpleNamespace(unk_token_id=0, convert_tokens_to_ids=lambda t: 0)
    assert _glm_stable_root(bad, [1] * 300 + [0]) == 0                # unknown token


@pytest.mark.parametrize("absorbed", [True, False])
def test_prefix_disk_delta_chain_restores_lossless_after_restart(tmp_path, monkeypatch, absorbed):
    """SSD-only + SSD tier (2026-10-12): a 70-token prompt snapshots at 16/32/48/64 and its end; every snapshot after
    the first is a DELTA (only its new MLA rows) of the previous one. A restarted store restores the whole chain
    (root first, token ids checked per link) and the tokens equal a cold run; the delta files are smaller than the
    root; a broken chain (parent removed) is a miss, never wrong state."""
    import vmlx_engine.glm_dflash2 as GD
    from vmlx_engine.models.glm5_next import glm5_next as G5
    monkeypatch.setattr(GD, "PREFIX_SNAPSHOT_EVERY", 16)
    monkeypatch.setenv("VMLX_SSD_ONLY_RAM_RELEASE", "1")
    monkeypatch.setattr(G5, "glm5_mla_absorb_enabled", lambda: absorbed)
    model = _target(11)
    target = GLMDFlashTarget(model, TAPS)
    try:
        prompt = list(range(5, 75))
        follow = prompt + list(range(90, 99))
        cold, _ = _run(model, target, follow, 12, None)
        first = GD.PrefixStore(4, disk=GD.PrefixDiskTier(tmp_path, 2 ** 30))
        _run(model, target, prompt, 12, first)
        first.disk.flush()
        assert first.delta_writes >= 3 and not first.entries            # no RAM copies under SSD-only
        files = sorted(first.disk.dir.glob("*.safetensors"), key=lambda p: p.stat().st_size)
        assert len(files) >= 4 and files[0].stat().st_size < files[-1].stat().st_size
        first.disk.close()
        second = GD.PrefixStore(4, disk=GD.PrefixDiskTier(tmp_path, 2 ** 30))   # restart
        got, st = _run(model, target, follow, 12, second)
        assert st.cached_tokens == len(prompt) and st.cache_source == "disk" and got == cold
        # break the chain: remove the root (the oldest, shortest snapshot); the longest is now unrestorable
        root = min(second.disk.dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
        root.unlink()
        root.with_suffix(".json").unlink(missing_ok=True)
        second.disk.close()
        third = GD.PrefixStore(4, disk=GD.PrefixDiskTier(tmp_path, 2 ** 30))
        got3, st3 = _run(model, target, follow, 12, third)
        assert got3 == cold                                              # whatever restored (or nothing), exact tokens
        third.disk.close()
    finally:
        target.close()


def test_prefix_disk_chain_parents_read_only_mla_rows(tmp_path, monkeypatch):
    """A chained restore materializes only the MLA token rows of each parent link: the parents' KDA state, drafter
    context and logits are superseded by the leaf and must stay unread (GLM-5.3: ~454 MB of KDA per link)."""
    import vmlx_engine.glm_dflash2 as GD
    monkeypatch.setattr(GD, "PREFIX_SNAPSHOT_EVERY", 16)
    monkeypatch.setenv("VMLX_SSD_ONLY_RAM_RELEASE", "1")
    model = _target(11)
    target = GLMDFlashTarget(model, TAPS)
    try:
        prompt = list(range(5, 75))
        follow = prompt + list(range(90, 99))
        cold, _ = _run(model, target, follow, 12, None)
        first = GD.PrefixStore(4, disk=GD.PrefixDiskTier(tmp_path, 2 ** 30))
        _run(model, target, prompt, 12, first)
        first.disk.flush()
        first.disk.close()
        second = GD.PrefixStore(4, disk=GD.PrefixDiskTier(tmp_path, 2 ** 30))
        calls = []
        orig = second.disk._read_link

        def spy(name, ids, leaf=True):
            out = orig(name, ids, leaf=leaf)
            calls.append((leaf, out[0]))
            return out
        second.disk._read_link = spy
        got, st = _run(model, target, follow, 12, second)
        assert st.cache_source == "disk" and got == cold
        parents = [link for leaf, link in calls if not leaf]
        assert calls[0][0] and len(parents) >= 3
        for arrays, layers, _ in parents:
            assert arrays and set(arrays) <= {f"c{i}.{j}" for i, e in enumerate(layers) if e[0] == "Glm5MLACache"
                                             for j in range(len(e[2]))}
            assert not {"hidden", "logits", "ids"} & set(arrays)
        second.disk.close()
    finally:
        target.close()


def test_adaptive_costs_are_shared_across_requests_on_one_target():
    """Round / AR costs belong to the model + kernels: a second request on the same target starts from the measured
    costs (no opening AR burst to re-measure t_ar); acceptance stays per request; another target has its own table;
    the first AR burst of a shared table is a warm-up and is not recorded."""
    from types import SimpleNamespace

    from vmlx_engine.glm_dflash2 import AdaptiveSpec, adaptive_costs

    tgt, other = SimpleNamespace(), SimpleNamespace()
    r1 = AdaptiveSpec(8, costs=adaptive_costs(tgt, 8))
    r1.update_ar(0.200, 8)                                  # first burst in the table: compile outlier, not stored
    assert r1.t_ar is None
    r1.update_ar(0.030, 8)
    r1.update_spec(4, 3, 0.900)                             # first round at block 4: compile outlier, not stored
    assert 4 not in r1.t_round
    for _ in range(10):
        r1.update_spec(4, 3, 0.050)
    assert abs(r1.t_round[4] - 0.050) < 1e-9
    r2 = AdaptiveSpec(8, costs=adaptive_costs(tgt, 8))
    assert r2.t_ar == r1.t_ar == 0.030 and 4 in r2.t_round
    assert r2.choose()[0] == "spec"                        # no opening AR burst
    assert r2.n_try == AdaptiveSpec(8).n_try               # acceptance starts from the prior again
    r3 = AdaptiveSpec(8, costs=adaptive_costs(other, 8))
    assert r3.t_ar is None and not r3.t_round
    assert adaptive_costs(tgt, 4) is not adaptive_costs(tgt, 8)


def test_adaptive_ar_cost_ignores_cut_bursts_and_reprobes_stale_estimates():
    """MiMo quality run 2026-10-16: the shared t_ar sat at 88 ms (real ~25) and AR was never re-measured."""
    from vmlx_engine.glm_dflash2 import AdaptiveSpec

    p = AdaptiveSpec(8, ar_reprobe_rounds=5)
    p.update_ar(0.025, 8)
    p.update_ar(0.300, 1)                                   # EOS after one token: launch overhead, ignored
    assert abs(p.t_ar - 0.025) < 1e-9
    p.t_ar = 0.088                                          # a stale estimate: spec always wins
    for _ in range(5):
        p.update_spec(3, 1, 0.064)
    mode, k = p.choose()
    assert (mode, k) == ("ar", 8)                           # re-probe after 5 spec rounds, normal burst length
    p.update_ar(0.025, 8)
    assert abs(p.t_ar - (0.3 * 0.088 + 0.7 * 0.025)) < 1e-9 # re-probe replaces most of the stale value
    assert p.choose()[0] in ("ar", "spec") and p.rounds_since_ar == 0


def test_adaptive_ar_cost_skips_a_single_stalled_burst():
    """quality-v2 pilot 2026-10-16: t_ar 22 -> 88 ms in one blend (a ~2 s burst); then AR was never chosen again."""
    from vmlx_engine.glm_dflash2 import AdaptiveSpec

    p = AdaptiveSpec(8, ar_reprobe_rounds=4)
    p.update_ar(0.022, 8)
    p.update_ar(0.240, 8)                                   # stalled burst: skipped
    assert abs(p.t_ar - 0.022) < 1e-9 and p.ar_outliers_skipped == 1
    p.update_ar(0.030, 8)                                   # ordinary drift is blended
    assert abs(p.t_ar - (0.7 * 0.022 + 0.3 * 0.030)) < 1e-9
    for _ in range(4):
        p.update_spec(3, 1, 0.064)
    assert p.choose() == ("ar", 8)
    p.update_ar(0.090, 8)                                   # a re-probe is always taken: real increases land
    assert p.t_ar > 0.06


class _SpreadDraft:
    """A non-DFlash2DraftModel drafter with a spread q (fixed logits per slot) for the fallback sampled path."""

    def __init__(self, block):
        self.config, self.calls = _Cfg(block), 0
        self.logits = mx.random.normal((1, block - 1, VOCAB), key=mx.random.key(5)) * 0.4

    def make_cache(self):
        return [_Cache()]

    def bind(self, target):
        return self

    def __call__(self, block, hidden, cache, logits_start=1):
        cache[0].offset += int(hidden.shape[1]) + block.shape[1]
        return self.logits[:, : block.shape[1] - 1]


def test_fallback_drafter_samples_its_drafts_from_q(monkeypatch):
    """Rejection sampling is exact only when the drafts are DRAWN from the q it is given; the fallback branch took
    the argmax (2026-10-16 audit). Spy on the rejection call: with a spread q the drafted tokens must not all be the
    argmax of q, and the share of argmax drafts must be near the mean max-q."""
    import dflash.model_mlx as DM

    seen = []
    real = DM._rejection_sample

    def spy(d_tok, t_prob, d_prob, d_idx=None):
        seen.append((np.array(d_tok[0]), np.array(mx.argmax(d_prob[0], axis=-1)), np.array(mx.max(d_prob[0], axis=-1))))
        return real(d_tok, t_prob, d_prob, d_idx)

    monkeypatch.setattr(DM, "_rejection_sample", spy)
    model = _target(7)
    target = GLMDFlashTarget(model, TAPS)
    try:
        mx.random.seed(0)
        list(generate_dflash2(target, _SpreadDraft(4), list(range(5, 30)), 60, eos=set(), temperature=1.0))
    finally:
        target.close()
    tok = np.concatenate([s[0] for s in seen]); top = np.concatenate([s[1] for s in seen])
    qmax = np.concatenate([s[2] for s in seen])
    share = float((tok == top).mean())
    assert len(tok) >= 30 and share < 0.9, (len(tok), share)
    assert abs(share - float(qmax.mean())) < 0.25, (share, float(qmax.mean()))



@pytest.mark.parametrize("gamma", [1, 3])
def test_rejection_sample_with_candidate_indices_is_exact(gamma):
    """The production selector path calls _rejection_sample with draft_indices (q over K candidates per slot). With the
    drafts DRAWN from that sparse q, the first emitted token (accepted draft, residual or bonus) must follow the
    target's p at that slot exactly. Synthetic p / q, so no model numerics; band = a perfect sampler's TV p99 x 1.25."""
    from dflash.model_mlx import _rejection_sample

    V, K, N = 32, 6, 6000
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(V) * 0.5, size=gamma + 1).astype(np.float32)          # target rows (gamma + 1, V)
    idx = np.stack([rng.choice(V, size=K, replace=False) for _ in range(gamma)])      # candidates per slot
    q = rng.dirichlet(np.ones(K), size=gamma).astype(np.float32)                       # draft probs over candidates
    tp, di, dq = mx.array(p[None]), mx.array(idx[None]), mx.array(q[None])
    counts = np.zeros(V)
    mx.random.seed(3)
    for _ in range(N):
        pick = [int(idx[s, rng.choice(K, p=q[s] / q[s].sum())]) for s in range(gamma)]
        accepted, bonus = _rejection_sample(mx.array([pick]), tp, dq, di)
        first = pick[0] if accepted >= 1 else bonus
        counts[first] += 1
    target = p[0].astype(np.float64) / p[0].sum()
    tv = 0.5 * np.abs(counts / N - target).sum()
    band = np.quantile([0.5 * np.abs(np.bincount(rng.choice(V, size=N, p=target), minlength=V) / N - target).sum()
                        for _ in range(40)], 0.99)
    assert tv <= 1.25 * band, (tv, band)


def test_shared_ar_cost_is_reprobed_across_short_requests():
    """2026-10-18 release audit R20 (MiMo, same clock 3/3 reps): the shared t_ar held 28.5-29 ms for the whole
    process (real pipelined step ~20 ms) because the re-probe counter was per request and no request reached 256
    rounds; prose then drafted at 46.6 tok/s where AR ran 50.4. The counter now lives in the shared table."""
    from types import SimpleNamespace

    from vmlx_engine.glm_dflash2 import AdaptiveSpec, adaptive_costs

    tgt = SimpleNamespace()
    costs = adaptive_costs(tgt, 8)
    first = AdaptiveSpec(8, costs=costs, ar_reprobe_rounds=10)
    first.update_ar(0.030, 8)                                # warm-up burst of the table
    first.update_ar(0.0285, 8)
    for _ in range(4):                                       # 4 + 4 + 3 rounds over three short requests
        first.update_spec(3, 1, 0.040)
    second = AdaptiveSpec(8, costs=costs, ar_reprobe_rounds=10)
    for _ in range(4):
        second.update_spec(3, 1, 0.040)
    third = AdaptiveSpec(8, costs=costs, ar_reprobe_rounds=10)
    for _ in range(3):
        third.update_spec(3, 1, 0.040)
    assert third.choose() == ("ar", 8)                       # re-probe due across requests
    third.update_ar(0.020, 8, requested=8)
    assert abs(third.t_ar - (0.3 * 0.0285 + 0.7 * 0.020)) < 1e-9
    assert costs["rounds_since_ar"] == 0


def test_burst_cut_before_its_planned_length_is_not_an_ar_sample():
    """The 28.5 ms seed was a 5-of-8-token burst ended by EOS at 6.8k context (above the 3-token floor)."""
    from vmlx_engine.glm_dflash2 import AdaptiveSpec

    p = AdaptiveSpec(8, ar_reprobe_rounds=4)
    p.update_ar(0.0285, 5, requested=8)                      # cut: no sample
    assert p.t_ar is None
    p.update_ar(0.020, 8, requested=8)
    assert abs(p.t_ar - 0.020) < 1e-9
    for _ in range(4):
        p.update_spec(3, 1, 0.040)
    assert p.choose() == ("ar", 8)
    p.update_ar(0.050, 3, requested=8)                       # the re-probe itself was cut ...
    assert abs(p.t_ar - 0.020) < 1e-9
    assert p.choose() == ("ar", 8)                           # ... so the next decision re-probes again
    p.update_ar(0.024, 8, requested=8)
    assert abs(p.t_ar - (0.3 * 0.020 + 0.7 * 0.024)) < 1e-9
